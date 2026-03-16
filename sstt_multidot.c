/*
 * sstt_multidot.c — Multi-Channel Dot Refinement
 *
 * The autopsy showed 407/414 errors happen at the refinement step,
 * where the current cascade uses PIXEL-ONLY ternary dot to rank top-K
 * candidates. The vote phase uses all channels (bytepacked); the dot
 * step throws two of them away.
 *
 * This file keeps the vote phase identical (bytepacked cascade) and
 * replaces the single-channel dot with:
 *
 *   score = w_px * dot(q_px, t_px)
 *         + w_hg * dot(q_hg, t_hg)
 *         + w_vg * dot(q_vg, t_vg)
 *
 * Then sweeps weights to find what the data prefers.
 *
 * Weight strategies tested:
 *   A. Pixel-only baseline     (w = 1, 0, 0)
 *   B. Equal                   (w = 1, 1, 1)
 *   C. IG-proportional         (w = total IG per channel, normalized)
 *   D. Empirical SNR           (w = signal/noise from training pairs)
 *   E. Grid search             (w_px=256, w_hg/w_vg ∈ {0..512} step 64)
 *
 * Build: make sstt_multidot
 */

#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define TRAIN_N         60000
#define TEST_N          10000
#define IMG_W           28
#define IMG_H           28
#define PIXELS          784
#define PADDED          800
#define N_CLASSES       10
#define CLS_PAD         16

#define BLKS_PER_ROW    9
#define N_BLOCKS        252
#define N_BVALS         27
#define SIG_PAD         256
#define BYTE_VALS       256

#define BG_PIXEL        0
#define BG_GRAD         13
#define BG_TRANS        13
#define BG_JOINT        20

#define H_TRANS_PER_ROW 8
#define N_HTRANS        (H_TRANS_PER_ROW * IMG_H)
#define V_TRANS_PER_COL 27
#define N_VTRANS        (BLKS_PER_ROW * V_TRANS_PER_COL)
#define TRANS_PAD       256

#define IG_SCALE        16
#define TOP_K           200

static const char *data_dir = "data/";

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ================================================================
 *  Data
 * ================================================================ */
static uint8_t *raw_train_img, *raw_test_img;
static uint8_t *train_labels, *test_labels;
static int8_t  *tern_train, *tern_test;
static int8_t  *hgrad_train, *hgrad_test, *vgrad_train, *vgrad_test;
static uint8_t *px_sigs_tr, *px_sigs_te;
static uint8_t *hg_sigs_tr, *hg_sigs_te;
static uint8_t *vg_sigs_tr, *vg_sigs_te;
static uint8_t *ht_sigs_tr, *ht_sigs_te;
static uint8_t *vt_sigs_tr, *vt_sigs_te;
static uint8_t *joint_sigs_tr, *joint_sigs_te;

/* Bytepacked index (vote phase) */
static uint32_t *joint_hot;
static uint16_t  ig_weights[N_BLOCKS];
static uint8_t   nbr_table[BYTE_VALS][8];   /* single bit-flip, original order */
static uint8_t   nbr_reord[BYTE_VALS][8];   /* single bit-flip, px+vg first */
static uint8_t   nbr_pair [BYTE_VALS][4];   /* pairwise px×vg diagonal probes */
static uint8_t   nbr_combo[BYTE_VALS][8];   /* 4 pairwise + 4 single px+vg */
static uint32_t  idx_off[N_BLOCKS][BYTE_VALS];
static uint16_t  idx_sz [N_BLOCKS][BYTE_VALS];
static uint32_t *idx_pool;

/* Per-channel IG for dot weighting */
static double chan_total_ig[3];   /* total IG per channel */

/* ================================================================
 *  Data loading (compact — errors crash)
 * ================================================================ */
static uint8_t *load_idx(const char *path, uint32_t *cnt,
                          uint32_t *ro, uint32_t *co) {
    FILE *f = fopen(path, "rb"); if (!f) { fprintf(stderr,"ERR: %s\n",path); exit(1); }
    uint32_t magic, n;
    fread(&magic,4,1,f); fread(&n,4,1,f);
    magic=__builtin_bswap32(magic); n=__builtin_bswap32(n); *cnt=n;
    size_t item=1;
    if ((magic&0xFF)>=3) {
        uint32_t r,c; fread(&r,4,1,f); fread(&c,4,1,f);
        r=__builtin_bswap32(r); c=__builtin_bswap32(c);
        if(ro)*ro=r; if(co)*co=c; item=(size_t)r*c;
    } else { if(ro)*ro=0; if(co)*co=0; }
    uint8_t *d=malloc((size_t)n*item); fread(d,1,(size_t)n*item,f); fclose(f); return d;
}
static void load_data(void) {
    uint32_t n,r,c; char p[256];
    snprintf(p,sizeof(p),"%strain-images-idx3-ubyte",data_dir); raw_train_img=load_idx(p,&n,&r,&c);
    snprintf(p,sizeof(p),"%strain-labels-idx1-ubyte",data_dir); train_labels =load_idx(p,&n,NULL,NULL);
    snprintf(p,sizeof(p),"%st10k-images-idx3-ubyte", data_dir); raw_test_img =load_idx(p,&n,&r,&c);
    snprintf(p,sizeof(p),"%st10k-labels-idx1-ubyte", data_dir); test_labels  =load_idx(p,&n,NULL,NULL);
}

/* ================================================================
 *  Features
 * ================================================================ */
static inline int8_t clamp_trit(int v){return v>0?1:v<0?-1:0;}

static void quantize_avx2(const uint8_t *src, int8_t *dst, int n) {
    const __m256i bias=_mm256_set1_epi8((char)0x80);
    const __m256i thi =_mm256_set1_epi8((char)(170^0x80));
    const __m256i tlo =_mm256_set1_epi8((char)(85^0x80));
    const __m256i one =_mm256_set1_epi8(1);
    for(int img=0;img<n;img++){
        const uint8_t*s=src+(size_t)img*PIXELS; int8_t*d=dst+(size_t)img*PADDED; int i;
        for(i=0;i+32<=PIXELS;i+=32){
            __m256i px=_mm256_loadu_si256((const __m256i*)(s+i));
            __m256i sp=_mm256_xor_si256(px,bias);
            _mm256_storeu_si256((__m256i*)(d+i),
                _mm256_sub_epi8(_mm256_and_si256(_mm256_cmpgt_epi8(sp,thi),one),
                                _mm256_and_si256(_mm256_cmpgt_epi8(tlo,sp),one)));
        }
        for(;i<PIXELS;i++) d[i]=s[i]>170?1:s[i]<85?-1:0;
        memset(d+PIXELS,0,PADDED-PIXELS);
    }
}
static void compute_gradients(const int8_t*t,int8_t*hg,int8_t*vg,int n){
    for(int img=0;img<n;img++){
        const int8_t*ti=t+(size_t)img*PADDED; int8_t*h=hg+(size_t)img*PADDED; int8_t*v=vg+(size_t)img*PADDED;
        for(int y=0;y<IMG_H;y++){for(int x=0;x<IMG_W-1;x++) h[y*IMG_W+x]=clamp_trit(ti[y*IMG_W+x+1]-ti[y*IMG_W+x]); h[y*IMG_W+IMG_W-1]=0;}
        memset(h+PIXELS,0,PADDED-PIXELS);
        for(int y=0;y<IMG_H-1;y++) for(int x=0;x<IMG_W;x++) v[y*IMG_W+x]=clamp_trit(ti[(y+1)*IMG_W+x]-ti[y*IMG_W+x]);
        memset(v+(IMG_H-1)*IMG_W,0,IMG_W); memset(v+PIXELS,0,PADDED-PIXELS);
    }
}
static inline uint8_t block_encode(int8_t t0,int8_t t1,int8_t t2){return(uint8_t)((t0+1)*9+(t1+1)*3+(t2+1));}
static void compute_block_sigs(const int8_t*data,uint8_t*sigs,int n){
    for(int i=0;i<n;i++){
        const int8_t*img=data+(size_t)i*PADDED; uint8_t*sig=sigs+(size_t)i*SIG_PAD;
        for(int y=0;y<IMG_H;y++) for(int s=0;s<BLKS_PER_ROW;s++){int b=y*IMG_W+s*3; sig[y*BLKS_PER_ROW+s]=block_encode(img[b],img[b+1],img[b+2]);}
        memset(sig+N_BLOCKS,0xFF,SIG_PAD-N_BLOCKS);
    }
}
static inline uint8_t trans_enc(uint8_t a,uint8_t b){
    int8_t a0=(a/9)-1,a1=((a/3)%3)-1,a2=(a%3)-1,b0=(b/9)-1,b1=((b/3)%3)-1,b2=(b%3)-1;
    return block_encode(clamp_trit(b0-a0),clamp_trit(b1-a1),clamp_trit(b2-a2));
}
static void compute_transitions(const uint8_t*bsig,int stride,uint8_t*ht,uint8_t*vt,int n){
    for(int i=0;i<n;i++){
        const uint8_t*s=bsig+(size_t)i*stride; uint8_t*h=ht+(size_t)i*TRANS_PAD; uint8_t*v=vt+(size_t)i*TRANS_PAD;
        for(int y=0;y<IMG_H;y++) for(int ss=0;ss<H_TRANS_PER_ROW;ss++) h[y*H_TRANS_PER_ROW+ss]=trans_enc(s[y*BLKS_PER_ROW+ss],s[y*BLKS_PER_ROW+ss+1]);
        memset(h+N_HTRANS,0xFF,TRANS_PAD-N_HTRANS);
        for(int y=0;y<V_TRANS_PER_COL;y++) for(int ss=0;ss<BLKS_PER_ROW;ss++) v[y*BLKS_PER_ROW+ss]=trans_enc(s[y*BLKS_PER_ROW+ss],s[(y+1)*BLKS_PER_ROW+ss]);
        memset(v+N_VTRANS,0xFF,TRANS_PAD-N_VTRANS);
    }
}
static uint8_t encode_d(uint8_t px,uint8_t hg,uint8_t vg,uint8_t ht,uint8_t vt){
    int ps=((px/9)-1)+(((px/3)%3)-1)+((px%3)-1);
    int hs=((hg/9)-1)+(((hg/3)%3)-1)+((hg%3)-1);
    int vs=((vg/9)-1)+(((vg/3)%3)-1)+((vg%3)-1);
    uint8_t pc=ps<0?0:ps==0?1:ps<3?2:3;
    uint8_t hc=hs<0?0:hs==0?1:hs<3?2:3;
    uint8_t vc=vs<0?0:vs==0?1:vs<3?2:3;
    return pc|(hc<<2)|(vc<<4)|((ht!=BG_TRANS)?1<<6:0)|((vt!=BG_TRANS)?1<<7:0);
}
static void compute_joint_sigs(uint8_t*out,int n,
    const uint8_t*px_s,const uint8_t*hg_s,const uint8_t*vg_s,
    const uint8_t*ht_s,const uint8_t*vt_s){
    for(int i=0;i<n;i++){
        const uint8_t*px=px_s+(size_t)i*SIG_PAD,*hg=hg_s+(size_t)i*SIG_PAD,*vg=vg_s+(size_t)i*SIG_PAD;
        const uint8_t*ht=ht_s+(size_t)i*TRANS_PAD,*vt=vt_s+(size_t)i*TRANS_PAD;
        uint8_t*os=out+(size_t)i*SIG_PAD;
        for(int y=0;y<IMG_H;y++) for(int s=0;s<BLKS_PER_ROW;s++){
            int k=y*BLKS_PER_ROW+s;
            uint8_t ht_bv=s>0?ht[y*H_TRANS_PER_ROW+(s-1)]:BG_TRANS;
            uint8_t vt_bv=y>0?vt[(y-1)*BLKS_PER_ROW+s]:BG_TRANS;
            os[k]=encode_d(px[k],hg[k],vg[k],ht_bv,vt_bv);
        }
        memset(os+N_BLOCKS,0xFF,SIG_PAD-N_BLOCKS);
    }
}

/* ================================================================
 *  Bytepacked hot map + IG + index (vote phase, unchanged)
 * ================================================================ */
static void build_vote_phase(uint8_t bg) {
    joint_hot=aligned_alloc(32,(size_t)N_BLOCKS*BYTE_VALS*CLS_PAD*4);
    memset(joint_hot,0,(size_t)N_BLOCKS*BYTE_VALS*CLS_PAD*4);
    for(int i=0;i<TRAIN_N;i++){
        int lbl=train_labels[i]; const uint8_t*sig=joint_sigs_tr+(size_t)i*SIG_PAD;
        for(int k=0;k<N_BLOCKS;k++) joint_hot[(size_t)k*BYTE_VALS*CLS_PAD+(size_t)sig[k]*CLS_PAD+lbl]++;
    }
    /* IG weights */
    int cc[N_CLASSES]={0}; for(int i=0;i<TRAIN_N;i++) cc[train_labels[i]]++;
    double h_class=0.0;
    for(int c=0;c<N_CLASSES;c++){double p=(double)cc[c]/TRAIN_N;if(p>0)h_class-=p*log2(p);}
    double raw[N_BLOCKS],mx=0.0;
    for(int k=0;k<N_BLOCKS;k++){
        double h_cond=0.0;
        for(int v=0;v<BYTE_VALS;v++){
            if((uint8_t)v==bg) continue;
            const uint32_t*h=joint_hot+(size_t)k*BYTE_VALS*CLS_PAD+(size_t)v*CLS_PAD;
            int vt=0; for(int c=0;c<N_CLASSES;c++) vt+=h[c]; if(!vt) continue;
            double pv=(double)vt/TRAIN_N,hv=0.0;
            for(int c=0;c<N_CLASSES;c++){double pc=(double)h[c]/vt;if(pc>0)hv-=pc*log2(pc);}
            h_cond+=pv*hv;
        }
        raw[k]=h_class-h_cond; if(raw[k]>mx) mx=raw[k];
    }
    for(int k=0;k<N_BLOCKS;k++){ig_weights[k]=mx>0?(uint16_t)(raw[k]/mx*IG_SCALE+0.5):1;if(!ig_weights[k])ig_weights[k]=1;}
    /* Original neighbor table: bit-flip order 0,1,2,3,4,5,6,7 */
    for(int v=0;v<BYTE_VALS;v++) for(int b=0;b<8;b++) nbr_table[v][b]=(uint8_t)(v^(1<<b));

    /* Reordered: pixel+vgrad bits first {0,1,4,5}, hgrad+trans last {2,3,6,7} */
    { int ord[]={0,1,4,5,2,3,6,7};
      for(int v=0;v<BYTE_VALS;v++) for(int i=0;i<8;i++) nbr_reord[v][i]=(uint8_t)(v^(1<<ord[i])); }

    /* Pairwise: jointly change pixel field (bits 0-1) AND vgrad field (bits 4-5).
     * Encoding D layout: [trans 7-6 | vg 5-4 | hg 3-2 | px 1-0]
     * 4 diagonal neighbors: (px±1, vg±1) mod 4, hg and trans fixed. */
    for(int v=0;v<BYTE_VALS;v++){
        uint8_t px=(uint8_t)(v&0x03), hg=(uint8_t)((v>>2)&0x03);
        uint8_t vg=(uint8_t)((v>>4)&0x03), tr=(uint8_t)((v>>6)&0x03);
        uint8_t pxp=(px+1)&0x03, pxm=(px-1)&0x03;
        uint8_t vgp=(vg+1)&0x03, vgm=(vg-1)&0x03;
        nbr_pair[v][0]=(uint8_t)((tr<<6)|(vgp<<4)|(hg<<2)|pxp); /* +,+ */
        nbr_pair[v][1]=(uint8_t)((tr<<6)|(vgm<<4)|(hg<<2)|pxp); /* +,- */
        nbr_pair[v][2]=(uint8_t)((tr<<6)|(vgp<<4)|(hg<<2)|pxm); /* -,+ */
        nbr_pair[v][3]=(uint8_t)((tr<<6)|(vgm<<4)|(hg<<2)|pxm); /* -,- */
    }

    /* Combo: 4 pairwise + 4 single-bit (px+vg) = 8 semantically targeted probes */
    for(int v=0;v<BYTE_VALS;v++){
        for(int i=0;i<4;i++) nbr_combo[v][i]  =nbr_pair[v][i];
        for(int i=0;i<4;i++) nbr_combo[v][4+i]=nbr_reord[v][i]; /* bits 0,1,4,5 */
    }
    /* Inverted index */
    memset(idx_sz,0,sizeof(idx_sz));
    for(int i=0;i<TRAIN_N;i++){const uint8_t*sig=joint_sigs_tr+(size_t)i*SIG_PAD;for(int k=0;k<N_BLOCKS;k++)if(sig[k]!=bg)idx_sz[k][sig[k]]++;}
    uint32_t total=0;
    for(int k=0;k<N_BLOCKS;k++) for(int v=0;v<BYTE_VALS;v++){idx_off[k][v]=total;total+=idx_sz[k][v];}
    idx_pool=malloc((size_t)total*sizeof(uint32_t));
    uint32_t(*wpos)[BYTE_VALS]=malloc((size_t)N_BLOCKS*BYTE_VALS*sizeof(uint32_t));
    memcpy(wpos,idx_off,(size_t)N_BLOCKS*BYTE_VALS*sizeof(uint32_t));
    for(int i=0;i<TRAIN_N;i++){const uint8_t*sig=joint_sigs_tr+(size_t)i*SIG_PAD;for(int k=0;k<N_BLOCKS;k++)if(sig[k]!=bg)idx_pool[wpos[k][sig[k]]++]=(uint32_t)i;}
    free(wpos);
}

/* ================================================================
 *  Per-channel IG (for dot weight strategy C)
 * ================================================================ */
static void compute_chan_ig_weights(void) {
    const uint8_t *chan_sigs[3] = {px_sigs_tr, hg_sigs_tr, vg_sigs_tr};
    const uint8_t bg_vals[3] = {BG_PIXEL, BG_GRAD, BG_GRAD};

    int cc[N_CLASSES]={0}; for(int i=0;i<TRAIN_N;i++) cc[train_labels[i]]++;
    double h_class=0.0;
    for(int c=0;c<N_CLASSES;c++){double p=(double)cc[c]/TRAIN_N;if(p>0)h_class-=p*log2(p);}

    for(int ch=0;ch<3;ch++){
        const uint8_t*sigs=chan_sigs[ch]; uint8_t bg=bg_vals[ch];
        double total=0.0;
        for(int k=0;k<N_BLOCKS;k++){
            int counts[N_BVALS][N_CLASSES],vt[N_BVALS];
            memset(counts,0,sizeof(counts)); memset(vt,0,sizeof(vt));
            for(int i=0;i<TRAIN_N;i++){uint8_t v=sigs[(size_t)i*SIG_PAD+k];counts[v][train_labels[i]]++;vt[v]++;}
            double h_cond=0.0;
            for(int v=0;v<N_BVALS;v++){
                if(!vt[v]||v==bg) continue;
                double pv=(double)vt[v]/TRAIN_N,hv=0.0;
                for(int c=0;c<N_CLASSES;c++){double pc=(double)counts[v][c]/vt[v];if(pc>0)hv-=pc*log2(pc);}
                h_cond+=pv*hv;
            }
            total+=(h_class-h_cond);
        }
        chan_total_ig[ch]=total;
    }
    printf("  Channel total IG:  px=%.4f  hg=%.4f  vg=%.4f\n",
           chan_total_ig[0], chan_total_ig[1], chan_total_ig[2]);
}

/* ================================================================
 *  Empirical SNR per channel (strategy D)
 *  SNR = (mean same-class dot - mean diff-class dot) / std diff-class dot
 * ================================================================ */
static void compute_chan_snr(double *snr_out) {
    const int8_t *chan_data[3] = {tern_train, hgrad_train, vgrad_train};
    const int PAIRS = 2000;

    for(int ch=0;ch<3;ch++){
        const int8_t*data=chan_data[ch];
        double same_sum=0,diff_sum=0,diff_sq=0;
        /* same-class pairs */
        for(int p=0;p<PAIRS;p++){
            int i=rand()%TRAIN_N, j;
            do { j=rand()%TRAIN_N; } while(train_labels[j]!=train_labels[i]||j==i);
            double d=0;
            const int8_t*a=data+(size_t)i*PADDED,*b=data+(size_t)j*PADDED;
            for(int x=0;x<PIXELS;x++) d+=a[x]*b[x];
            same_sum+=d;
        }
        /* diff-class pairs */
        for(int p=0;p<PAIRS;p++){
            int i=rand()%TRAIN_N, j;
            do { j=rand()%TRAIN_N; } while(train_labels[j]==train_labels[i]);
            double d=0;
            const int8_t*a=data+(size_t)i*PADDED,*b=data+(size_t)j*PADDED;
            for(int x=0;x<PIXELS;x++) d+=a[x]*b[x];
            diff_sum+=d; diff_sq+=d*d;
        }
        double mean_same=same_sum/PAIRS, mean_diff=diff_sum/PAIRS;
        double var_diff=diff_sq/PAIRS-mean_diff*mean_diff;
        snr_out[ch]=(mean_same-mean_diff)/(sqrt(var_diff)+1e-9);
    }
    printf("  Channel SNR:  px=%.2f  hg=%.2f  vg=%.2f\n",
           snr_out[0], snr_out[1], snr_out[2]);
}

/* ================================================================
 *  AVX2 ternary dot
 * ================================================================ */
static inline int32_t ternary_dot(const int8_t*a,const int8_t*b){
    __m256i acc=_mm256_setzero_si256();
    for(int i=0;i<PADDED;i+=32)
        acc=_mm256_add_epi8(acc,_mm256_sign_epi8(
            _mm256_load_si256((const __m256i*)(a+i)),
            _mm256_load_si256((const __m256i*)(b+i))));
    __m256i lo=_mm256_cvtepi8_epi16(_mm256_castsi256_si128(acc));
    __m256i hi=_mm256_cvtepi8_epi16(_mm256_extracti128_si256(acc,1));
    __m256i s32=_mm256_madd_epi16(_mm256_add_epi16(lo,hi),_mm256_set1_epi16(1));
    __m128i s=_mm_add_epi32(_mm256_castsi256_si128(s32),_mm256_extracti128_si256(s32,1));
    s=_mm_hadd_epi32(s,s); s=_mm_hadd_epi32(s,s);
    return _mm_cvtsi128_si32(s);
}

/* ================================================================
 *  Candidate with 3-channel dots
 * ================================================================ */
typedef struct {
    uint32_t id;
    uint32_t votes;
    int32_t  dot_px, dot_hg, dot_vg;
    int64_t  combined;   /* weighted sum, set per run */
} cand_t;

static int cmp_votes_d(const void*a,const void*b){return(int)((const cand_t*)b)->votes-(int)((const cand_t*)a)->votes;}
static int cmp_combined_d(const void*a,const void*b){
    int64_t da=((const cand_t*)a)->combined, db=((const cand_t*)b)->combined;
    return (db>da)-(db<da);
}

/* Pre-allocated histogram — eliminates per-image calloc/free.
 * Zero only touched entries on cleanup (second votes scan)
 * instead of memset of the whole buffer. */
/* Pre-allocated histogram — eliminates per-image calloc cold-miss penalty.
 * calloc returns freshly mapped pages (cold to cache); pre-alloc keeps
 * hist warm in L2/L3 across images.
 * Reset via memset of only [0..mx]: sequential write, ~1μs for 32KB,
 * vs random-branch zero-pass (~38μs) or cold calloc (~26μs miss penalty). */
static int   *g_hist     = NULL;
static size_t g_hist_cap = 0;

static int select_top_k(const uint32_t*votes,int n,cand_t*out,int k){
    uint32_t mx=0; for(int i=0;i<n;i++) if(votes[i]>mx) mx=votes[i];
    if(!mx) return 0;
    if((size_t)(mx+1)>g_hist_cap){
        g_hist_cap=(size_t)(mx+1)+4096;
        free(g_hist); g_hist=malloc(g_hist_cap*sizeof(int));
    }
    memset(g_hist,0,(mx+1)*sizeof(int));          /* sequential: ~1μs for 32KB */
    for(int i=0;i<n;i++) if(votes[i]) g_hist[votes[i]]++;
    int cum=0,thr; for(thr=(int)mx;thr>=1;thr--){cum+=g_hist[thr];if(cum>=k)break;}
    if(thr<1)thr=1;
    int nc=0;
    for(int i=0;i<n&&nc<k;i++)
        if(votes[i]>=(uint32_t)thr){out[nc]=(cand_t){(uint32_t)i,votes[i],0,0,0,0};nc++;}
    qsort(out,(size_t)nc,sizeof(cand_t),cmp_votes_d);
    return nc;
}

/* Run one test configuration, return accuracy */
static int run_config(const cand_t *precomputed_cands, const int *nc_arr,
                       int w_px, int w_hg, int w_vg) {
    int correct = 0;
    for(int i=0;i<TEST_N;i++){
        int nc=nc_arr[i];
        /* Copy candidates and apply weights */
        cand_t cands[TOP_K];
        memcpy(cands, precomputed_cands + (size_t)i*TOP_K, (size_t)nc*sizeof(cand_t));
        for(int j=0;j<nc;j++)
            cands[j].combined = (int64_t)w_px*cands[j].dot_px
                               +(int64_t)w_hg*cands[j].dot_hg
                               +(int64_t)w_vg*cands[j].dot_vg;
        qsort(cands,(size_t)nc,sizeof(cand_t),cmp_combined_d);
        int votes[N_CLASSES]={0}; int k3=nc<3?nc:3;
        for(int j=0;j<k3;j++) votes[train_labels[cands[j].id]]++;
        int pred=0; for(int c=1;c<N_CLASSES;c++) if(votes[c]>votes[pred]) pred=c;
        if(pred==test_labels[i]) correct++;
    }
    return correct;
}

/* Confusion matrix deltas vs baseline */
static void print_pair_deltas(const cand_t *pre, const int *nc_arr,
                               int w_px, int w_hg, int w_vg,
                               const uint8_t *baseline_preds) {
    int conf_base[N_CLASSES][N_CLASSES]={};
    int conf_new [N_CLASSES][N_CLASSES]={};
    for(int i=0;i<TEST_N;i++){
        int nc=nc_arr[i];
        cand_t cands[TOP_K];
        memcpy(cands,pre+(size_t)i*TOP_K,(size_t)nc*sizeof(cand_t));
        for(int j=0;j<nc;j++)
            cands[j].combined=(int64_t)w_px*cands[j].dot_px
                              +(int64_t)w_hg*cands[j].dot_hg
                              +(int64_t)w_vg*cands[j].dot_vg;
        qsort(cands,(size_t)nc,sizeof(cand_t),cmp_combined_d);
        int votes[N_CLASSES]={0}; int k3=nc<3?nc:3;
        for(int j=0;j<k3;j++) votes[train_labels[cands[j].id]]++;
        int pred=0; for(int c=1;c<N_CLASSES;c++) if(votes[c]>votes[pred]) pred=c;
        conf_base[test_labels[i]][(int)baseline_preds[i]]++;
        conf_new [test_labels[i]][pred]++;
    }
    printf("  Per-class recall delta (new - baseline):\n  ");
    for(int c=0;c<N_CLASSES;c++){
        int base_wrong=0,new_wrong=0;
        for(int p=0;p<N_CLASSES;p++){if(p!=c){base_wrong+=conf_base[c][p];new_wrong+=conf_new[c][p];}}
        int delta=base_wrong-new_wrong;
        printf("  %d:%+d", c, delta);
    }
    printf("\n  Top confusion pairs (true→pred): baseline → new\n");
    typedef struct{int a,b,base,nw;} pair_t;
    pair_t pairs[90]; int np=0;
    for(int a=0;a<N_CLASSES;a++) for(int b=0;b<N_CLASSES;b++) if(a!=b&&(conf_base[a][b]>0||conf_new[a][b]>0))
        pairs[np++]=(pair_t){a,b,conf_base[a][b],conf_new[a][b]};
    for(int i=0;i<np-1;i++) for(int j=i+1;j<np;j++) if(pairs[j].base>pairs[i].base){pair_t t=pairs[i];pairs[i]=pairs[j];pairs[j]=t;}
    for(int i=0;i<np&&i<12;i++) if(pairs[i].base>1||pairs[i].nw>1){
        int d=pairs[i].base-pairs[i].nw;
        printf("    %d\xe2\x86\x92%d:  %d \xe2\x86\x92 %d  (%+d)\n",pairs[i].a,pairs[i].b,pairs[i].base,pairs[i].nw,d);
    }
}

/* ================================================================
 *  Generic probe strategy runner
 *  Runs vote+3-dot precompute with arbitrary probe table, applies
 *  best dot weights, returns accuracy and wall time.
 * ================================================================ */
static int run_strategy(const uint8_t (*tbl)[8], int np, uint8_t bg,
                         cand_t *pre, int *nc_arr, uint32_t *votes,
                         int w_px, int w_hg, int w_vg,
                         double *t_out) {
    double ts = now_sec();
    for(int i=0;i<TEST_N;i++){
        memset(votes,0,TRAIN_N*sizeof(uint32_t));
        const uint8_t*sig=joint_sigs_te+(size_t)i*SIG_PAD;
        for(int k=0;k<N_BLOCKS;k++){
            uint8_t bv=sig[k]; if(bv==bg) continue;
            uint16_t w=ig_weights[k],wh=w>1?w/2:1;
            {uint32_t off=idx_off[k][bv];uint16_t sz=idx_sz[k][bv];
             const uint32_t*ids=idx_pool+off;
             for(uint16_t j=0;j<sz;j++) votes[ids[j]]+=w;}
            for(int nb=0;nb<np;nb++){
                uint8_t nv=tbl[bv][nb]; if(nv==bg) continue;
                uint32_t noff=idx_off[k][nv];uint16_t nsz=idx_sz[k][nv];
                const uint32_t*nids=idx_pool+noff;
                for(uint16_t j=0;j<nsz;j++) votes[nids[j]]+=wh;
            }
        }
        cand_t *row=pre+(size_t)i*TOP_K;
        int nc=select_top_k(votes,TRAIN_N,row,TOP_K);
        nc_arr[i]=nc;
        const int8_t *qp=tern_test+(size_t)i*PADDED;
        const int8_t *qh=hgrad_test+(size_t)i*PADDED;
        const int8_t *qv=vgrad_test+(size_t)i*PADDED;
        for(int j=0;j<nc;j++){
            uint32_t id=row[j].id;
            row[j].dot_px=ternary_dot(qp,tern_train+(size_t)id*PADDED);
            row[j].dot_hg=ternary_dot(qh,hgrad_train+(size_t)id*PADDED);
            row[j].dot_vg=ternary_dot(qv,vgrad_train+(size_t)id*PADDED);
        }
    }
    *t_out=now_sec()-ts;
    return run_config(pre,nc_arr,w_px,w_hg,w_vg);
}

/* ================================================================
 *  Main
 * ================================================================ */
int main(int argc, char **argv) {
    double t0=now_sec(); srand(42);
    if(argc>1){
        data_dir=argv[1]; size_t len=strlen(data_dir);
        if(len&&data_dir[len-1]!='/'){char*buf=malloc(len+2);memcpy(buf,data_dir,len);buf[len]='/';buf[len+1]='\0';data_dir=buf;}
    }
    const char*ds=strstr(data_dir,"fashion")?"Fashion-MNIST":"MNIST";
    printf("=== SSTT Multi-Channel Dot Refinement (%s) ===\n\n",ds);

    load_data();
    tern_train =aligned_alloc(32,(size_t)TRAIN_N*PADDED); tern_test  =aligned_alloc(32,(size_t)TEST_N *PADDED);
    hgrad_train=aligned_alloc(32,(size_t)TRAIN_N*PADDED); hgrad_test =aligned_alloc(32,(size_t)TEST_N *PADDED);
    vgrad_train=aligned_alloc(32,(size_t)TRAIN_N*PADDED); vgrad_test =aligned_alloc(32,(size_t)TEST_N *PADDED);
    quantize_avx2(raw_train_img,tern_train,TRAIN_N); quantize_avx2(raw_test_img,tern_test,TEST_N);
    compute_gradients(tern_train,hgrad_train,vgrad_train,TRAIN_N);
    compute_gradients(tern_test, hgrad_test, vgrad_test, TEST_N);

    px_sigs_tr=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD); px_sigs_te=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    hg_sigs_tr=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD); hg_sigs_te=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    vg_sigs_tr=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD); vg_sigs_te=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    compute_block_sigs(tern_train, px_sigs_tr,TRAIN_N); compute_block_sigs(tern_test, px_sigs_te,TEST_N);
    compute_block_sigs(hgrad_train,hg_sigs_tr,TRAIN_N); compute_block_sigs(hgrad_test,hg_sigs_te,TEST_N);
    compute_block_sigs(vgrad_train,vg_sigs_tr,TRAIN_N); compute_block_sigs(vgrad_test,vg_sigs_te,TEST_N);

    ht_sigs_tr=aligned_alloc(32,(size_t)TRAIN_N*TRANS_PAD); ht_sigs_te=aligned_alloc(32,(size_t)TEST_N*TRANS_PAD);
    vt_sigs_tr=aligned_alloc(32,(size_t)TRAIN_N*TRANS_PAD); vt_sigs_te=aligned_alloc(32,(size_t)TEST_N*TRANS_PAD);
    compute_transitions(px_sigs_tr,SIG_PAD,ht_sigs_tr,vt_sigs_tr,TRAIN_N);
    compute_transitions(px_sigs_te,SIG_PAD,ht_sigs_te,vt_sigs_te,TEST_N);
    joint_sigs_tr=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD); joint_sigs_te=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    compute_joint_sigs(joint_sigs_tr,TRAIN_N,px_sigs_tr,hg_sigs_tr,vg_sigs_tr,ht_sigs_tr,vt_sigs_tr);
    compute_joint_sigs(joint_sigs_te,TEST_N, px_sigs_te,hg_sigs_te,vg_sigs_te,ht_sigs_te,vt_sigs_te);

    /* Background */
    long vc[BYTE_VALS]={0};
    for(int i=0;i<TRAIN_N;i++){const uint8_t*s=joint_sigs_tr+(size_t)i*SIG_PAD;for(int k=0;k<N_BLOCKS;k++)vc[s[k]]++;}
    uint8_t bg=0; long mc=0; for(int v=0;v<BYTE_VALS;v++) if(vc[v]>mc){mc=vc[v];bg=(uint8_t)v;}

    printf("Building vote phase...\n");
    build_vote_phase(bg);
    printf("Computing per-channel IG...\n");
    compute_chan_ig_weights();
    printf("Computing empirical SNR...\n");
    double snr[3]; compute_chan_snr(snr);
    printf("  Setup (%.2f sec)\n\n", now_sec()-t0);

    /* ----------------------------------------------------------------
     * PHASE 1: Probe sweep — run vote+dot precompute for each probe
     * count to measure accuracy vs speed tradeoff.
     * Best dot weights from grid search: px=256, hg=0, vg=192.
     * ---------------------------------------------------------------- */
    cand_t   *pre    = malloc((size_t)TEST_N * TOP_K * sizeof(cand_t));
    int      *nc_arr = malloc((size_t)TEST_N * sizeof(int));
    uint32_t *votes  = calloc(TRAIN_N, sizeof(uint32_t));

    int probe_counts[] = {0, 1, 2, 4, 8};
    int n_probe_tests  = 5;
    /* Best dot weights from grid search */
    int best_wpx = 256, best_whg = 0, best_wvg = 192;

    printf("=== Probe Sweep (accuracy + speed vs neighbor count) ===\n");
    printf("  %-8s  %-8s  %-10s  %-10s\n",
           "Probes", "Acc%", "VoteTime", "TotalTime");
    printf("  %-8s  %-8s  %-10s  %-10s\n",
           "-------", "-------", "---------", "---------");

    double t_vote_8 = 0, t_total_8 = 0;  /* store 8-probe baseline for reference */

    for(int pi = 0; pi < n_probe_tests; pi++){
        int np = probe_counts[pi];
        double t_start = now_sec();

        for(int i=0;i<TEST_N;i++){
            memset(votes,0,TRAIN_N*sizeof(uint32_t));
            const uint8_t*sig=joint_sigs_te+(size_t)i*SIG_PAD;
            for(int k=0;k<N_BLOCKS;k++){
                uint8_t bv=sig[k]; if(bv==bg) continue;
                uint16_t w=ig_weights[k],wh=w>1?w/2:1;
                {uint32_t off=idx_off[k][bv];uint16_t sz=idx_sz[k][bv];
                 const uint32_t*ids=idx_pool+off;
                 for(uint16_t j=0;j<sz;j++) votes[ids[j]]+=w;}
                for(int nb=0;nb<np;nb++){
                    uint8_t nv=nbr_table[bv][nb]; if(nv==bg) continue;
                    uint32_t noff=idx_off[k][nv]; uint16_t nsz=idx_sz[k][nv];
                    const uint32_t*nids=idx_pool+noff;
                    for(uint16_t j=0;j<nsz;j++) votes[nids[j]]+=wh;
                }
            }
            cand_t *row = pre + (size_t)i*TOP_K;
            int nc = select_top_k(votes, TRAIN_N, row, TOP_K);
            nc_arr[i] = nc;

            const int8_t *q_px = tern_test  + (size_t)i*PADDED;
            const int8_t *q_hg = hgrad_test + (size_t)i*PADDED;
            const int8_t *q_vg = vgrad_test + (size_t)i*PADDED;
            for(int j=0;j<nc;j++){
                uint32_t id=row[j].id;
                row[j].dot_px=ternary_dot(q_px, tern_train  +(size_t)id*PADDED);
                row[j].dot_hg=ternary_dot(q_hg, hgrad_train +(size_t)id*PADDED);
                row[j].dot_vg=ternary_dot(q_vg, vgrad_train +(size_t)id*PADDED);
            }
        }
        double t_total = now_sec() - t_start;
        /* vote-only time: subtract dot cost (≈ 0.093 sec, measured) */
        double t_vote_est = t_total - 0.093;

        int correct = run_config(pre, nc_arr, best_wpx, best_whg, best_wvg);
        double acc = 100.0 * correct / TEST_N;

        if(np == 8){ t_vote_8 = t_vote_est; t_total_8 = t_total; }

        printf("  %-8d  %6.2f%%  %6.2f sec    %6.2f sec%s\n",
               np, acc, t_vote_est, t_total,
               np==8 ? "  ← baseline" : "");
    }

    printf("\nProbe speedup summary (vs 8 probes):\n");
    printf("  Each probe removed saves ~%.0f μs/img  (%.3f sec/10K)\n",
           (t_vote_8 - 0.0) / TEST_N * 1e6 / 8,
           (t_vote_8) / 8);

    /* Re-run 8 probes to populate pre/nc_arr for weight strategy section */
    printf("\nRunning vote phase (8 probes) + precomputing 3-channel dots...\n");
    double t_vote = now_sec();
    for(int i=0;i<TEST_N;i++){
        memset(votes,0,TRAIN_N*sizeof(uint32_t));
        const uint8_t*sig=joint_sigs_te+(size_t)i*SIG_PAD;
        for(int k=0;k<N_BLOCKS;k++){
            uint8_t bv=sig[k]; if(bv==bg) continue;
            uint16_t w=ig_weights[k],wh=w>1?w/2:1;
            {uint32_t off=idx_off[k][bv];uint16_t sz=idx_sz[k][bv];
             const uint32_t*ids=idx_pool+off;for(uint16_t j=0;j<sz;j++)votes[ids[j]]+=w;}
            for(int nb=0;nb<8;nb++){
                uint8_t nv=nbr_table[bv][nb];if(nv==bg)continue;
                uint32_t noff=idx_off[k][nv];uint16_t nsz=idx_sz[k][nv];
                const uint32_t*nids=idx_pool+noff;
                for(uint16_t j=0;j<nsz;j++)votes[nids[j]]+=wh;
            }
        }
        cand_t *row = pre + (size_t)i*TOP_K;
        int nc = select_top_k(votes, TRAIN_N, row, TOP_K);
        nc_arr[i] = nc;
        const int8_t *q_px=tern_test+(size_t)i*PADDED;
        const int8_t *q_hg=hgrad_test+(size_t)i*PADDED;
        const int8_t *q_vg=vgrad_test+(size_t)i*PADDED;
        for(int j=0;j<nc;j++){
            uint32_t id=row[j].id;
            row[j].dot_px=ternary_dot(q_px,tern_train+(size_t)id*PADDED);
            row[j].dot_hg=ternary_dot(q_hg,hgrad_train+(size_t)id*PADDED);
            row[j].dot_vg=ternary_dot(q_vg,vgrad_train+(size_t)id*PADDED);
        }
        if((i+1)%2000==0) fprintf(stderr,"  %d/%d\r",i+1,TEST_N);
    }
    fprintf(stderr,"\n");
    free(votes);
    printf("  Vote + 3-dot precompute: %.2f sec\n\n", now_sec()-t_vote);

    /* ----------------------------------------------------------------
     * PHASE 2: Weight strategies — all free after precompute
     * ---------------------------------------------------------------- */

    /* Strategy A: pixel only (baseline) */
    uint8_t baseline_preds[TEST_N];
    {
        int c=run_config(pre,nc_arr,256,0,0);
        printf("--- A: Pixel-only (baseline) ---\n");
        printf("  Accuracy: %.2f%%  (%d errors)\n\n", 100.0*c/TEST_N, TEST_N-c);
        /* Re-run to capture predictions */
        for(int i=0;i<TEST_N;i++){
            int nc=nc_arr[i]; cand_t cands[TOP_K];
            memcpy(cands,pre+(size_t)i*TOP_K,(size_t)nc*sizeof(cand_t));
            for(int j=0;j<nc;j++) cands[j].combined=cands[j].dot_px;
            qsort(cands,(size_t)nc,sizeof(cand_t),cmp_combined_d);
            int kv[N_CLASSES]={0}; int k3=nc<3?nc:3;
            for(int j=0;j<k3;j++) kv[train_labels[cands[j].id]]++;
            int pred=0; for(int cl=1;cl<N_CLASSES;cl++) if(kv[cl]>kv[pred]) pred=cl;
            baseline_preds[i]=(uint8_t)pred;
        }
    }

    /* Strategy B: equal weights */
    {
        int c=run_config(pre,nc_arr,256,256,256);
        printf("--- B: Equal weights (px=hg=vg) ---\n");
        printf("  Accuracy: %.2f%%  (%d errors, %+d vs baseline)\n",
               100.0*c/TEST_N, TEST_N-c, (TEST_N-c)-(TEST_N-run_config(pre,nc_arr,256,0,0)));
        print_pair_deltas(pre,nc_arr,256,256,256,baseline_preds);
        printf("\n");
    }

    /* Strategy C: IG-proportional */
    {
        double sum_ig=chan_total_ig[0]+chan_total_ig[1]+chan_total_ig[2];
        int w_px=(int)(256*chan_total_ig[0]/sum_ig+0.5);
        int w_hg=(int)(256*chan_total_ig[1]/sum_ig+0.5);
        int w_vg=(int)(256*chan_total_ig[2]/sum_ig+0.5);
        int c=run_config(pre,nc_arr,w_px,w_hg,w_vg);
        printf("--- C: IG-proportional (px=%d hg=%d vg=%d) ---\n",w_px,w_hg,w_vg);
        printf("  Accuracy: %.2f%%  (%d errors)\n",100.0*c/TEST_N,TEST_N-c);
        print_pair_deltas(pre,nc_arr,w_px,w_hg,w_vg,baseline_preds);
        printf("\n");
    }

    /* Strategy D: SNR-weighted */
    {
        double sum_snr=snr[0]+snr[1]+snr[2];
        int w_px=(int)(256*snr[0]/sum_snr+0.5);
        int w_hg=(int)(256*snr[1]/sum_snr+0.5);
        int w_vg=(int)(256*snr[2]/sum_snr+0.5);
        int c=run_config(pre,nc_arr,w_px,w_hg,w_vg);
        printf("--- D: Empirical SNR (px=%d hg=%d vg=%d) ---\n",w_px,w_hg,w_vg);
        printf("  Accuracy: %.2f%%  (%d errors)\n",100.0*c/TEST_N,TEST_N-c);
        print_pair_deltas(pre,nc_arr,w_px,w_hg,w_vg,baseline_preds);
        printf("\n");
    }

    /* Strategy E: grid search — w_px=256 fixed, sweep w_hg and w_vg */
    printf("--- E: Grid search (w_px=256 fixed, w_hg/w_vg swept) ---\n");
    int grid_vals[]={0,64,128,192,256,320,384,448,512};
    int n_grid=9;
    int best_correct=0, best_whg_grid=0, best_wvg_grid=0; (void)t_vote_8; (void)t_total_8;
    printf("  w_hg\\w_vg");
    for(int vi=0;vi<n_grid;vi++) printf(" %4d",grid_vals[vi]);
    printf("\n  ---------");
    for(int vi=0;vi<n_grid;vi++) (void)vi,printf("-----");
    printf("\n");
    for(int hi=0;hi<n_grid;hi++){
        printf("  %9d",grid_vals[hi]);
        for(int vi=0;vi<n_grid;vi++){
            int c=run_config(pre,nc_arr,256,grid_vals[hi],grid_vals[vi]);
            printf(" %4.1f",100.0*c/TEST_N);
            if(c>best_correct){best_correct=c;best_whg_grid=grid_vals[hi];best_wvg_grid=grid_vals[vi];}
        }
        printf("\n");
    }
    printf("\n  Best: w_px=256 w_hg=%d w_vg=%d → %.2f%%  (%d errors)\n\n",
           best_whg_grid, best_wvg_grid, 100.0*best_correct/TEST_N, TEST_N-best_correct);

    /* Deep dive on best config */
    printf("--- Best configuration detail ---\n");
    print_pair_deltas(pre,nc_arr,256,best_whg_grid,best_wvg_grid,baseline_preds);

    printf("\n=== SUMMARY ===\n");
    printf("  A. Pixel-only baseline:   %.2f%%\n", 100.0*run_config(pre,nc_arr,256,0,0)/TEST_N);
    printf("  B. Equal (1,1,1):         %.2f%%\n", 100.0*run_config(pre,nc_arr,256,256,256)/TEST_N);
    {double s=chan_total_ig[0]+chan_total_ig[1]+chan_total_ig[2];
     printf("  C. IG-prop:               %.2f%%\n", 100.0*run_config(pre,nc_arr,(int)(256*chan_total_ig[0]/s),(int)(256*chan_total_ig[1]/s),(int)(256*chan_total_ig[2]/s))/TEST_N);}
    {double s=snr[0]+snr[1]+snr[2];
     printf("  D. SNR-weighted:          %.2f%%\n", 100.0*run_config(pre,nc_arr,(int)(256*snr[0]/s),(int)(256*snr[1]/s),(int)(256*snr[2]/s))/TEST_N);}
    printf("  E. Grid best (px=256,hg=%d,vg=%d): %.2f%%\n",
           best_whg_grid, best_wvg_grid, 100.0*best_correct/TEST_N);
    /* ----------------------------------------------------------------
     * Probe Type Comparison — same budget (4 or 8 probes), different
     * neighbor selection strategies.
     * Best dot weights: px=256, hg=0, vg=192 throughout.
     * ---------------------------------------------------------------- */
    printf("\n=== Probe Type Comparison (px=256 hg=0 vg=192) ===\n");
    printf("  %-40s  %-8s  %-10s\n", "Strategy", "Acc%", "Time");
    printf("  %-40s  %-8s  %-10s\n",
           "----------------------------------------","--------","----------");

    uint32_t *votes2 = calloc(TRAIN_N, sizeof(uint32_t));
    double t_strat;
    int    c_strat;

    /* 1. Original 4 probes: bit-flip order {0,1,2,3} = px+hg */
    c_strat=run_strategy(nbr_table,4,bg,pre,nc_arr,votes2,256,0,192,&t_strat);
    printf("  %-40s  %6.2f%%  %6.2f sec\n",
           "4-probe original {0,1,2,3} px+hg",100.0*c_strat/TEST_N,t_strat);

    /* 2. Reordered 4 probes: bit-flip order {0,1,4,5} = px+vg  */
    c_strat=run_strategy(nbr_reord,4,bg,pre,nc_arr,votes2,256,0,192,&t_strat);
    printf("  %-40s  %6.2f%%  %6.2f sec\n",
           "4-probe reordered {0,1,4,5} px+vg",100.0*c_strat/TEST_N,t_strat);

    /* 3. Pairwise 4 probes: diagonal (px±1, vg±1) joint */
    c_strat=run_strategy(nbr_pair,4,bg,pre,nc_arr,votes2,256,0,192,&t_strat);
    printf("  %-40s  %6.2f%%  %6.2f sec\n",
           "4-probe pairwise (px±1,vg±1) diagonal",100.0*c_strat/TEST_N,t_strat);

    /* 4. Original 8 probes: baseline */
    c_strat=run_strategy(nbr_table,8,bg,pre,nc_arr,votes2,256,0,192,&t_strat);
    printf("  %-40s  %6.2f%%  %6.2f sec  ← current best\n",
           "8-probe original (baseline)",100.0*c_strat/TEST_N,t_strat);

    /* 5. Reordered 8 probes: px+vg first */
    c_strat=run_strategy(nbr_reord,8,bg,pre,nc_arr,votes2,256,0,192,&t_strat);
    printf("  %-40s  %6.2f%%  %6.2f sec\n",
           "8-probe reordered px+vg first",100.0*c_strat/TEST_N,t_strat);

    /* 6. Combo 8 probes: 4 pairwise + 4 single px+vg */
    c_strat=run_strategy(nbr_combo,8,bg,pre,nc_arr,votes2,256,0,192,&t_strat);
    printf("  %-40s  %6.2f%%  %6.2f sec\n",
           "8-probe combo (4 pairwise + 4 single)",100.0*c_strat/TEST_N,t_strat);

    /* Per-pair breakdown on best new strategy */
    printf("\n--- Per-pair deltas (vs original 8-probe baseline) ---\n");
    /* Re-run 8-probe to get baseline_preds for this section */
    { run_strategy(nbr_table,8,bg,pre,nc_arr,votes2,256,0,192,&t_strat);
      for(int i=0;i<TEST_N;i++){
          int nc=nc_arr[i]; cand_t cands[TOP_K];
          memcpy(cands,pre+(size_t)i*TOP_K,(size_t)nc*sizeof(cand_t));
          for(int j=0;j<nc;j++) cands[j].combined=(int64_t)256*cands[j].dot_px+(int64_t)192*cands[j].dot_vg;
          qsort(cands,(size_t)nc,sizeof(cand_t),cmp_combined_d);
          int kv[N_CLASSES]={0}; int k3=nc<3?nc:3;
          for(int j=0;j<k3;j++) kv[train_labels[cands[j].id]]++;
          int pred=0; for(int cl=1;cl<N_CLASSES;cl++) if(kv[cl]>kv[pred]) pred=cl;
          baseline_preds[i]=(uint8_t)pred;
      }
    }
    printf("  Pairwise 4-probe:\n");
    run_strategy(nbr_pair,4,bg,pre,nc_arr,votes2,256,0,192,&t_strat);
    print_pair_deltas(pre,nc_arr,256,0,192,baseline_preds);
    printf("  Combo 8-probe:\n");
    run_strategy(nbr_combo,8,bg,pre,nc_arr,votes2,256,0,192,&t_strat);
    print_pair_deltas(pre,nc_arr,256,0,192,baseline_preds);

    free(votes2);

    printf("\n=== FINAL SUMMARY ===\n");
    printf("  E. Grid best (8-probe original):       %.2f%%\n", 100.0*best_correct/TEST_N);
    printf("  See probe type comparison table above.\n");
    printf("\nTotal runtime: %.2f sec\n", now_sec()-t0);

    free(pre); free(nc_arr);
    return 0;
}
