/*
 * sstt_oracle_v2.c — Multi-Specialist Oracle with Pentary Specialist
 *
 * Replaces oracle.c's secondary (3-channel ternary, same feature family)
 * with a PENTARY pixel specialist — orthogonal to the primary because it
 * operates on a fundamentally different quantization of the same signal.
 *
 * Primary:    Bytepacked cascade  — joint 8-bit encoding, 256 values/block
 * Secondary:  Pentary pixel       — 5-level pixel quant, 125 values/block
 *
 * Why pentary is complementary:
 *   Ternary maps pixel values 85–170 to 0 (the gray zone is silent).
 *   Pentary maps 40–100 → -1,  100–180 → 0,  180–230 → +1.
 *   Faint loop closures (4↔9), thin second curves (3↔8), faint top bars
 *   (7↔1) sit in the 40–100 or 180–230 range.  The primary votes nothing
 *   at those blocks.  The pentary specialist votes specifically there.
 *   Their candidate pools are structurally different, not just encoded differently.
 *
 * Routing (same gate as oracle.c):
 *   k=3 unanimous (3-0) → primary only  (~92% of images, 98.8% accurate)
 *   k=3 split     (2-1) → merge primary + pentary pools, re-rank, k=3
 *
 * Tests:
 *   A. Primary only       (bytepacked cascade baseline)
 *   B. Routed oracle v2   (split → pentary merge)
 *   C. Full ensemble      (always both)
 *
 * Pentary thresholds: 40 / 100 / 180 / 230  (optimal from sstt_pentary.c)
 *
 * Build: make sstt_oracle_v2
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
#define SIG_PAD         256

/* Bytepacked primary */
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

/* Pentary specialist */
#define PENT_BVALS      125     /* 5^3 */
#define PENT_BG         0       /* penc(-2,-2,-2) = all-white block */
#define PENT_PADDED     800

/* Pentary quantization thresholds (optimal from sstt_pentary.c) */
#define PENT_T1  40
#define PENT_T2  100
#define PENT_T3  180
#define PENT_T4  230

#define IG_SCALE        16
#define TOP_K           200
#define TOP_K_MERGE     300

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
static uint8_t *train_labels,  *test_labels;

/* Ternary pixel / gradients (for primary features + dot refinement) */
static int8_t  *tern_train, *tern_test;
static int8_t  *hgrad_train, *hgrad_test;
static int8_t  *vgrad_train, *vgrad_test;

/* Bytepacked primary signatures */
static uint8_t *px_tr, *px_te;
static uint8_t *hg_tr, *hg_te;
static uint8_t *vg_tr, *vg_te;
static uint8_t *ht_tr, *ht_te;
static uint8_t *vt_tr, *vt_te;
static uint8_t *joint_tr, *joint_te;

/* Pentary pixel images */
static int8_t  *pent_train, *pent_test;

/* Pentary pixel block signatures */
static uint8_t *pent_tr, *pent_te;

/* ---- Primary (bytepacked) index ---- */
static uint16_t  p_ig[N_BLOCKS];
static uint8_t   p_nbr[BYTE_VALS][8];
static uint32_t  p_off[N_BLOCKS][BYTE_VALS];
static uint16_t  p_sz [N_BLOCKS][BYTE_VALS];
static uint32_t *p_pool;
static uint8_t   p_bg;

/* ---- Pentary specialist index ---- */
static uint16_t  q_ig[N_BLOCKS];
static uint8_t   q_nbr[PENT_BVALS][12];
static uint8_t   q_nbr_cnt[PENT_BVALS];
static uint32_t  q_off[N_BLOCKS][PENT_BVALS];
static uint16_t  q_sz [N_BLOCKS][PENT_BVALS];
static uint32_t *q_pool;

/* Pre-allocated histogram for select_top_k */
static int   *g_hist     = NULL;
static size_t g_hist_cap = 0;

/* ================================================================
 *  Data loading
 * ================================================================ */
static uint8_t *load_idx(const char *path, uint32_t *cnt,
                          uint32_t *ro, uint32_t *co) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "ERR: %s\n", path); exit(1); }
    uint32_t m, n;
    fread(&m,4,1,f); fread(&n,4,1,f);
    m=__builtin_bswap32(m); n=__builtin_bswap32(n); *cnt=n;
    size_t s=1;
    if ((m&0xFF)>=3) {
        uint32_t r,c; fread(&r,4,1,f); fread(&c,4,1,f);
        r=__builtin_bswap32(r); c=__builtin_bswap32(c);
        if(ro)*ro=r; if(co)*co=c; s=(size_t)r*c;
    } else { if(ro)*ro=0; if(co)*co=0; }
    uint8_t *d=malloc((size_t)n*s);
    fread(d,1,(size_t)n*s,f); fclose(f); return d;
}
static void load_data(void) {
    uint32_t n,r,c; char p[256];
    snprintf(p,sizeof(p),"%strain-images-idx3-ubyte",data_dir); raw_train_img=load_idx(p,&n,&r,&c);
    snprintf(p,sizeof(p),"%strain-labels-idx1-ubyte",data_dir); train_labels =load_idx(p,&n,NULL,NULL);
    snprintf(p,sizeof(p),"%st10k-images-idx3-ubyte", data_dir); raw_test_img =load_idx(p,&n,&r,&c);
    snprintf(p,sizeof(p),"%st10k-labels-idx1-ubyte", data_dir); test_labels  =load_idx(p,&n,NULL,NULL);
}

/* ================================================================
 *  Feature computation
 * ================================================================ */
static inline int8_t clamp_trit(int v) { return v>0?1:v<0?-1:0; }

/* Ternary quantization (AVX2) */
static void quant_tern(const uint8_t *src, int8_t *dst, int n) {
    const __m256i bias=_mm256_set1_epi8((char)0x80);
    const __m256i thi =_mm256_set1_epi8((char)(170^0x80));
    const __m256i tlo =_mm256_set1_epi8((char)(85^0x80));
    const __m256i one =_mm256_set1_epi8(1);
    for(int i=0;i<n;i++){
        const uint8_t*s=src+(size_t)i*PIXELS; int8_t*d=dst+(size_t)i*PADDED; int k;
        for(k=0;k+32<=PIXELS;k+=32){
            __m256i px=_mm256_loadu_si256((const __m256i*)(s+k));
            __m256i sp=_mm256_xor_si256(px,bias);
            _mm256_storeu_si256((__m256i*)(d+k),
                _mm256_sub_epi8(_mm256_and_si256(_mm256_cmpgt_epi8(sp,thi),one),
                                _mm256_and_si256(_mm256_cmpgt_epi8(tlo,sp),one)));
        }
        for(;k<PIXELS;k++) d[k]=s[k]>170?1:s[k]<85?-1:0;
        memset(d+PIXELS,0,PADDED-PIXELS);
    }
}

/* Pentary quantization (scalar — 5 levels) */
static inline int8_t quant_pent_px(uint8_t p) {
    if (p >= PENT_T4) return  2;
    if (p >= PENT_T3) return  1;
    if (p >= PENT_T2) return  0;
    if (p >= PENT_T1) return -1;
    return -2;
}
static void quant_pent_all(const uint8_t *src, int8_t *dst, int n) {
    for(int i=0;i<n;i++){
        const uint8_t*s=src+(size_t)i*PIXELS; int8_t*d=dst+(size_t)i*PADDED;
        for(int k=0;k<PIXELS;k++) d[k]=quant_pent_px(s[k]);
        memset(d+PIXELS,0,PADDED-PIXELS);
    }
}

static void gradients(const int8_t *t,int8_t *h,int8_t *v,int n) {
    for(int i=0;i<n;i++){
        const int8_t*ti=t+(size_t)i*PADDED; int8_t*hi=h+(size_t)i*PADDED; int8_t*vi=v+(size_t)i*PADDED;
        for(int y=0;y<IMG_H;y++){
            for(int x=0;x<IMG_W-1;x++) hi[y*IMG_W+x]=clamp_trit(ti[y*IMG_W+x+1]-ti[y*IMG_W+x]);
            hi[y*IMG_W+IMG_W-1]=0;
        }
        memset(hi+PIXELS,0,PADDED-PIXELS);
        for(int y=0;y<IMG_H-1;y++) for(int x=0;x<IMG_W;x++) vi[y*IMG_W+x]=clamp_trit(ti[(y+1)*IMG_W+x]-ti[y*IMG_W+x]);
        memset(vi+(IMG_H-1)*IMG_W,0,IMG_W); memset(vi+PIXELS,0,PADDED-PIXELS);
    }
}

static inline uint8_t benc(int8_t a,int8_t b,int8_t c){return(uint8_t)((a+1)*9+(b+1)*3+(c+1));}
static void block_sigs_tern(const int8_t*data,uint8_t*sigs,int n){
    for(int i=0;i<n;i++){
        const int8_t*img=data+(size_t)i*PADDED; uint8_t*sig=sigs+(size_t)i*SIG_PAD;
        for(int y=0;y<IMG_H;y++) for(int s=0;s<BLKS_PER_ROW;s++){int b=y*IMG_W+s*3;sig[y*BLKS_PER_ROW+s]=benc(img[b],img[b+1],img[b+2]);}
        memset(sig+N_BLOCKS,0xFF,SIG_PAD-N_BLOCKS);
    }
}

/* Pentary block encoding: (p+2)*25 + (p+2)*5 + (p+2), range [0,124] */
static inline uint8_t penc(int8_t a,int8_t b,int8_t c){return(uint8_t)((a+2)*25+(b+2)*5+(c+2));}
static void block_sigs_pent(const int8_t*data,uint8_t*sigs,int n){
    for(int i=0;i<n;i++){
        const int8_t*img=data+(size_t)i*PADDED; uint8_t*sig=sigs+(size_t)i*SIG_PAD;
        for(int y=0;y<IMG_H;y++) for(int s=0;s<BLKS_PER_ROW;s++){int b=y*IMG_W+s*3;sig[y*BLKS_PER_ROW+s]=penc(img[b],img[b+1],img[b+2]);}
        memset(sig+N_BLOCKS,0xFF,SIG_PAD-N_BLOCKS);
    }
}

static inline uint8_t tenc(uint8_t a,uint8_t b){
    int8_t a0=(a/9)-1,a1=((a/3)%3)-1,a2=(a%3)-1,b0=(b/9)-1,b1=((b/3)%3)-1,b2=(b%3)-1;
    return benc(clamp_trit(b0-a0),clamp_trit(b1-a1),clamp_trit(b2-a2));
}
static void trans(const uint8_t*bs,int str,uint8_t*ht,uint8_t*vt,int n){
    for(int i=0;i<n;i++){
        const uint8_t*s=bs+(size_t)i*str; uint8_t*h=ht+(size_t)i*TRANS_PAD; uint8_t*v=vt+(size_t)i*TRANS_PAD;
        for(int y=0;y<IMG_H;y++) for(int ss=0;ss<H_TRANS_PER_ROW;ss++) h[y*H_TRANS_PER_ROW+ss]=tenc(s[y*BLKS_PER_ROW+ss],s[y*BLKS_PER_ROW+ss+1]);
        memset(h+N_HTRANS,0xFF,TRANS_PAD-N_HTRANS);
        for(int y=0;y<V_TRANS_PER_COL;y++) for(int ss=0;ss<BLKS_PER_ROW;ss++) v[y*BLKS_PER_ROW+ss]=tenc(s[y*BLKS_PER_ROW+ss],s[(y+1)*BLKS_PER_ROW+ss]);
        memset(v+N_VTRANS,0xFF,TRANS_PAD-N_VTRANS);
    }
}
static uint8_t enc_d(uint8_t px,uint8_t hg,uint8_t vg,uint8_t ht,uint8_t vt){
    int ps=((px/9)-1)+(((px/3)%3)-1)+((px%3)-1);
    int hs=((hg/9)-1)+(((hg/3)%3)-1)+((hg%3)-1);
    int vs=((vg/9)-1)+(((vg/3)%3)-1)+((vg%3)-1);
    uint8_t pc=ps<0?0:ps==0?1:ps<3?2:3;
    uint8_t hc=hs<0?0:hs==0?1:hs<3?2:3;
    uint8_t vc=vs<0?0:vs==0?1:vs<3?2:3;
    return pc|(hc<<2)|(vc<<4)|((ht!=BG_TRANS)?1<<6:0)|((vt!=BG_TRANS)?1<<7:0);
}
static void joint_sigs(uint8_t*out,int n,const uint8_t*px,const uint8_t*hg,const uint8_t*vg,const uint8_t*ht,const uint8_t*vt){
    for(int i=0;i<n;i++){
        const uint8_t*pi=px+(size_t)i*SIG_PAD,*hi=hg+(size_t)i*SIG_PAD,*vi=vg+(size_t)i*SIG_PAD;
        const uint8_t*hti=ht+(size_t)i*TRANS_PAD,*vti=vt+(size_t)i*TRANS_PAD; uint8_t*oi=out+(size_t)i*SIG_PAD;
        for(int y=0;y<IMG_H;y++) for(int s=0;s<BLKS_PER_ROW;s++){
            int k=y*BLKS_PER_ROW+s;
            uint8_t htb=s>0?hti[y*H_TRANS_PER_ROW+(s-1)]:BG_TRANS;
            uint8_t vtb=y>0?vti[(y-1)*BLKS_PER_ROW+s]:BG_TRANS;
            oi[k]=enc_d(pi[k],hi[k],vi[k],htb,vtb);
        }
        memset(oi+N_BLOCKS,0xFF,SIG_PAD-N_BLOCKS);
    }
}

/* ================================================================
 *  IG computation (generic: works for any n_vals / bg)
 * ================================================================ */
static void compute_ig(const uint8_t *sigs, int n_vals, uint8_t bg,
                        uint16_t *ig_out) {
    int cc[N_CLASSES]={0};
    for(int i=0;i<TRAIN_N;i++) cc[train_labels[i]]++;
    double hc=0;
    for(int c=0;c<N_CLASSES;c++){double p=(double)cc[c]/TRAIN_N;if(p>0)hc-=p*log2(p);}
    double raw[N_BLOCKS],mx=0;
    for(int k=0;k<N_BLOCKS;k++){
        int *cnt=calloc((size_t)n_vals*N_CLASSES,sizeof(int));
        int *vt =calloc(n_vals,sizeof(int));
        for(int i=0;i<TRAIN_N;i++){
            int v=sigs[(size_t)i*SIG_PAD+k];
            cnt[v*N_CLASSES+train_labels[i]]++; vt[v]++;
        }
        double hcond=0;
        for(int v=0;v<n_vals;v++){
            if(!vt[v]||v==bg) continue;
            double pv=(double)vt[v]/TRAIN_N,hv=0;
            for(int c=0;c<N_CLASSES;c++){double pc=(double)cnt[v*N_CLASSES+c]/vt[v];if(pc>0)hv-=pc*log2(pc);}
            hcond+=pv*hv;
        }
        raw[k]=hc-hcond; if(raw[k]>mx) mx=raw[k];
        free(cnt); free(vt);
    }
    for(int k=0;k<N_BLOCKS;k++){ig_out[k]=mx>0?(uint16_t)(raw[k]/mx*IG_SCALE+0.5):1;if(!ig_out[k])ig_out[k]=1;}
}

/* ================================================================
 *  Primary index (bytepacked Encoding D)
 * ================================================================ */
static void build_primary(void) {
    long vc[BYTE_VALS]={0};
    for(int i=0;i<TRAIN_N;i++){const uint8_t*s=joint_tr+(size_t)i*SIG_PAD;for(int k=0;k<N_BLOCKS;k++)vc[s[k]]++;}
    p_bg=0; long mc=0;
    for(int v=0;v<BYTE_VALS;v++) if(vc[v]>mc){mc=vc[v];p_bg=(uint8_t)v;}

    compute_ig(joint_tr,BYTE_VALS,p_bg,p_ig);

    for(int v=0;v<BYTE_VALS;v++) for(int b=0;b<8;b++) p_nbr[v][b]=(uint8_t)(v^(1<<b));

    memset(p_sz,0,sizeof(p_sz));
    for(int i=0;i<TRAIN_N;i++){const uint8_t*s=joint_tr+(size_t)i*SIG_PAD;for(int k=0;k<N_BLOCKS;k++)if(s[k]!=p_bg)p_sz[k][s[k]]++;}
    uint32_t tot=0;
    for(int k=0;k<N_BLOCKS;k++) for(int v=0;v<BYTE_VALS;v++){p_off[k][v]=tot;tot+=p_sz[k][v];}
    p_pool=malloc((size_t)tot*sizeof(uint32_t));
    uint32_t(*wp)[BYTE_VALS]=malloc((size_t)N_BLOCKS*BYTE_VALS*4);
    memcpy(wp,p_off,(size_t)N_BLOCKS*BYTE_VALS*4);
    for(int i=0;i<TRAIN_N;i++){const uint8_t*s=joint_tr+(size_t)i*SIG_PAD;for(int k=0;k<N_BLOCKS;k++)if(s[k]!=p_bg)p_pool[wp[k][s[k]]++]=(uint32_t)i;}
    free(wp);
    printf("  Primary  (bytepacked): %u entries (%.1f MB)\n",tot,(double)tot*4/1048576);
}

/* ================================================================
 *  Pentary specialist index
 *
 *  Multi-probe neighbors: for each of the 3 trit positions in a
 *  pentary block, any of the 4 other pentary values → up to 12 nbrs.
 * ================================================================ */
static void build_pentary_nbr_table(void) {
    int pvals[5]={-2,-1,0,1,2};
    for(int v=0;v<PENT_BVALS;v++){
        int p0=(v/25)-2, p1=((v/5)%5)-2, p2=(v%5)-2;
        int orig[3]={p0,p1,p2}; int nc=0;
        for(int pos=0;pos<3;pos++){
            for(int alt=0;alt<5;alt++){
                if(pvals[alt]==orig[pos]) continue;
                int m[3]={orig[0],orig[1],orig[2]}; m[pos]=pvals[alt];
                q_nbr[v][nc++]=penc((int8_t)m[0],(int8_t)m[1],(int8_t)m[2]);
            }
        }
        q_nbr_cnt[v]=(uint8_t)nc;
    }
}

static void build_pentary(void) {
    build_pentary_nbr_table();
    compute_ig(pent_tr,PENT_BVALS,PENT_BG,q_ig);

    memset(q_sz,0,sizeof(q_sz));
    for(int i=0;i<TRAIN_N;i++){const uint8_t*s=pent_tr+(size_t)i*SIG_PAD;for(int k=0;k<N_BLOCKS;k++)if(s[k]!=PENT_BG)q_sz[k][s[k]]++;}
    uint32_t tot=0;
    for(int k=0;k<N_BLOCKS;k++) for(int v=0;v<PENT_BVALS;v++){q_off[k][v]=tot;tot+=q_sz[k][v];}
    q_pool=malloc((size_t)tot*sizeof(uint32_t));
    uint32_t wpos[N_BLOCKS][PENT_BVALS];
    memcpy(wpos,q_off,sizeof(wpos));
    for(int i=0;i<TRAIN_N;i++){const uint8_t*s=pent_tr+(size_t)i*SIG_PAD;for(int k=0;k<N_BLOCKS;k++)if(s[k]!=PENT_BG)q_pool[wpos[k][s[k]]++]=(uint32_t)i;}
    printf("  Specialist (pentary) : %u entries (%.1f MB)\n",tot,(double)tot*4/1048576);
}

/* ================================================================
 *  AVX2 ternary dot
 * ================================================================ */
static inline int32_t tdot(const int8_t*a,const int8_t*b){
    __m256i acc=_mm256_setzero_si256();
    for(int i=0;i<PADDED;i+=32) acc=_mm256_add_epi8(acc,_mm256_sign_epi8(_mm256_load_si256((const __m256i*)(a+i)),_mm256_load_si256((const __m256i*)(b+i))));
    __m256i lo=_mm256_cvtepi8_epi16(_mm256_castsi256_si128(acc));
    __m256i hi=_mm256_cvtepi8_epi16(_mm256_extracti128_si256(acc,1));
    __m256i s32=_mm256_madd_epi16(_mm256_add_epi16(lo,hi),_mm256_set1_epi16(1));
    __m128i s=_mm_add_epi32(_mm256_castsi256_si128(s32),_mm256_extracti128_si256(s32,1));
    s=_mm_hadd_epi32(s,s); s=_mm_hadd_epi32(s,s); return _mm_cvtsi128_si32(s);
}

/* ================================================================
 *  Candidate machinery
 * ================================================================ */
typedef struct { uint32_t id,votes; int32_t dot_px,dot_hg,dot_vg; int64_t combined; } cand_t;
static int cmp_votes_d(const void*a,const void*b){return(int)((const cand_t*)b)->votes-(int)((const cand_t*)a)->votes;}
static int cmp_comb_d (const void*a,const void*b){int64_t da=((const cand_t*)a)->combined,db=((const cand_t*)b)->combined;return(db>da)-(db<da);}

static int select_top_k(const uint32_t*votes,int n,cand_t*out,int k){
    uint32_t mx=0; for(int i=0;i<n;i++) if(votes[i]>mx) mx=votes[i];
    if(!mx) return 0;
    if((size_t)(mx+1)>g_hist_cap){g_hist_cap=(size_t)(mx+1)+4096;free(g_hist);g_hist=malloc(g_hist_cap*sizeof(int));}
    memset(g_hist,0,(mx+1)*sizeof(int));
    for(int i=0;i<n;i++) if(votes[i]) g_hist[votes[i]]++;
    int cum=0,thr; for(thr=(int)mx;thr>=1;thr--){cum+=g_hist[thr];if(cum>=k)break;}
    if(thr<1)thr=1;
    int nc=0;
    for(int i=0;i<n&&nc<k;i++) if(votes[i]>=(uint32_t)thr) out[nc++]=(cand_t){(uint32_t)i,votes[i],0,0,0,0};
    qsort(out,(size_t)nc,sizeof(cand_t),cmp_votes_d); return nc;
}

static int knn_vote(const cand_t*c,int nc,int k){
    int v[N_CLASSES]={0}; if(k>nc)k=nc;
    for(int i=0;i<k;i++) v[train_labels[c[i].id]]++;
    int best=0; for(int c2=1;c2<N_CLASSES;c2++) if(v[c2]>v[best]) best=c2; return best;
}

static void compute_dots(cand_t*cands,int nc,int ti){
    const int8_t*qp=tern_train+(size_t)ti*PADDED; /* NOTE: ti = training image id */
    (void)qp; /* computed per-candidate below */
    const int8_t*tp=tern_test +(size_t)ti*PADDED;
    const int8_t*th=hgrad_test+(size_t)ti*PADDED;
    const int8_t*tv=vgrad_test+(size_t)ti*PADDED;
    for(int j=0;j<nc;j++){
        uint32_t id=cands[j].id;
        cands[j].dot_px=tdot(tp,tern_train  +(size_t)id*PADDED);
        cands[j].dot_hg=tdot(th,hgrad_train +(size_t)id*PADDED);
        cands[j].dot_vg=tdot(tv,vgrad_train +(size_t)id*PADDED);
        cands[j].combined=(int64_t)256*cands[j].dot_px+(int64_t)192*cands[j].dot_vg;
    }
}

/* Merge two candidate pools: union by id, accumulate votes */
static int merge_pools(const cand_t*a,int na,const cand_t*b,int nb,cand_t*out,int max){
    int n=na<max?na:max;
    memcpy(out,a,(size_t)n*sizeof(cand_t));
    for(int j=0;j<nb&&n<max;j++){
        int found=0;
        for(int i=0;i<na;i++) if(out[i].id==b[j].id){out[i].votes+=b[j].votes;found=1;break;}
        if(!found) out[n++]=b[j];
    }
    return n;
}

/* ================================================================
 *  Vote accumulation
 * ================================================================ */
static void vote_primary(uint32_t*votes,int img){
    memset(votes,0,TRAIN_N*sizeof(uint32_t));
    const uint8_t*sig=joint_te+(size_t)img*SIG_PAD;
    for(int k=0;k<N_BLOCKS;k++){
        uint8_t bv=sig[k]; if(bv==p_bg) continue;
        uint16_t w=p_ig[k],wh=w>1?w/2:1;
        {uint32_t off=p_off[k][bv];uint16_t sz=p_sz[k][bv];const uint32_t*ids=p_pool+off;for(uint16_t j=0;j<sz;j++)votes[ids[j]]+=w;}
        for(int nb=0;nb<8;nb++){uint8_t nv=p_nbr[bv][nb];if(nv==p_bg)continue;uint32_t noff=p_off[k][nv];uint16_t nsz=p_sz[k][nv];const uint32_t*nids=p_pool+noff;for(uint16_t j=0;j<nsz;j++)votes[nids[j]]+=wh;}
    }
}

static void vote_pentary(uint32_t*votes,int img){
    memset(votes,0,TRAIN_N*sizeof(uint32_t));
    const uint8_t*sig=pent_te+(size_t)img*SIG_PAD;
    for(int k=0;k<N_BLOCKS;k++){
        uint8_t bv=sig[k]; if(bv==PENT_BG) continue;
        uint16_t w=q_ig[k],wh=w>1?w/2:1;
        {uint32_t off=q_off[k][bv];uint16_t sz=q_sz[k][bv];const uint32_t*ids=q_pool+off;for(uint16_t j=0;j<sz;j++)votes[ids[j]]+=w;}
        for(int nb=0;nb<q_nbr_cnt[bv];nb++){uint8_t nv=q_nbr[bv][nb];if(nv==PENT_BG)continue;uint32_t noff=q_off[k][nv];uint16_t nsz=q_sz[k][nv];const uint32_t*nids=q_pool+noff;for(uint16_t j=0;j<nsz;j++)votes[nids[j]]+=wh;}
    }
}

/* ================================================================
 *  Reporting
 * ================================================================ */
static void report(const uint8_t *preds, const char *label, int correct, double t) {
    printf("--- %s ---\n", label);
    printf("  Accuracy: %.2f%%  (%d errors)  %.2f sec\n\n",
           100.0*correct/TEST_N, TEST_N-correct, t);
    int conf[N_CLASSES][N_CLASSES]; memset(conf,0,sizeof(conf));
    for(int i=0;i<TEST_N;i++) conf[test_labels[i]][preds[i]]++;
    printf("  Confusion (rows=actual, cols=predicted):\n       ");
    for(int c=0;c<N_CLASSES;c++) printf(" %4d",c);
    printf("   | Recall\n  -----");
    for(int c=0;c<N_CLASSES;c++) (void)c,printf("-----");
    printf("---+-------\n");
    for(int r=0;r<N_CLASSES;r++){
        printf("    %d: ",r); int rt=0;
        for(int c=0;c<N_CLASSES;c++){printf(" %4d",conf[r][c]);rt+=conf[r][c];}
        printf("   | %5.1f%%\n",rt>0?100.0*conf[r][r]/rt:0.0);
    }
    typedef struct{int a,b,count;}pair_t;
    pair_t pairs[45]; int np=0;
    for(int a=0;a<N_CLASSES;a++) for(int b=a+1;b<N_CLASSES;b++) pairs[np++]=(pair_t){a,b,conf[a][b]+conf[b][a]};
    for(int i=0;i<np-1;i++) for(int j=i+1;j<np;j++) if(pairs[j].count>pairs[i].count){pair_t t=pairs[i];pairs[i]=pairs[j];pairs[j]=t;}
    printf("  Top-5 confused:");
    for(int i=0;i<5&&i<np;i++) printf("  %d\xe2\x86\x94%d:%d",pairs[i].a,pairs[i].b,pairs[i].count);
    printf("\n\n");
}

/* ================================================================
 *  Main
 * ================================================================ */
int main(int argc, char **argv) {
    double t0=now_sec();
    if(argc>1){
        data_dir=argv[1]; size_t l=strlen(data_dir);
        if(l&&data_dir[l-1]!='/'){char*buf=malloc(l+2);memcpy(buf,data_dir,l);buf[l]='/';buf[l+1]='\0';data_dir=buf;}
    }
    const char*ds=strstr(data_dir,"fashion")?"Fashion-MNIST":"MNIST";
    printf("=== SSTT Oracle v2 — Pentary Specialist (%s) ===\n\n",ds);

    load_data();

    /* Ternary features */
    tern_train =aligned_alloc(32,(size_t)TRAIN_N*PADDED); tern_test  =aligned_alloc(32,(size_t)TEST_N *PADDED);
    hgrad_train=aligned_alloc(32,(size_t)TRAIN_N*PADDED); hgrad_test =aligned_alloc(32,(size_t)TEST_N *PADDED);
    vgrad_train=aligned_alloc(32,(size_t)TRAIN_N*PADDED); vgrad_test =aligned_alloc(32,(size_t)TEST_N *PADDED);
    quant_tern(raw_train_img,tern_train,TRAIN_N); quant_tern(raw_test_img,tern_test,TEST_N);
    gradients(tern_train,hgrad_train,vgrad_train,TRAIN_N);
    gradients(tern_test, hgrad_test, vgrad_test, TEST_N);

    /* Bytepacked primary signatures */
    px_tr=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD); px_te=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    hg_tr=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD); hg_te=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    vg_tr=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD); vg_te=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    block_sigs_tern(tern_train, px_tr,TRAIN_N); block_sigs_tern(tern_test, px_te,TEST_N);
    block_sigs_tern(hgrad_train,hg_tr,TRAIN_N); block_sigs_tern(hgrad_test,hg_te,TEST_N);
    block_sigs_tern(vgrad_train,vg_tr,TRAIN_N); block_sigs_tern(vgrad_test,vg_te,TEST_N);
    ht_tr=aligned_alloc(32,(size_t)TRAIN_N*TRANS_PAD); ht_te=aligned_alloc(32,(size_t)TEST_N*TRANS_PAD);
    vt_tr=aligned_alloc(32,(size_t)TRAIN_N*TRANS_PAD); vt_te=aligned_alloc(32,(size_t)TEST_N*TRANS_PAD);
    trans(px_tr,SIG_PAD,ht_tr,vt_tr,TRAIN_N); trans(px_te,SIG_PAD,ht_te,vt_te,TEST_N);
    joint_tr=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD); joint_te=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    joint_sigs(joint_tr,TRAIN_N,px_tr,hg_tr,vg_tr,ht_tr,vt_tr);
    joint_sigs(joint_te,TEST_N, px_te,hg_te,vg_te,ht_te,vt_te);

    /* Pentary specialist signatures */
    pent_train=aligned_alloc(32,(size_t)TRAIN_N*PADDED); pent_test=aligned_alloc(32,(size_t)TEST_N*PADDED);
    quant_pent_all(raw_train_img,pent_train,TRAIN_N);
    quant_pent_all(raw_test_img, pent_test, TEST_N);
    pent_tr=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD); pent_te=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    block_sigs_pent(pent_train,pent_tr,TRAIN_N); block_sigs_pent(pent_test,pent_te,TEST_N);

    printf("Building indices...\n");
    build_primary();
    build_pentary();
    printf("  Setup: %.2f sec\n\n",now_sec()-t0);

    /* Verify pentary background */
    long vc[PENT_BVALS]={0};
    for(int i=0;i<TRAIN_N;i++){const uint8_t*s=pent_tr+(size_t)i*SIG_PAD;for(int k=0;k<N_BLOCKS;k++)vc[s[k]]++;}
    int pbg=0; long pmc=0; for(int v=0;v<PENT_BVALS;v++) if(vc[v]>pmc){pmc=vc[v];pbg=v;}
    printf("  Pentary bg detected: %d (%.1f%%)  [expected %d]\n\n",
           pbg,100.0*pmc/((long)TRAIN_N*N_BLOCKS),PENT_BG);

    uint32_t *vp =calloc(TRAIN_N,sizeof(uint32_t));
    uint32_t *vq =calloc(TRAIN_N,sizeof(uint32_t));
    cand_t   *cp =malloc(TOP_K      *sizeof(cand_t));
    cand_t   *cq =malloc(TOP_K      *sizeof(cand_t));
    cand_t   *cm =malloc(TOP_K_MERGE*sizeof(cand_t));
    uint8_t  *pA =calloc(TEST_N,1);
    uint8_t  *pB =calloc(TEST_N,1);
    uint8_t  *pC =calloc(TEST_N,1);

    int cA=0,cB=0,cC=0;
    int n_unani=0,n_escal=0,cB_u=0,cB_e=0;
    double tA=0,tB=0,tC=0,ts;

    printf("Running oracle...\n");
    for(int i=0;i<TEST_N;i++){

        /* ---- A: primary only ---- */
        ts=now_sec();
        vote_primary(vp,i);
        int np=select_top_k(vp,TRAIN_N,cp,TOP_K);
        compute_dots(cp,np,i);
        qsort(cp,(size_t)np,sizeof(cand_t),cmp_comb_d);
        int pa=knn_vote(cp,np,3);
        pA[i]=(uint8_t)pa; if(pa==test_labels[i]) cA++;
        tA+=now_sec()-ts;

        /* Check unanimity */
        int ballot[N_CLASSES]={0},k3=np<3?np:3;
        for(int j=0;j<k3;j++) ballot[train_labels[cp[j].id]]++;
        int top=0; for(int c=0;c<N_CLASSES;c++) if(ballot[c]>top) top=ballot[c];
        int unanimous=(top==3);

        /* ---- B: routed — unanimous→A, split→pentary merge ---- */
        ts=now_sec();
        if(unanimous){
            pB[i]=(uint8_t)pa;
            if(pa==test_labels[i]){cB++;cB_u++;} n_unani++;
        } else {
            vote_pentary(vq,i);
            int nq=select_top_k(vq,TRAIN_N,cq,TOP_K);
            compute_dots(cq,nq,i);
            int nm=merge_pools(cp,np,cq,nq,cm,TOP_K_MERGE);
            for(int j=0;j<nm;j++) cm[j].combined=(int64_t)256*cm[j].dot_px+(int64_t)192*cm[j].dot_vg;
            qsort(cm,(size_t)nm,sizeof(cand_t),cmp_comb_d);
            int pb=knn_vote(cm,nm,3);
            pB[i]=(uint8_t)pb;
            if(pb==test_labels[i]){cB++;cB_e++;} n_escal++;
        }
        tB+=now_sec()-ts;

        /* ---- C: full ensemble — always merge ---- */
        ts=now_sec();
        if(unanimous){
            /* Need pentary for this image too */
            vote_pentary(vq,i);
            int nq=select_top_k(vq,TRAIN_N,cq,TOP_K);
            compute_dots(cq,nq,i);
            int nm=merge_pools(cp,np,cq,nq,cm,TOP_K_MERGE);
            for(int j=0;j<nm;j++) cm[j].combined=(int64_t)256*cm[j].dot_px+(int64_t)192*cm[j].dot_vg;
            qsort(cm,(size_t)nm,sizeof(cand_t),cmp_comb_d);
            int pc=knn_vote(cm,nm,3); pC[i]=(uint8_t)pc;
            if(pc==test_labels[i]) cC++;
        } else {
            /* Reuse cq and merged pool from B */
            int nq=select_top_k(vq,TRAIN_N,cq,TOP_K);
            compute_dots(cq,nq,i);
            int nm=merge_pools(cp,np,cq,nq,cm,TOP_K_MERGE);
            for(int j=0;j<nm;j++) cm[j].combined=(int64_t)256*cm[j].dot_px+(int64_t)192*cm[j].dot_vg;
            qsort(cm,(size_t)nm,sizeof(cand_t),cmp_comb_d);
            int pc=knn_vote(cm,nm,3); pC[i]=(uint8_t)pc;
            if(pc==test_labels[i]) cC++;
        }
        tC+=now_sec()-ts;

        if((i+1)%2000==0)
            fprintf(stderr,"  %d/%d  A:%.2f%%  B:%.2f%%  C:%.2f%%\r",
                    i+1,TEST_N,100.0*cA/(i+1),100.0*cB/(i+1),100.0*cC/(i+1));
    }
    fprintf(stderr,"\n\n");

    report(pA,"A — Primary only (bytepacked, baseline)",cA,tA);
    printf("--- B — Routed Oracle v2 (unanimous→primary, split→pentary merge) ---\n");
    printf("  Accuracy:   %.2f%%  (%d errors)  %.2f sec\n",100.0*cB/TEST_N,TEST_N-cB,tB);
    printf("  Unanimous:  %d (%.1f%%) → primary path:   %.2f%%\n",n_unani,100.0*n_unani/TEST_N,n_unani>0?100.0*cB_u/n_unani:0.0);
    printf("  Escalated:  %d (%.1f%%) → pentary merge:  %.2f%%\n\n",n_escal,100.0*n_escal/TEST_N,n_escal>0?100.0*cB_e/n_escal:0.0);
    report(pB,"B — Routed oracle v2",cB,tB);
    report(pC,"C — Full ensemble (always both)",cC,tC);

    printf("=== ORACLE v2 SUMMARY ===\n");
    printf("  A. Primary only:         %.2f%%  (%.2f sec)\n",100.0*cA/TEST_N,tA);
    printf("  B. Routed pentary:       %.2f%%  (%.2f sec, %d escalated = %.1f%%)\n",100.0*cB/TEST_N,tB,n_escal,100.0*n_escal/TEST_N);
    printf("  C. Full ensemble:        %.2f%%  (%.2f sec)\n",100.0*cC/TEST_N,tC);
    printf("\n  vs oracle v1 (ternary secondary):  B=96.44%%  C=96.42%%\n");
    printf("  Delta v2 - v1:  B=%+.2f pp  C=%+.2f pp\n",
           100.0*cB/TEST_N-96.44, 100.0*cC/TEST_N-96.42);
    printf("\nTotal: %.2f sec\n",now_sec()-t0);

    free(vp);free(vq);free(cp);free(cq);free(cm);
    free(pA);free(pB);free(pC);
    free(p_pool);free(q_pool);
    free(joint_tr);free(joint_te);free(pent_tr);free(pent_te);
    free(px_tr);free(px_te);free(hg_tr);free(hg_te);free(vg_tr);free(vg_te);
    free(ht_tr);free(ht_te);free(vt_tr);free(vt_te);
    free(tern_train);free(tern_test);free(hgrad_train);free(hgrad_test);
    free(vgrad_train);free(vgrad_test);free(pent_train);free(pent_test);
    free(raw_train_img);free(raw_test_img);free(train_labels);free(test_labels);
    return 0;
}
