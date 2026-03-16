/*
 * sstt_oracle_v3.c — Pair-Targeted Re-Vote Oracle
 *
 * The key insight the previous oracles missed:
 *   Global IG weights maximise average information across ALL 10 classes.
 *   But when the oracle identifies a split vote between classes A and B,
 *   we know EXACTLY which discrimination challenge we face.
 *   The globally-optimal weights are NOT the pair-optimal weights.
 *
 * Architecture:
 *   First pass:  bytepacked cascade with global IG
 *                → unanimous 3-0: done (98.82% accuracy, ~92% of images)
 *                → split 2-1:    identifies pair (class_a, class_b)
 *
 *   Second pass: re-vote the SAME bytepacked index with IG weights
 *                computed specifically for discriminating class_a vs class_b
 *                → multi-channel dot → k=3 → prediction
 *
 * Pair-specific IG (precomputed at training time, 45 pairs × 252 blocks):
 *   For pair (a,b): compute IG from only the ~12K training images
 *   with labels a or b. The weights reflect which block positions
 *   best separate these two specific classes — not the global average.
 *
 *   Hypothesis: for 4↔9, the blocks at rows 0–5 (top of digit,
 *   where the loop-closure distinction lives) have LOW global IG
 *   but HIGH pair-specific IG. Re-voting with those weights
 *   upweighted will pull different training images to the top-K.
 *
 * Tests:
 *   A. Primary only (bytepacked, global IG)        ← baseline
 *   B. Oracle v3: split → pair-specific IG re-vote
 *   C. Oracle v2 (split → pentary merge)           ← previous best
 *      Runs B and C for direct comparison
 *
 * Also outputs: top-10 most discriminative blocks per confusion pair,
 *   comparing global IG vs pair IG to verify the hypothesis.
 *
 * Build: make sstt_oracle_v3
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
#define BYTE_VALS       256
#define BG_PIXEL        0
#define BG_GRAD         13
#define BG_TRANS        13
#define H_TRANS_PER_ROW 8
#define N_HTRANS        (H_TRANS_PER_ROW * IMG_H)
#define V_TRANS_PER_COL 27
#define N_VTRANS        (BLKS_PER_ROW * V_TRANS_PER_COL)
#define TRANS_PAD       256
#define PENT_BVALS      125
#define PENT_BG         0
#define PENT_T1         40
#define PENT_T2         100
#define PENT_T3         180
#define PENT_T4         230
#define IG_SCALE        16
#define TOP_K           200

static const char *data_dir = "data/";
static double now_sec(void){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return ts.tv_sec+ts.tv_nsec*1e-9;}

/* ================================================================
 *  Data
 * ================================================================ */
static uint8_t *raw_train_img,*raw_test_img,*train_labels,*test_labels;
static int8_t  *tern_train,*tern_test,*hgrad_train,*hgrad_test,*vgrad_train,*vgrad_test;
static int8_t  *pent_train,*pent_test;
static uint8_t *px_tr,*px_te,*hg_tr,*hg_te,*vg_tr,*vg_te;
static uint8_t *ht_tr,*ht_te,*vt_tr,*vt_te,*joint_tr,*joint_te;
static uint8_t *pent_tr,*pent_te;

/* ---- Bytepacked primary index ---- */
static uint16_t p_ig[N_BLOCKS];
static uint8_t  p_nbr[BYTE_VALS][8];
static uint32_t p_off[N_BLOCKS][BYTE_VALS];
static uint16_t p_sz [N_BLOCKS][BYTE_VALS];
static uint32_t *p_pool;
static uint8_t  p_bg;

/* ---- Pentary index (for oracle v2 comparison) ---- */
static uint16_t q_ig[N_BLOCKS];
static uint8_t  q_nbr[PENT_BVALS][12],q_nbr_cnt[PENT_BVALS];
static uint32_t q_off[N_BLOCKS][PENT_BVALS];
static uint16_t q_sz [N_BLOCKS][PENT_BVALS];
static uint32_t *q_pool;

/* ---- Pair-specific IG (the key new structure) ----
 * pair_ig[a][b][k] = IG weight for block k when discriminating class a vs b
 * Symmetric: pair_ig[a][b] == pair_ig[b][a]
 * 10 × 10 × 252 × 2 bytes = ~49 KB                                    */
static uint16_t pair_ig[N_CLASSES][N_CLASSES][N_BLOCKS];

/* ---- Per-class block value counts (for efficient pair IG computation)
 * class_block_counts[c][k][v] = #training images with label c,
 *   block position k, bytepacked value v
 * 10 × 252 × 256 × 2 bytes = ~1.2 MB                                  */
static uint16_t class_block_counts[N_CLASSES][N_BLOCKS][BYTE_VALS];

static int *g_hist=NULL; static size_t g_hist_cap=0;

/* ================================================================
 *  Feature computation (compact, reused across oracle files)
 * ================================================================ */
static uint8_t *load_idx(const char *p,uint32_t *cnt,uint32_t *ro,uint32_t *co){
    FILE*f=fopen(p,"rb");if(!f){fprintf(stderr,"ERR:%s\n",p);exit(1);}
    uint32_t m,n;fread(&m,4,1,f);fread(&n,4,1,f);m=__builtin_bswap32(m);n=__builtin_bswap32(n);*cnt=n;
    size_t s=1;if((m&0xFF)>=3){uint32_t r,c;fread(&r,4,1,f);fread(&c,4,1,f);r=__builtin_bswap32(r);c=__builtin_bswap32(c);if(ro)*ro=r;if(co)*co=c;s=(size_t)r*c;}else{if(ro)*ro=0;if(co)*co=0;}
    uint8_t*d=malloc((size_t)n*s);fread(d,1,(size_t)n*s,f);fclose(f);return d;
}
static void load_data(void){
    uint32_t n,r,c;char p[256];
    snprintf(p,sizeof(p),"%strain-images-idx3-ubyte",data_dir);raw_train_img=load_idx(p,&n,&r,&c);
    snprintf(p,sizeof(p),"%strain-labels-idx1-ubyte",data_dir);train_labels =load_idx(p,&n,NULL,NULL);
    snprintf(p,sizeof(p),"%st10k-images-idx3-ubyte", data_dir);raw_test_img =load_idx(p,&n,&r,&c);
    snprintf(p,sizeof(p),"%st10k-labels-idx1-ubyte", data_dir);test_labels  =load_idx(p,&n,NULL,NULL);
}
static inline int8_t clamp_trit(int v){return v>0?1:v<0?-1:0;}
static void quant_tern(const uint8_t*src,int8_t*dst,int n){
    const __m256i bias=_mm256_set1_epi8((char)0x80),thi=_mm256_set1_epi8((char)(170^0x80)),tlo=_mm256_set1_epi8((char)(85^0x80)),one=_mm256_set1_epi8(1);
    for(int i=0;i<n;i++){const uint8_t*s=src+(size_t)i*PIXELS;int8_t*d=dst+(size_t)i*PADDED;int k;
    for(k=0;k+32<=PIXELS;k+=32){__m256i px=_mm256_loadu_si256((const __m256i*)(s+k));__m256i sp=_mm256_xor_si256(px,bias);_mm256_storeu_si256((__m256i*)(d+k),_mm256_sub_epi8(_mm256_and_si256(_mm256_cmpgt_epi8(sp,thi),one),_mm256_and_si256(_mm256_cmpgt_epi8(tlo,sp),one)));}
    for(;k<PIXELS;k++)d[k]=s[k]>170?1:s[k]<85?-1:0;memset(d+PIXELS,0,PADDED-PIXELS);}
}
static void quant_pent(const uint8_t*src,int8_t*dst,int n){
    for(int i=0;i<n;i++){const uint8_t*s=src+(size_t)i*PIXELS;int8_t*d=dst+(size_t)i*PADDED;
    for(int k=0;k<PIXELS;k++){uint8_t p=s[k];d[k]=p>=PENT_T4?2:p>=PENT_T3?1:p>=PENT_T2?0:p>=PENT_T1?-1:-2;}
    memset(d+PIXELS,0,PADDED-PIXELS);}
}
static void gradients(const int8_t*t,int8_t*h,int8_t*v,int n){
    for(int i=0;i<n;i++){const int8_t*ti=t+(size_t)i*PADDED;int8_t*hi=h+(size_t)i*PADDED;int8_t*vi=v+(size_t)i*PADDED;
    for(int y=0;y<IMG_H;y++){for(int x=0;x<IMG_W-1;x++)hi[y*IMG_W+x]=clamp_trit(ti[y*IMG_W+x+1]-ti[y*IMG_W+x]);hi[y*IMG_W+IMG_W-1]=0;}
    memset(hi+PIXELS,0,PADDED-PIXELS);
    for(int y=0;y<IMG_H-1;y++)for(int x=0;x<IMG_W;x++)vi[y*IMG_W+x]=clamp_trit(ti[(y+1)*IMG_W+x]-ti[y*IMG_W+x]);
    memset(vi+(IMG_H-1)*IMG_W,0,IMG_W);memset(vi+PIXELS,0,PADDED-PIXELS);}
}
static inline uint8_t benc(int8_t a,int8_t b,int8_t c){return(uint8_t)((a+1)*9+(b+1)*3+(c+1));}
static void bsigs(const int8_t*data,uint8_t*sigs,int n){
    for(int i=0;i<n;i++){const int8_t*img=data+(size_t)i*PADDED;uint8_t*sig=sigs+(size_t)i*SIG_PAD;
    for(int y=0;y<IMG_H;y++)for(int s=0;s<BLKS_PER_ROW;s++){int b=y*IMG_W+s*3;sig[y*BLKS_PER_ROW+s]=benc(img[b],img[b+1],img[b+2]);}
    memset(sig+N_BLOCKS,0xFF,SIG_PAD-N_BLOCKS);}
}
static inline uint8_t penc(int8_t a,int8_t b,int8_t c){return(uint8_t)((a+2)*25+(b+2)*5+(c+2));}
static void psigs(const int8_t*data,uint8_t*sigs,int n){
    for(int i=0;i<n;i++){const int8_t*img=data+(size_t)i*PADDED;uint8_t*sig=sigs+(size_t)i*SIG_PAD;
    for(int y=0;y<IMG_H;y++)for(int s=0;s<BLKS_PER_ROW;s++){int b=y*IMG_W+s*3;sig[y*BLKS_PER_ROW+s]=penc(img[b],img[b+1],img[b+2]);}
    memset(sig+N_BLOCKS,0xFF,SIG_PAD-N_BLOCKS);}
}
static inline uint8_t tenc(uint8_t a,uint8_t b){
    int8_t a0=(a/9)-1,a1=((a/3)%3)-1,a2=(a%3)-1,b0=(b/9)-1,b1=((b/3)%3)-1,b2=(b%3)-1;
    return benc(clamp_trit(b0-a0),clamp_trit(b1-a1),clamp_trit(b2-a2));
}
static void trans(const uint8_t*bs,int str,uint8_t*ht,uint8_t*vt,int n){
    for(int i=0;i<n;i++){const uint8_t*s=bs+(size_t)i*str;uint8_t*h=ht+(size_t)i*TRANS_PAD;uint8_t*v=vt+(size_t)i*TRANS_PAD;
    for(int y=0;y<IMG_H;y++)for(int ss=0;ss<H_TRANS_PER_ROW;ss++)h[y*H_TRANS_PER_ROW+ss]=tenc(s[y*BLKS_PER_ROW+ss],s[y*BLKS_PER_ROW+ss+1]);
    memset(h+N_HTRANS,0xFF,TRANS_PAD-N_HTRANS);
    for(int y=0;y<V_TRANS_PER_COL;y++)for(int ss=0;ss<BLKS_PER_ROW;ss++)v[y*BLKS_PER_ROW+ss]=tenc(s[y*BLKS_PER_ROW+ss],s[(y+1)*BLKS_PER_ROW+ss]);
    memset(v+N_VTRANS,0xFF,TRANS_PAD-N_VTRANS);}
}
static uint8_t enc_d(uint8_t px,uint8_t hg,uint8_t vg,uint8_t ht,uint8_t vt){
    int ps=((px/9)-1)+(((px/3)%3)-1)+((px%3)-1),hs=((hg/9)-1)+(((hg/3)%3)-1)+((hg%3)-1),vs=((vg/9)-1)+(((vg/3)%3)-1)+((vg%3)-1);
    uint8_t pc=ps<0?0:ps==0?1:ps<3?2:3,hc=hs<0?0:hs==0?1:hs<3?2:3,vc=vs<0?0:vs==0?1:vs<3?2:3;
    return pc|(hc<<2)|(vc<<4)|((ht!=BG_TRANS)?1<<6:0)|((vt!=BG_TRANS)?1<<7:0);
}
static void jsigs(uint8_t*out,int n,const uint8_t*px,const uint8_t*hg,const uint8_t*vg,const uint8_t*ht,const uint8_t*vt){
    for(int i=0;i<n;i++){const uint8_t*pi=px+(size_t)i*SIG_PAD,*hi=hg+(size_t)i*SIG_PAD,*vi=vg+(size_t)i*SIG_PAD;
    const uint8_t*hti=ht+(size_t)i*TRANS_PAD,*vti=vt+(size_t)i*TRANS_PAD;uint8_t*oi=out+(size_t)i*SIG_PAD;
    for(int y=0;y<IMG_H;y++)for(int s=0;s<BLKS_PER_ROW;s++){int k=y*BLKS_PER_ROW+s;
    uint8_t htb=s>0?hti[y*H_TRANS_PER_ROW+(s-1)]:BG_TRANS,vtb=y>0?vti[(y-1)*BLKS_PER_ROW+s]:BG_TRANS;
    oi[k]=enc_d(pi[k],hi[k],vi[k],htb,vtb);}memset(oi+N_BLOCKS,0xFF,SIG_PAD-N_BLOCKS);}
}

/* ================================================================
 *  Global IG computation
 * ================================================================ */
static void compute_global_ig(void){
    int cc[N_CLASSES]={0};for(int i=0;i<TRAIN_N;i++)cc[train_labels[i]]++;
    double hc=0;for(int c=0;c<N_CLASSES;c++){double p=(double)cc[c]/TRAIN_N;if(p>0)hc-=p*log2(p);}
    double raw[N_BLOCKS],mx=0;
    for(int k=0;k<N_BLOCKS;k++){
        double hcond=0;
        for(int v=0;v<BYTE_VALS;v++){if((uint8_t)v==p_bg)continue;
        int vt=0;for(int c=0;c<N_CLASSES;c++)vt+=class_block_counts[c][k][v];if(!vt)continue;
        double pv=(double)vt/TRAIN_N,hv=0;
        for(int c=0;c<N_CLASSES;c++){double pc=(double)class_block_counts[c][k][v]/vt;if(pc>0)hv-=pc*log2(pc);}
        hcond+=pv*hv;}
        raw[k]=hc-hcond;if(raw[k]>mx)mx=raw[k];
    }
    for(int k=0;k<N_BLOCKS;k++){p_ig[k]=mx>0?(uint16_t)(raw[k]/mx*IG_SCALE+0.5):1;if(!p_ig[k])p_ig[k]=1;}
}

/* ================================================================
 *  Pair-specific IG computation
 *
 *  For each pair (a,b), binary IG using only training images with
 *  labels a or b. The block positions that best separate a from b
 *  get high weights — regardless of their global discriminability.
 * ================================================================ */
static void compute_pair_ig(void){
    /* Count training images per class */
    int class_n[N_CLASSES]={0};
    for(int i=0;i<TRAIN_N;i++) class_n[train_labels[i]]++;

    for(int a=0;a<N_CLASSES;a++){
        for(int b=a+1;b<N_CLASSES;b++){
            int na=class_n[a], nb=class_n[b];
            int ntotal=na+nb;
            double pa=(double)na/ntotal, pb=(double)nb/ntotal;
            double h_prior=(pa>0?-pa*log2(pa):0)+(pb>0?-pb*log2(pb):0);

            double raw[N_BLOCKS],mx=0;
            for(int k=0;k<N_BLOCKS;k++){
                double hcond=0;
                for(int v=0;v<BYTE_VALS;v++){
                    if((uint8_t)v==p_bg) continue;
                    int nav=class_block_counts[a][k][v];
                    int nbv=class_block_counts[b][k][v];
                    int tot=nav+nbv;
                    if(!tot) continue;
                    double pv=(double)tot/ntotal;
                    double pa_v=(double)nav/tot,pb_v=(double)nbv/tot;
                    double hv=(pa_v>0?-pa_v*log2(pa_v):0)+(pb_v>0?-pb_v*log2(pb_v):0);
                    hcond+=pv*hv;
                }
                raw[k]=h_prior-hcond;
                if(raw[k]>mx)mx=raw[k];
            }
            for(int k=0;k<N_BLOCKS;k++){
                uint16_t w=mx>0?(uint16_t)(raw[k]/mx*IG_SCALE+0.5):1;
                if(!w)w=1;
                pair_ig[a][b][k]=w;
                pair_ig[b][a][k]=w;  /* symmetric */
            }
        }
    }
}

/* ================================================================
 *  Index builders
 * ================================================================ */
static void build_bytepacked(void){
    /* Detect background */
    long vc[BYTE_VALS]={0};
    for(int i=0;i<TRAIN_N;i++){const uint8_t*s=joint_tr+(size_t)i*SIG_PAD;for(int k=0;k<N_BLOCKS;k++)vc[s[k]]++;}
    p_bg=0;long mc=0;for(int v=0;v<BYTE_VALS;v++)if(vc[v]>mc){mc=vc[v];p_bg=(uint8_t)v;}

    /* Accumulate per-class block counts (used by both global and pair IG) */
    memset(class_block_counts,0,sizeof(class_block_counts));
    for(int i=0;i<TRAIN_N;i++){
        int lbl=train_labels[i];const uint8_t*s=joint_tr+(size_t)i*SIG_PAD;
        for(int k=0;k<N_BLOCKS;k++) class_block_counts[lbl][k][s[k]]++;
    }

    compute_global_ig();

    for(int v=0;v<BYTE_VALS;v++) for(int b=0;b<8;b++) p_nbr[v][b]=(uint8_t)(v^(1<<b));

    memset(p_sz,0,sizeof(p_sz));
    for(int i=0;i<TRAIN_N;i++){const uint8_t*s=joint_tr+(size_t)i*SIG_PAD;for(int k=0;k<N_BLOCKS;k++)if(s[k]!=p_bg)p_sz[k][s[k]]++;}
    uint32_t tot=0;for(int k=0;k<N_BLOCKS;k++)for(int v=0;v<BYTE_VALS;v++){p_off[k][v]=tot;tot+=p_sz[k][v];}
    p_pool=malloc((size_t)tot*sizeof(uint32_t));
    uint32_t(*wp)[BYTE_VALS]=malloc((size_t)N_BLOCKS*BYTE_VALS*4);
    memcpy(wp,p_off,(size_t)N_BLOCKS*BYTE_VALS*4);
    for(int i=0;i<TRAIN_N;i++){const uint8_t*s=joint_tr+(size_t)i*SIG_PAD;for(int k=0;k<N_BLOCKS;k++)if(s[k]!=p_bg)p_pool[wp[k][s[k]]++]=(uint32_t)i;}
    free(wp);
    printf("  Bytepacked index: %u entries (%.1f MB)\n",tot,(double)tot*4/1048576);
}

static void build_pentary(void){
    int pv[5]={-2,-1,0,1,2};
    for(int v=0;v<PENT_BVALS;v++){int p0=(v/25)-2,p1=((v/5)%5)-2,p2=(v%5)-2,orig[3]={p0,p1,p2};int nc=0;
    for(int pos=0;pos<3;pos++)for(int alt=0;alt<5;alt++){if(pv[alt]==orig[pos])continue;int m[3]={orig[0],orig[1],orig[2]};m[pos]=pv[alt];q_nbr[v][nc++]=penc((int8_t)m[0],(int8_t)m[1],(int8_t)m[2]);}q_nbr_cnt[v]=(uint8_t)nc;}
    /* Pentary IG (simplified: use global formula) */
    int cc[N_CLASSES]={0};for(int i=0;i<TRAIN_N;i++)cc[train_labels[i]]++;
    double hc=0;for(int c=0;c<N_CLASSES;c++){double p=(double)cc[c]/TRAIN_N;if(p>0)hc-=p*log2(p);}
    double raw[N_BLOCKS],mx=0;
    for(int k=0;k<N_BLOCKS;k++){
        int *cnt=calloc((size_t)PENT_BVALS*N_CLASSES,sizeof(int));int *vt=calloc(PENT_BVALS,sizeof(int));
        for(int i=0;i<TRAIN_N;i++){int v2=pent_tr[(size_t)i*SIG_PAD+k];cnt[v2*N_CLASSES+train_labels[i]]++;vt[v2]++;}
        double hcond=0;for(int v2=0;v2<PENT_BVALS;v2++){if(!vt[v2]||v2==PENT_BG)continue;double pv=(double)vt[v2]/TRAIN_N,hv=0;for(int c=0;c<N_CLASSES;c++){double pc=(double)cnt[v2*N_CLASSES+c]/vt[v2];if(pc>0)hv-=pc*log2(pc);}hcond+=pv*hv;}
        raw[k]=hc-hcond;if(raw[k]>mx)mx=raw[k];free(cnt);free(vt);
    }
    for(int k=0;k<N_BLOCKS;k++){q_ig[k]=mx>0?(uint16_t)(raw[k]/mx*IG_SCALE+0.5):1;if(!q_ig[k])q_ig[k]=1;}
    memset(q_sz,0,sizeof(q_sz));
    for(int i=0;i<TRAIN_N;i++){const uint8_t*s=pent_tr+(size_t)i*SIG_PAD;for(int k=0;k<N_BLOCKS;k++)if(s[k]!=PENT_BG)q_sz[k][s[k]]++;}
    uint32_t tot=0;for(int k=0;k<N_BLOCKS;k++)for(int v=0;v<PENT_BVALS;v++){q_off[k][v]=tot;tot+=q_sz[k][v];}
    q_pool=malloc((size_t)tot*sizeof(uint32_t));
    uint32_t wpos[N_BLOCKS][PENT_BVALS];
    memcpy(wpos,q_off,sizeof(wpos));
    for(int i=0;i<TRAIN_N;i++){const uint8_t*s=pent_tr+(size_t)i*SIG_PAD;for(int k=0;k<N_BLOCKS;k++)if(s[k]!=PENT_BG)q_pool[wpos[k][s[k]]++]=(uint32_t)i;}
    printf("  Pentary index:    %u entries (%.1f MB)\n",tot,(double)tot*4/1048576);
}

/* ================================================================
 *  AVX2 dot
 * ================================================================ */
static inline int32_t tdot(const int8_t*a,const int8_t*b){
    __m256i acc=_mm256_setzero_si256();
    for(int i=0;i<PADDED;i+=32)acc=_mm256_add_epi8(acc,_mm256_sign_epi8(_mm256_load_si256((const __m256i*)(a+i)),_mm256_load_si256((const __m256i*)(b+i))));
    __m256i lo=_mm256_cvtepi8_epi16(_mm256_castsi256_si128(acc)),hi=_mm256_cvtepi8_epi16(_mm256_extracti128_si256(acc,1));
    __m256i s32=_mm256_madd_epi16(_mm256_add_epi16(lo,hi),_mm256_set1_epi16(1));
    __m128i s=_mm_add_epi32(_mm256_castsi256_si128(s32),_mm256_extracti128_si256(s32,1));
    s=_mm_hadd_epi32(s,s);s=_mm_hadd_epi32(s,s);return _mm_cvtsi128_si32(s);
}

/* ================================================================
 *  Candidate machinery
 * ================================================================ */
typedef struct{uint32_t id,votes;int32_t dot_px,dot_hg,dot_vg;int64_t combined;}cand_t;
static int cmp_votes_d(const void*a,const void*b){return(int)((const cand_t*)b)->votes-(int)((const cand_t*)a)->votes;}
static int cmp_comb_d (const void*a,const void*b){int64_t da=((const cand_t*)a)->combined,db=((const cand_t*)b)->combined;return(db>da)-(db<da);}

static int select_top_k(const uint32_t*votes,int n,cand_t*out,int k){
    uint32_t mx=0;for(int i=0;i<n;i++)if(votes[i]>mx)mx=votes[i];if(!mx)return 0;
    if((size_t)(mx+1)>g_hist_cap){g_hist_cap=(size_t)(mx+1)+4096;free(g_hist);g_hist=malloc(g_hist_cap*sizeof(int));}
    memset(g_hist,0,(mx+1)*sizeof(int));
    for(int i=0;i<n;i++)if(votes[i])g_hist[votes[i]]++;
    int cum=0,thr;for(thr=(int)mx;thr>=1;thr--){cum+=g_hist[thr];if(cum>=k)break;}if(thr<1)thr=1;
    int nc=0;for(int i=0;i<n&&nc<k;i++)if(votes[i]>=(uint32_t)thr)out[nc++]=(cand_t){(uint32_t)i,votes[i],0,0,0,0};
    qsort(out,(size_t)nc,sizeof(cand_t),cmp_votes_d);return nc;
}
static int knn_vote(const cand_t*c,int nc,int k){
    int v[N_CLASSES]={0};if(k>nc)k=nc;
    for(int i=0;i<k;i++)v[train_labels[c[i].id]]++;
    int best=0;for(int c2=1;c2<N_CLASSES;c2++)if(v[c2]>v[best])best=c2;return best;
}
static void compute_dots(cand_t*cands,int nc,int ti){
    const int8_t*tp=tern_test+(size_t)ti*PADDED,*th=hgrad_test+(size_t)ti*PADDED,*tv=vgrad_test+(size_t)ti*PADDED;
    for(int j=0;j<nc;j++){uint32_t id=cands[j].id;
    cands[j].dot_px=tdot(tp,tern_train+(size_t)id*PADDED);
    cands[j].dot_hg=tdot(th,hgrad_train+(size_t)id*PADDED);
    cands[j].dot_vg=tdot(tv,vgrad_train+(size_t)id*PADDED);
    cands[j].combined=(int64_t)256*cands[j].dot_px+(int64_t)192*cands[j].dot_vg;}
}

/* ================================================================
 *  Vote accumulation (generic: takes explicit IG weight array)
 * ================================================================ */
static void vote_with_ig(uint32_t*votes,int img,const uint16_t*ig){
    memset(votes,0,TRAIN_N*sizeof(uint32_t));
    const uint8_t*sig=joint_te+(size_t)img*SIG_PAD;
    for(int k=0;k<N_BLOCKS;k++){
        uint8_t bv=sig[k];if(bv==p_bg)continue;
        uint16_t w=ig[k],wh=w>1?w/2:1;
        {uint32_t off=p_off[k][bv];uint16_t sz=p_sz[k][bv];const uint32_t*ids=p_pool+off;for(uint16_t j=0;j<sz;j++)votes[ids[j]]+=w;}
        for(int nb=0;nb<8;nb++){uint8_t nv=p_nbr[bv][nb];if(nv==p_bg)continue;uint32_t noff=p_off[k][nv];uint16_t nsz=p_sz[k][nv];const uint32_t*nids=p_pool+noff;for(uint16_t j=0;j<nsz;j++)votes[nids[j]]+=wh;}
    }
}
static void vote_pentary(uint32_t*votes,int img){
    memset(votes,0,TRAIN_N*sizeof(uint32_t));
    const uint8_t*sig=pent_te+(size_t)img*SIG_PAD;
    for(int k=0;k<N_BLOCKS;k++){
        uint8_t bv=sig[k];if(bv==PENT_BG)continue;
        uint16_t w=q_ig[k],wh=w>1?w/2:1;
        {uint32_t off=q_off[k][bv];uint16_t sz=q_sz[k][bv];const uint32_t*ids=q_pool+off;for(uint16_t j=0;j<sz;j++)votes[ids[j]]+=w;}
        for(int nb=0;nb<q_nbr_cnt[bv];nb++){uint8_t nv=q_nbr[bv][nb];if(nv==PENT_BG)continue;uint32_t noff=q_off[k][nv];uint16_t nsz=q_sz[k][nv];const uint32_t*nids=q_pool+noff;for(uint16_t j=0;j<nsz;j++)votes[nids[j]]+=wh;}
    }
}

/* ================================================================
 *  Diagnostic: compare global IG vs pair IG for key pairs
 * ================================================================ */
static void print_ig_comparison(void){
    printf("=== IG Weight Comparison: Global vs Pair-Specific ===\n");
    printf("  (Top-5 block positions by each, row×col = spatial location)\n\n");

    /* Key confusion pairs to examine */
    int pairs[5][2]={{4,9},{3,5},{3,8},{1,7},{7,9}};
    const char *pair_names[]={"4↔9","3↔5","3↔8","1↔7","7↔9"};

    for(int pi=0;pi<5;pi++){
        int a=pairs[pi][0],b=pairs[pi][1];
        printf("  Pair %s:\n",pair_names[pi]);

        /* Rank blocks by global IG */
        int grank[N_BLOCKS];for(int k=0;k<N_BLOCKS;k++)grank[k]=k;
        /* Simple insertion sort for top-5 */
        printf("    Global IG top-5:  ");
        int used[N_BLOCKS]={0};
        for(int t=0;t<5;t++){
            int best=-1;uint16_t best_w=0;
            for(int k=0;k<N_BLOCKS;k++){if(!used[k]&&p_ig[k]>best_w){best_w=p_ig[k];best=k;}}
            if(best<0)break;used[best]=1;
            int row=best/BLKS_PER_ROW,col=best%BLKS_PER_ROW;
            printf("  [r%d,c%d]=%d",row,col,best_w);
        }
        printf("\n");

        /* Rank blocks by pair IG */
        printf("    Pair IG  top-5:  ");
        memset(used,0,sizeof(used));
        for(int t=0;t<5;t++){
            int best=-1;uint16_t best_w=0;
            for(int k=0;k<N_BLOCKS;k++){if(!used[k]&&pair_ig[a][b][k]>best_w){best_w=pair_ig[a][b][k];best=k;}}
            if(best<0)break;used[best]=1;
            int row=best/BLKS_PER_ROW,col=best%BLKS_PER_ROW;
            printf("  [r%d,c%d]=%d",row,col,best_w);
        }
        printf("\n");

        /* Compute correlation: do the top blocks overlap? */
        int global_top20[20],pair_top20[20];
        memset(used,0,sizeof(used));
        for(int t=0;t<20;t++){int best=-1;uint16_t bw=0;for(int k=0;k<N_BLOCKS;k++){if(!used[k]&&p_ig[k]>bw){bw=p_ig[k];best=k;}}if(best<0)break;used[best]=1;global_top20[t]=best;}
        memset(used,0,sizeof(used));
        for(int t=0;t<20;t++){int best=-1;uint16_t bw=0;for(int k=0;k<N_BLOCKS;k++){if(!used[k]&&pair_ig[a][b][k]>bw){bw=pair_ig[a][b][k];best=k;}}if(best<0)break;used[best]=1;pair_top20[t]=best;}
        int overlap=0;
        for(int i=0;i<20;i++)for(int j=0;j<20;j++)if(global_top20[i]==pair_top20[j])overlap++;
        printf("    Top-20 overlap: %d/20 (%.0f%% shared)\n\n",overlap,100.0*overlap/20);
    }
}

/* ================================================================
 *  Reporting
 * ================================================================ */
static void print_conf(const uint8_t*preds,const char*label,int correct,double t){
    printf("--- %s ---\n  %.2f%%  (%d errors)  %.2f sec\n",label,100.0*correct/TEST_N,TEST_N-correct,t);
    int conf[N_CLASSES][N_CLASSES];memset(conf,0,sizeof(conf));
    for(int i=0;i<TEST_N;i++)conf[test_labels[i]][preds[i]]++;
    printf("       ");for(int c=0;c<N_CLASSES;c++)printf(" %4d",c);printf("   | Recall\n  -----");
    for(int c=0;c<N_CLASSES;c++)(void)c,printf("-----");printf("---+-------\n");
    for(int r=0;r<N_CLASSES;r++){printf("    %d: ",r);int rt=0;for(int c=0;c<N_CLASSES;c++){printf(" %4d",conf[r][c]);rt+=conf[r][c];}printf("   | %5.1f%%\n",rt>0?100.0*conf[r][r]/rt:0.0);}
    typedef struct{int a,b,n;}pt;pt pairs[45];int np=0;
    for(int a=0;a<N_CLASSES;a++)for(int b=a+1;b<N_CLASSES;b++)pairs[np++]=(pt){a,b,conf[a][b]+conf[b][a]};
    for(int i=0;i<np-1;i++)for(int j=i+1;j<np;j++)if(pairs[j].n>pairs[i].n){pt t=pairs[i];pairs[i]=pairs[j];pairs[j]=t;}
    printf("  Top-5 confused:");for(int i=0;i<5&&i<np;i++)printf("  %d\xe2\x86\x94%d:%d",pairs[i].a,pairs[i].b,pairs[i].n);printf("\n\n");
}

/* ================================================================
 *  Main
 * ================================================================ */
int main(int argc,char**argv){
    double t0=now_sec();
    if(argc>1){data_dir=argv[1];size_t l=strlen(data_dir);if(l&&data_dir[l-1]!='/'){char*buf=malloc(l+2);memcpy(buf,data_dir,l);buf[l]='/';buf[l+1]='\0';data_dir=buf;}}
    const char*ds=strstr(data_dir,"fashion")?"Fashion-MNIST":"MNIST";
    printf("=== SSTT Oracle v3 — Pair-Targeted Re-Vote (%s) ===\n\n",ds);

    load_data();
    tern_train =aligned_alloc(32,(size_t)TRAIN_N*PADDED);tern_test  =aligned_alloc(32,(size_t)TEST_N *PADDED);
    hgrad_train=aligned_alloc(32,(size_t)TRAIN_N*PADDED);hgrad_test =aligned_alloc(32,(size_t)TEST_N *PADDED);
    vgrad_train=aligned_alloc(32,(size_t)TRAIN_N*PADDED);vgrad_test =aligned_alloc(32,(size_t)TEST_N *PADDED);
    pent_train =aligned_alloc(32,(size_t)TRAIN_N*PADDED);pent_test  =aligned_alloc(32,(size_t)TEST_N *PADDED);
    quant_tern(raw_train_img,tern_train,TRAIN_N);quant_tern(raw_test_img,tern_test,TEST_N);
    quant_pent(raw_train_img,pent_train,TRAIN_N);quant_pent(raw_test_img,pent_test,TEST_N);
    gradients(tern_train,hgrad_train,vgrad_train,TRAIN_N);gradients(tern_test,hgrad_test,vgrad_test,TEST_N);
    px_tr=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD);px_te=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    hg_tr=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD);hg_te=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    vg_tr=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD);vg_te=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    bsigs(tern_train,px_tr,TRAIN_N);bsigs(tern_test,px_te,TEST_N);
    bsigs(hgrad_train,hg_tr,TRAIN_N);bsigs(hgrad_test,hg_te,TEST_N);
    bsigs(vgrad_train,vg_tr,TRAIN_N);bsigs(vgrad_test,vg_te,TEST_N);
    ht_tr=aligned_alloc(32,(size_t)TRAIN_N*TRANS_PAD);ht_te=aligned_alloc(32,(size_t)TEST_N*TRANS_PAD);
    vt_tr=aligned_alloc(32,(size_t)TRAIN_N*TRANS_PAD);vt_te=aligned_alloc(32,(size_t)TEST_N*TRANS_PAD);
    trans(px_tr,SIG_PAD,ht_tr,vt_tr,TRAIN_N);trans(px_te,SIG_PAD,ht_te,vt_te,TEST_N);
    joint_tr=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD);joint_te=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    jsigs(joint_tr,TRAIN_N,px_tr,hg_tr,vg_tr,ht_tr,vt_tr);jsigs(joint_te,TEST_N,px_te,hg_te,vg_te,ht_te,vt_te);
    pent_tr=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD);pent_te=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    psigs(pent_train,pent_tr,TRAIN_N);psigs(pent_test,pent_te,TEST_N);

    printf("Building indices...\n");
    build_bytepacked();
    build_pentary();
    printf("Computing pair-specific IG (45 pairs × 252 blocks)...\n");
    double t_pig=now_sec();
    compute_pair_ig();
    printf("  Done (%.2f sec) — pair IG table: %.1f KB\n\n",now_sec()-t_pig,(double)sizeof(pair_ig)/1024);
    printf("  Setup: %.2f sec\n\n",now_sec()-t0);

    /* Diagnostic: verify pair IG differs from global IG */
    print_ig_comparison();

    /* ---- Main test loop ---- */
    uint32_t *votes=calloc(TRAIN_N,sizeof(uint32_t));
    uint32_t *votes2=calloc(TRAIN_N,sizeof(uint32_t));  /* for pentary re-vote */
    cand_t   *cp=malloc(TOP_K*sizeof(cand_t));
    cand_t   *cp2=malloc(TOP_K*sizeof(cand_t));         /* pair re-vote */
    cand_t   *cm=malloc(TOP_K*2*sizeof(cand_t));        /* oracle v2 merge */
    uint8_t  *pA=calloc(TEST_N,1),*pB=calloc(TEST_N,1),*pC=calloc(TEST_N,1);
    int cA=0,cB=0,cC=0;
    int n_unani=0,n_escal=0;
    int cB_u=0,cB_e=0;
    /* Track per-pair improvement */
    int pair_errors_A[N_CLASSES][N_CLASSES]={},pair_errors_B[N_CLASSES][N_CLASSES]={};
    double tA=0,tB=0,tC=0,ts;

    printf("Running oracle v3...\n");
    for(int i=0;i<TEST_N;i++){
        /* --- A: primary only (global IG) --- */
        ts=now_sec();
        vote_with_ig(votes,i,p_ig);
        int np=select_top_k(votes,TRAIN_N,cp,TOP_K);
        compute_dots(cp,np,i);
        qsort(cp,(size_t)np,sizeof(cand_t),cmp_comb_d);
        int pa=knn_vote(cp,np,3);
        pA[i]=(uint8_t)pa;if(pa==test_labels[i])cA++;
        tA+=now_sec()-ts;

        /* Identify split pair */
        int ballot[N_CLASSES]={0},k3=np<3?np:3;
        for(int j=0;j<k3;j++)ballot[train_labels[cp[j].id]]++;
        int top1=0,top2=-1;
        for(int c=1;c<N_CLASSES;c++)if(ballot[c]>ballot[top1])top1=c;
        for(int c=0;c<N_CLASSES;c++)if(c!=top1&&(top2<0||ballot[c]>ballot[top2]))top2=c;
        int unanimous=(ballot[top1]==3);

        /* --- B: oracle v3 — split → pair-specific IG re-vote --- */
        ts=now_sec();
        if(unanimous){
            pB[i]=(uint8_t)pa;if(pa==test_labels[i]){cB++;cB_u++;}n_unani++;
        } else {
            /* Re-vote same bytepacked index with pair-specific IG */
            vote_with_ig(votes,i,pair_ig[top1][top2]);
            int np2=select_top_k(votes,TRAIN_N,cp2,TOP_K);
            compute_dots(cp2,np2,i);
            qsort(cp2,(size_t)np2,sizeof(cand_t),cmp_comb_d);
            int pb=knn_vote(cp2,np2,3);
            pB[i]=(uint8_t)pb;if(pb==test_labels[i]){cB++;cB_e++;}n_escal++;
        }
        tB+=now_sec()-ts;

        /* Track per-pair for B */
        if(pA[i]!=test_labels[i]) pair_errors_A[test_labels[i]][pA[i]]++;
        if(pB[i]!=test_labels[i]) pair_errors_B[test_labels[i]][pB[i]]++;

        /* --- C: oracle v2 (pentary merge, for comparison) --- */
        ts=now_sec();
        if(unanimous){
            pC[i]=(uint8_t)pa;if(pa==test_labels[i])cC++;
        } else {
            vote_pentary(votes2,i);
            int nq=select_top_k(votes2,TRAIN_N,cm,TOP_K);
            compute_dots(cm,nq,i);
            /* Merge: add primary candidates not in pentary pool */
            int nm=nq;
            for(int j=0;j<np&&nm<TOP_K*2;j++){
                int found=0;for(int jj=0;jj<nq;jj++)if(cm[jj].id==cp[j].id){cm[jj].votes+=cp[j].votes;found=1;break;}
                if(!found)cm[nm++]=cp[j];
            }
            for(int j=0;j<nm;j++)cm[j].combined=(int64_t)256*cm[j].dot_px+(int64_t)192*cm[j].dot_vg;
            qsort(cm,(size_t)nm,sizeof(cand_t),cmp_comb_d);
            int pc=knn_vote(cm,nm,3);pC[i]=(uint8_t)pc;if(pc==test_labels[i])cC++;
        }
        tC+=now_sec()-ts;

        if((i+1)%2000==0)fprintf(stderr,"  %d/%d  A:%.2f%%  B:%.2f%%  C:%.2f%%\r",i+1,TEST_N,100.0*cA/(i+1),100.0*cB/(i+1),100.0*cC/(i+1));
    }
    fprintf(stderr,"\n\n");

    print_conf(pA,"A — Primary (global IG)",cA,tA);
    printf("--- B — Oracle v3: pair-specific IG re-vote ---\n");
    printf("  %.2f%%  (%d errors)  %.2f sec\n",100.0*cB/TEST_N,TEST_N-cB,tB);
    printf("  Unanimous: %d (%.1f%%) → %.2f%% acc\n",n_unani,100.0*n_unani/TEST_N,n_unani>0?100.0*cB_u/n_unani:0.0);
    printf("  Escalated: %d (%.1f%%) → %.2f%% acc\n\n",n_escal,100.0*n_escal/TEST_N,n_escal>0?100.0*cB_e/n_escal:0.0);
    print_conf(pB,"B — Oracle v3 confusion",cB,tB);
    print_conf(pC,"C — Oracle v2 (pentary merge, reference)",cC,tC);

    /* Per-confusion-pair improvement breakdown */
    printf("=== Per-Pair Error Delta (A→B) ===\n");
    printf("  %-10s  %-8s  %-8s  %-8s\n","Pair","Errors A","Errors B","Delta");
    typedef struct{int a,b,ea,eb;}row_t;row_t rows[90];int nr=0;
    for(int a=0;a<N_CLASSES;a++)for(int b=0;b<N_CLASSES;b++)if(a!=b&&(pair_errors_A[a][b]>0||pair_errors_B[a][b]>0))
        rows[nr++]=(row_t){a,b,pair_errors_A[a][b],pair_errors_B[a][b]};
    for(int i=0;i<nr-1;i++)for(int j=i+1;j<nr;j++)if(rows[j].ea>rows[i].ea){row_t t=rows[i];rows[i]=rows[j];rows[j]=t;}
    for(int i=0;i<nr&&i<12;i++)if(rows[i].ea>0||rows[i].eb>0)
        printf("  %d\xe2\x86\x92%-7d  %-8d  %-8d  %+d\n",rows[i].a,rows[i].b,rows[i].ea,rows[i].eb,rows[i].ea-rows[i].eb);

    printf("\n=== ORACLE v3 SUMMARY ===\n");
    printf("  A. Primary (global IG):        %.2f%%  (%.2f sec)\n",100.0*cA/TEST_N,tA);
    printf("  B. Oracle v3 (pair IG):        %.2f%%  (%.2f sec, %d escalated)\n",100.0*cB/TEST_N,tB,n_escal);
    printf("  C. Oracle v2 (pentary merge):  %.2f%%  (%.2f sec)\n",100.0*cC/TEST_N,tC);
    printf("\n  B vs A:  %+.2f pp  (%d errors → %d)\n",100.0*(cB-cA)/TEST_N,TEST_N-cA,TEST_N-cB);
    printf("  B vs C:  %+.2f pp\n",100.0*(cB-cC)/TEST_N);
    printf("\nTotal: %.2f sec\n",now_sec()-t0);

    free(votes);free(votes2);free(cp);free(cp2);free(cm);
    free(pA);free(pB);free(pC);free(p_pool);free(q_pool);
    free(joint_tr);free(joint_te);free(pent_tr);free(pent_te);
    free(px_tr);free(px_te);free(hg_tr);free(hg_te);free(vg_tr);free(vg_te);
    free(ht_tr);free(ht_te);free(vt_tr);free(vt_te);
    free(tern_train);free(tern_test);free(hgrad_train);free(hgrad_test);
    free(vgrad_train);free(vgrad_test);free(pent_train);free(pent_test);
    free(raw_train_img);free(raw_test_img);free(train_labels);free(test_labels);
    return 0;
}
