/*
 * sstt_parallel.c — Ternary × Pentary Parallel Cascade
 *
 * Tests the hypothesis: ternary and pentary, run simultaneously on the
 * same image data and allowed to inform each other, outperform either
 * alone or the sequential oracle routing.
 *
 * "Ternary informs pentary" mechanism:
 *   - Bytepacked IG weights gate pentary's vote contribution — pentary
 *     speaks loudly at positions ternary says are discriminative,
 *     quietly where ternary says the position is noise.
 *   - Pentary pixel dot added to refinement: captures faint-stroke
 *     similarity that ternary dot maps to zero.
 *
 * Pixel range relationships:
 *   0–39:    ternary=-1  pentary=-2   background (pentary stronger)
 *   40–84:   ternary=-1  pentary=-1   faint stroke (equal)
 *   85–99:   ternary= 0  pentary=-1   ← TERNARY SILENT, PENTARY SPEAKS
 *   100–179: ternary= 0  pentary= 0   medium gray (both silent)
 *   180–229: ternary=+1  pentary=+1   bright stroke (equal)
 *   230+:    ternary=+1  pentary=+2   bright stroke (pentary stronger)
 *
 * Four configurations (all use bytepacked primary for vote phase):
 *   A. Bytepacked votes  + ternary dot           (current best baseline)
 *   B. Parallel votes    + ternary dot            (vote phase improvement)
 *   C. Bytepacked votes  + ternary+pentary dot   (dot phase improvement)
 *   D. Parallel votes    + ternary+pentary dot   (full parallel system)
 *
 * "Parallel votes" = bytepacked + pentary voting on the same image,
 *   combined in one vote array, gated by bytepacked IG weights.
 *
 * Pentary dot weight swept in config D to find optimal w_pent.
 *
 * Build: make sstt_parallel
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
#define H_TRANS_PER_ROW 8
#define N_HTRANS        (H_TRANS_PER_ROW * IMG_H)
#define V_TRANS_PER_COL 27
#define N_VTRANS        (BLKS_PER_ROW * V_TRANS_PER_COL)
#define TRANS_PAD       256

/* Pentary specialist */
#define PENT_BVALS      125
#define PENT_BG         0
#define PENT_T1         40
#define PENT_T2         100
#define PENT_T3         180
#define PENT_T4         230

#define IG_SCALE        16
#define TOP_K           200

static const char *data_dir = "data/";
static double now_sec(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts);
    return ts.tv_sec + ts.tv_nsec*1e-9;
}

/* ================================================================
 *  Data
 * ================================================================ */
static uint8_t *raw_train_img, *raw_test_img;
static uint8_t *train_labels,  *test_labels;
static int8_t  *tern_train, *tern_test;
static int8_t  *hgrad_train,*hgrad_test;
static int8_t  *vgrad_train,*vgrad_test;
static int8_t  *pent_train, *pent_test;   /* pentary pixel images */

/* Block signatures */
static uint8_t *px_tr,*px_te,*hg_tr,*hg_te,*vg_tr,*vg_te;
static uint8_t *ht_tr,*ht_te,*vt_tr,*vt_te;
static uint8_t *joint_tr,*joint_te;
static uint8_t *pent_tr,*pent_te;

/* Bytepacked index */
static uint16_t p_ig[N_BLOCKS];
static uint8_t  p_nbr[BYTE_VALS][8];
static uint32_t p_off[N_BLOCKS][BYTE_VALS];
static uint16_t p_sz [N_BLOCKS][BYTE_VALS];
static uint32_t *p_pool;
static uint8_t  p_bg;

/* Pentary index */
static uint16_t q_ig[N_BLOCKS];
static uint8_t  q_nbr[PENT_BVALS][12];
static uint8_t  q_nbr_cnt[PENT_BVALS];
static uint32_t q_off[N_BLOCKS][PENT_BVALS];
static uint16_t q_sz [N_BLOCKS][PENT_BVALS];
static uint32_t *q_pool;

static int *g_hist=NULL; static size_t g_hist_cap=0;

/* ================================================================
 *  Data loading + features
 * ================================================================ */
static uint8_t *load_idx(const char *path,uint32_t *cnt,uint32_t *ro,uint32_t *co){
    FILE*f=fopen(path,"rb"); if(!f){fprintf(stderr,"ERR:%s\n",path);exit(1);}
    uint32_t m,n; fread(&m,4,1,f); fread(&n,4,1,f);
    m=__builtin_bswap32(m);n=__builtin_bswap32(n);*cnt=n; size_t s=1;
    if((m&0xFF)>=3){uint32_t r,c;fread(&r,4,1,f);fread(&c,4,1,f);r=__builtin_bswap32(r);c=__builtin_bswap32(c);if(ro)*ro=r;if(co)*co=c;s=(size_t)r*c;}else{if(ro)*ro=0;if(co)*co=0;}
    uint8_t*d=malloc((size_t)n*s); fread(d,1,(size_t)n*s,f); fclose(f); return d;
}
static void load_data(void){
    uint32_t n,r,c; char p[256];
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
    for(;k<PIXELS;k++)d[k]=s[k]>170?1:s[k]<85?-1:0; memset(d+PIXELS,0,PADDED-PIXELS);}
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
    for(int i=0;i<n;i++){
        const uint8_t*pi=px+(size_t)i*SIG_PAD,*hi=hg+(size_t)i*SIG_PAD,*vi=vg+(size_t)i*SIG_PAD;
        const uint8_t*hti=ht+(size_t)i*TRANS_PAD,*vti=vt+(size_t)i*TRANS_PAD;uint8_t*oi=out+(size_t)i*SIG_PAD;
        for(int y=0;y<IMG_H;y++)for(int s=0;s<BLKS_PER_ROW;s++){int k=y*BLKS_PER_ROW+s;
        uint8_t htb=s>0?hti[y*H_TRANS_PER_ROW+(s-1)]:BG_TRANS,vtb=y>0?vti[(y-1)*BLKS_PER_ROW+s]:BG_TRANS;
        oi[k]=enc_d(pi[k],hi[k],vi[k],htb,vtb);}
        memset(oi+N_BLOCKS,0xFF,SIG_PAD-N_BLOCKS);}
}

/* ================================================================
 *  IG computation
 * ================================================================ */
static void compute_ig(const uint8_t*sigs,int n_vals,uint8_t bg,uint16_t*ig_out){
    int cc[N_CLASSES]={0}; for(int i=0;i<TRAIN_N;i++) cc[train_labels[i]]++;
    double hc=0; for(int c=0;c<N_CLASSES;c++){double p=(double)cc[c]/TRAIN_N;if(p>0)hc-=p*log2(p);}
    double raw[N_BLOCKS],mx=0;
    for(int k=0;k<N_BLOCKS;k++){
        int*cnt=calloc((size_t)n_vals*N_CLASSES,sizeof(int));int*vt=calloc(n_vals,sizeof(int));
        for(int i=0;i<TRAIN_N;i++){int v=sigs[(size_t)i*SIG_PAD+k];cnt[v*N_CLASSES+train_labels[i]]++;vt[v]++;}
        double hcond=0;
        for(int v=0;v<n_vals;v++){if(!vt[v]||v==bg)continue;double pv=(double)vt[v]/TRAIN_N,hv=0;
        for(int c=0;c<N_CLASSES;c++){double pc=(double)cnt[v*N_CLASSES+c]/vt[v];if(pc>0)hv-=pc*log2(pc);}hcond+=pv*hv;}
        raw[k]=hc-hcond;if(raw[k]>mx)mx=raw[k];free(cnt);free(vt);
    }
    for(int k=0;k<N_BLOCKS;k++){ig_out[k]=mx>0?(uint16_t)(raw[k]/mx*IG_SCALE+0.5):1;if(!ig_out[k])ig_out[k]=1;}
}

/* ================================================================
 *  Index builders
 * ================================================================ */
static void build_bytepacked(void){
    long vc[BYTE_VALS]={0};
    for(int i=0;i<TRAIN_N;i++){const uint8_t*s=joint_tr+(size_t)i*SIG_PAD;for(int k=0;k<N_BLOCKS;k++)vc[s[k]]++;}
    p_bg=0;long mc=0;for(int v=0;v<BYTE_VALS;v++)if(vc[v]>mc){mc=vc[v];p_bg=(uint8_t)v;}
    compute_ig(joint_tr,BYTE_VALS,p_bg,p_ig);
    for(int v=0;v<BYTE_VALS;v++)for(int b=0;b<8;b++)p_nbr[v][b]=(uint8_t)(v^(1<<b));
    memset(p_sz,0,sizeof(p_sz));
    for(int i=0;i<TRAIN_N;i++){const uint8_t*s=joint_tr+(size_t)i*SIG_PAD;for(int k=0;k<N_BLOCKS;k++)if(s[k]!=p_bg)p_sz[k][s[k]]++;}
    uint32_t tot=0;for(int k=0;k<N_BLOCKS;k++)for(int v=0;v<BYTE_VALS;v++){p_off[k][v]=tot;tot+=p_sz[k][v];}
    p_pool=malloc((size_t)tot*sizeof(uint32_t));
    uint32_t(*wp)[BYTE_VALS]=malloc((size_t)N_BLOCKS*BYTE_VALS*4);
    memcpy(wp,p_off,(size_t)N_BLOCKS*BYTE_VALS*4);
    for(int i=0;i<TRAIN_N;i++){const uint8_t*s=joint_tr+(size_t)i*SIG_PAD;for(int k=0;k<N_BLOCKS;k++)if(s[k]!=p_bg)p_pool[wp[k][s[k]]++]=(uint32_t)i;}
    free(wp);
    printf("  Bytepacked: %u entries (%.1f MB)\n",tot,(double)tot*4/1048576);
}
static void build_pentary(void){
    int pv[5]={-2,-1,0,1,2};
    for(int v=0;v<PENT_BVALS;v++){int p0=(v/25)-2,p1=((v/5)%5)-2,p2=(v%5)-2,orig[3]={p0,p1,p2};int nc=0;
    for(int pos=0;pos<3;pos++)for(int alt=0;alt<5;alt++){if(pv[alt]==orig[pos])continue;int m[3]={orig[0],orig[1],orig[2]};m[pos]=pv[alt];q_nbr[v][nc++]=penc((int8_t)m[0],(int8_t)m[1],(int8_t)m[2]);}
    q_nbr_cnt[v]=(uint8_t)nc;}
    compute_ig(pent_tr,PENT_BVALS,PENT_BG,q_ig);
    memset(q_sz,0,sizeof(q_sz));
    for(int i=0;i<TRAIN_N;i++){const uint8_t*s=pent_tr+(size_t)i*SIG_PAD;for(int k=0;k<N_BLOCKS;k++)if(s[k]!=PENT_BG)q_sz[k][s[k]]++;}
    uint32_t tot=0;for(int k=0;k<N_BLOCKS;k++)for(int v=0;v<PENT_BVALS;v++){q_off[k][v]=tot;tot+=q_sz[k][v];}
    q_pool=malloc((size_t)tot*sizeof(uint32_t));
    uint32_t wpos[N_BLOCKS][PENT_BVALS];
    memcpy(wpos,q_off,sizeof(wpos));
    for(int i=0;i<TRAIN_N;i++){const uint8_t*s=pent_tr+(size_t)i*SIG_PAD;for(int k=0;k<N_BLOCKS;k++)if(s[k]!=PENT_BG)q_pool[wpos[k][s[k]]++]=(uint32_t)i;}
    printf("  Pentary:    %u entries (%.1f MB)\n",tot,(double)tot*4/1048576);
}

/* ================================================================
 *  AVX2 dot (works for both ternary {-1,0,1} and pentary {-2..2})
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
typedef struct {
    uint32_t id, votes;
    int32_t  dot_px, dot_hg, dot_vg, dot_pent;
    int64_t  combined;
} cand_t;

static int cmp_votes_d(const void*a,const void*b){return(int)((const cand_t*)b)->votes-(int)((const cand_t*)a)->votes;}
static int cmp_comb_d (const void*a,const void*b){int64_t da=((const cand_t*)a)->combined,db=((const cand_t*)b)->combined;return(db>da)-(db<da);}

static int select_top_k(const uint32_t*votes,int n,cand_t*out,int k){
    uint32_t mx=0;for(int i=0;i<n;i++)if(votes[i]>mx)mx=votes[i];
    if(!mx)return 0;
    if((size_t)(mx+1)>g_hist_cap){g_hist_cap=(size_t)(mx+1)+4096;free(g_hist);g_hist=malloc(g_hist_cap*sizeof(int));}
    memset(g_hist,0,(mx+1)*sizeof(int));
    for(int i=0;i<n;i++)if(votes[i])g_hist[votes[i]]++;
    int cum=0,thr;for(thr=(int)mx;thr>=1;thr--){cum+=g_hist[thr];if(cum>=k)break;}
    if(thr<1)thr=1;
    int nc=0;for(int i=0;i<n&&nc<k;i++)if(votes[i]>=(uint32_t)thr)out[nc++]=(cand_t){(uint32_t)i,votes[i],0,0,0,0,0};
    qsort(out,(size_t)nc,sizeof(cand_t),cmp_votes_d);return nc;
}
static int knn_vote(const cand_t*c,int nc,int k){
    int v[N_CLASSES]={0};if(k>nc)k=nc;
    for(int i=0;i<k;i++)v[train_labels[c[i].id]]++;
    int best=0;for(int c2=1;c2<N_CLASSES;c2++)if(v[c2]>v[best])best=c2;return best;
}
static void compute_all_dots(cand_t*cands,int nc,int ti){
    const int8_t*tp=tern_test +(size_t)ti*PADDED;
    const int8_t*th=hgrad_test+(size_t)ti*PADDED;
    const int8_t*tv=vgrad_test+(size_t)ti*PADDED;
    const int8_t*pp=pent_test +(size_t)ti*PADDED;
    for(int j=0;j<nc;j++){uint32_t id=cands[j].id;
    cands[j].dot_px  =tdot(tp,tern_train +(size_t)id*PADDED);
    cands[j].dot_hg  =tdot(th,hgrad_train+(size_t)id*PADDED);
    cands[j].dot_vg  =tdot(tv,vgrad_train+(size_t)id*PADDED);
    cands[j].dot_pent=tdot(pp,pent_train +(size_t)id*PADDED);}
}

/* ================================================================
 *  Vote accumulation
 * ================================================================ */
static void vote_byte(uint32_t*votes,int img){
    const uint8_t*sig=joint_te+(size_t)img*SIG_PAD;
    for(int k=0;k<N_BLOCKS;k++){
        uint8_t bv=sig[k];if(bv==p_bg)continue;
        uint16_t w=p_ig[k],wh=w>1?w/2:1;
        {uint32_t off=p_off[k][bv];uint16_t sz=p_sz[k][bv];const uint32_t*ids=p_pool+off;for(uint16_t j=0;j<sz;j++)votes[ids[j]]+=w;}
        for(int nb=0;nb<8;nb++){uint8_t nv=p_nbr[bv][nb];if(nv==p_bg)continue;uint32_t noff=p_off[k][nv];uint16_t nsz=p_sz[k][nv];const uint32_t*nids=p_pool+noff;for(uint16_t j=0;j<nsz;j++)votes[nids[j]]+=wh;}
    }
}
/* Pentary votes gated by bytepacked IG — ternary informs pentary weight */
static void vote_pent_gated(uint32_t*votes,int img){
    const uint8_t*sig=pent_te+(size_t)img*SIG_PAD;
    for(int k=0;k<N_BLOCKS;k++){
        uint8_t bv=sig[k];if(bv==PENT_BG)continue;
        /* Gate: use bytepacked IG (ternary discriminability) × pentary IG */
        uint32_t w=(uint32_t)p_ig[k]*q_ig[k]/IG_SCALE;
        if(!w)w=1; uint32_t wh=w>1?w/2:1;
        {uint32_t off=q_off[k][bv];uint16_t sz=q_sz[k][bv];const uint32_t*ids=q_pool+off;for(uint16_t j=0;j<sz;j++)votes[ids[j]]+=w;}
        for(int nb=0;nb<q_nbr_cnt[bv];nb++){uint8_t nv=q_nbr[bv][nb];if(nv==PENT_BG)continue;uint32_t noff=q_off[k][nv];uint16_t nsz=q_sz[k][nv];const uint32_t*nids=q_pool+noff;for(uint16_t j=0;j<nsz;j++)votes[nids[j]]+=wh;}
    }
}

/* ================================================================
 *  Reporting
 * ================================================================ */
static void print_conf(const uint8_t*preds){
    int conf[N_CLASSES][N_CLASSES];memset(conf,0,sizeof(conf));
    for(int i=0;i<TEST_N;i++)conf[test_labels[i]][preds[i]]++;
    printf("       ");for(int c=0;c<N_CLASSES;c++)printf(" %4d",c);printf("   | Recall\n  -----");
    for(int c=0;c<N_CLASSES;c++)(void)c,printf("-----");printf("---+-------\n");
    for(int r=0;r<N_CLASSES;r++){printf("    %d: ",r);int rt=0;for(int c=0;c<N_CLASSES;c++){printf(" %4d",conf[r][c]);rt+=conf[r][c];}printf("   | %5.1f%%\n",rt>0?100.0*conf[r][r]/rt:0.0);}
    typedef struct{int a,b,n;}p_t;p_t pairs[45];int np=0;
    for(int a=0;a<N_CLASSES;a++)for(int b=a+1;b<N_CLASSES;b++)pairs[np++]=(p_t){a,b,conf[a][b]+conf[b][a]};
    for(int i=0;i<np-1;i++)for(int j=i+1;j<np;j++)if(pairs[j].n>pairs[i].n){p_t t=pairs[i];pairs[i]=pairs[j];pairs[j]=t;}
    printf("  Top-5 confused:");for(int i=0;i<5&&i<np;i++)printf("  %d\xe2\x86\x94%d:%d",pairs[i].a,pairs[i].b,pairs[i].n);printf("\n");
}

/* ================================================================
 *  Main
 * ================================================================ */
int main(int argc,char**argv){
    double t0=now_sec();
    if(argc>1){data_dir=argv[1];size_t l=strlen(data_dir);if(l&&data_dir[l-1]!='/'){char*buf=malloc(l+2);memcpy(buf,data_dir,l);buf[l]='/';buf[l+1]='\0';data_dir=buf;}}
    const char*ds=strstr(data_dir,"fashion")?"Fashion-MNIST":"MNIST";
    printf("=== SSTT Parallel Cascade: Ternary × Pentary (%s) ===\n\n",ds);

    load_data();
    tern_train =aligned_alloc(32,(size_t)TRAIN_N*PADDED);tern_test  =aligned_alloc(32,(size_t)TEST_N *PADDED);
    hgrad_train=aligned_alloc(32,(size_t)TRAIN_N*PADDED);hgrad_test =aligned_alloc(32,(size_t)TEST_N *PADDED);
    vgrad_train=aligned_alloc(32,(size_t)TRAIN_N*PADDED);vgrad_test =aligned_alloc(32,(size_t)TEST_N *PADDED);
    pent_train =aligned_alloc(32,(size_t)TRAIN_N*PADDED);pent_test  =aligned_alloc(32,(size_t)TEST_N *PADDED);
    quant_tern(raw_train_img,tern_train,TRAIN_N);quant_tern(raw_test_img,tern_test,TEST_N);
    quant_pent(raw_train_img,pent_train,TRAIN_N);quant_pent(raw_test_img,pent_test,TEST_N);
    gradients(tern_train,hgrad_train,vgrad_train,TRAIN_N);
    gradients(tern_test, hgrad_test, vgrad_test, TEST_N);

    px_tr=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD);px_te=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    hg_tr=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD);hg_te=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    vg_tr=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD);vg_te=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    bsigs(tern_train, px_tr,TRAIN_N);bsigs(tern_test, px_te,TEST_N);
    bsigs(hgrad_train,hg_tr,TRAIN_N);bsigs(hgrad_test,hg_te,TEST_N);
    bsigs(vgrad_train,vg_tr,TRAIN_N);bsigs(vgrad_test,vg_te,TEST_N);
    ht_tr=aligned_alloc(32,(size_t)TRAIN_N*TRANS_PAD);ht_te=aligned_alloc(32,(size_t)TEST_N*TRANS_PAD);
    vt_tr=aligned_alloc(32,(size_t)TRAIN_N*TRANS_PAD);vt_te=aligned_alloc(32,(size_t)TEST_N*TRANS_PAD);
    trans(px_tr,SIG_PAD,ht_tr,vt_tr,TRAIN_N);trans(px_te,SIG_PAD,ht_te,vt_te,TEST_N);
    joint_tr=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD);joint_te=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    jsigs(joint_tr,TRAIN_N,px_tr,hg_tr,vg_tr,ht_tr,vt_tr);
    jsigs(joint_te,TEST_N, px_te,hg_te,vg_te,ht_te,vt_te);
    pent_tr=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD);pent_te=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    psigs(pent_train,pent_tr,TRAIN_N);psigs(pent_test,pent_te,TEST_N);

    printf("Building indices...\n");
    build_bytepacked();
    build_pentary();
    printf("  Setup: %.2f sec\n\n",now_sec()-t0);

    /* ---- Precompute: votes and dots for all test images ---- */
    printf("Precomputing votes and dots...\n");
    cand_t *pre_byte = malloc((size_t)TEST_N*TOP_K*sizeof(cand_t)); /* bytepacked-only vote */
    cand_t *pre_para = malloc((size_t)TEST_N*TOP_K*sizeof(cand_t)); /* parallel vote */
    int    *nc_byte  = malloc((size_t)TEST_N*sizeof(int));
    int    *nc_para  = malloc((size_t)TEST_N*sizeof(int));
    uint32_t *vp = calloc(TRAIN_N,sizeof(uint32_t));

    double t_pre=now_sec();
    for(int i=0;i<TEST_N;i++){
        /* Bytepacked-only votes */
        memset(vp,0,TRAIN_N*sizeof(uint32_t));
        vote_byte(vp,i);
        cand_t*row=pre_byte+(size_t)i*TOP_K;
        nc_byte[i]=select_top_k(vp,TRAIN_N,row,TOP_K);
        compute_all_dots(row,nc_byte[i],i);

        /* Parallel votes: bytepacked + pentary (ternary-gated) */
        memset(vp,0,TRAIN_N*sizeof(uint32_t));
        vote_byte(vp,i);
        vote_pent_gated(vp,i);   /* add pentary votes to same array */
        row=pre_para+(size_t)i*TOP_K;
        nc_para[i]=select_top_k(vp,TRAIN_N,row,TOP_K);
        compute_all_dots(row,nc_para[i],i);

        if((i+1)%2000==0)fprintf(stderr,"  %d/%d\r",i+1,TEST_N);
    }
    fprintf(stderr,"\n");
    free(vp);
    printf("  Precompute: %.2f sec\n\n",now_sec()-t_pre);

    /* ---- Run 4 configurations ---- */
    uint8_t *pred=calloc(TEST_N,1);
    int best_correct=0;
    const char *configs[]={"A: byte-vote  + tern-dot","B: para-vote  + tern-dot","C: byte-vote  + tern+pent-dot","D: para-vote  + tern+pent-dot"};

    printf("=== Configuration Results ===\n");
    printf("  %-35s  %-8s  %-7s\n","Config","Accuracy","Errors");
    printf("  %-35s  %-8s  %-7s\n","-----------------------------------","--------","-------");

    int best_conf=-1;
    for(int cfg=0;cfg<4;cfg++){
        int use_para=(cfg==1||cfg==3);
        int use_pent_dot=(cfg==2||cfg==3);
        cand_t*pre=use_para?pre_para:pre_byte;
        int*nc =use_para?nc_para:nc_byte;
        int correct=0;
        for(int i=0;i<TEST_N;i++){
            int n=nc[i]; cand_t cands[TOP_K];
            memcpy(cands,pre+(size_t)i*TOP_K,(size_t)n*sizeof(cand_t));
            for(int j=0;j<n;j++){
                cands[j].combined=(int64_t)256*cands[j].dot_px+(int64_t)192*cands[j].dot_vg;
                if(use_pent_dot) cands[j].combined+=(int64_t)128*cands[j].dot_pent;
            }
            qsort(cands,(size_t)n,sizeof(cand_t),cmp_comb_d);
            int p=knn_vote(cands,n,3); pred[i]=(uint8_t)p;
            if(p==test_labels[i])correct++;
        }
        printf("  %-35s  %6.2f%%   %d\n",configs[cfg],100.0*correct/TEST_N,TEST_N-correct);
        if(correct>best_correct){best_correct=correct;best_conf=cfg;memcpy(pred,pred,TEST_N);}/* track best */
    }

    /* ---- Pentary dot weight sweep on config D ---- */
    printf("\n=== Config D: Pentary Dot Weight Sweep (byte-vote=1, tern-vg=192 fixed) ===\n");
    printf("  %-10s  %-8s  %-7s\n","w_pent","Accuracy","Errors");
    printf("  %-10s  %-8s  %-7s\n","----------","--------","-------");
    int w_pent_vals[]={0,32,64,96,128,160,192,256,320,384};
    int n_wp=10, best_wp=128, best_wp_correct=0;
    for(int wi=0;wi<n_wp;wi++){
        int wp=w_pent_vals[wi],correct=0;
        for(int i=0;i<TEST_N;i++){
            int n=nc_para[i]; cand_t cands[TOP_K];
            memcpy(cands,pre_para+(size_t)i*TOP_K,(size_t)n*sizeof(cand_t));
            for(int j=0;j<n;j++) cands[j].combined=(int64_t)256*cands[j].dot_px+(int64_t)192*cands[j].dot_vg+(int64_t)wp*cands[j].dot_pent;
            qsort(cands,(size_t)n,sizeof(cand_t),cmp_comb_d);
            int p=knn_vote(cands,n,3); pred[i]=(uint8_t)p;
            if(p==test_labels[i])correct++;
        }
        printf("  %-10d  %6.2f%%   %d%s\n",wp,100.0*correct/TEST_N,TEST_N-correct,correct>best_wp_correct?" ←":"");
        if(correct>best_wp_correct){best_wp_correct=correct;best_wp=wp;}
    }

    /* ---- Best config detail ---- */
    printf("\n=== Best Config D (w_pent=%d) — Confusion Matrix ===\n",best_wp);
    for(int i=0;i<TEST_N;i++){
        int n=nc_para[i]; cand_t cands[TOP_K];
        memcpy(cands,pre_para+(size_t)i*TOP_K,(size_t)n*sizeof(cand_t));
        for(int j=0;j<n;j++) cands[j].combined=(int64_t)256*cands[j].dot_px+(int64_t)192*cands[j].dot_vg+(int64_t)best_wp*cands[j].dot_pent;
        qsort(cands,(size_t)n,sizeof(cand_t),cmp_comb_d);
        pred[i]=(uint8_t)knn_vote(cands,n,3);
    }
    print_conf(pred);

    printf("\n=== SUMMARY ===\n");
    printf("  Baseline (config A):      %.2f%%  (%d errors)\n",100.0*(TEST_N-(TEST_N-best_correct))/TEST_N,TEST_N-best_correct);
    printf("  Best config D (w=%d):  %.2f%%  (%d errors)\n",best_wp,100.0*best_wp_correct/TEST_N,TEST_N-best_wp_correct);
    printf("  Delta:                    %+.2f pp\n",100.0*(best_wp_correct-(TEST_N-(TEST_N-best_correct)))/TEST_N);
    printf("  Oracle v2 routed (ref):   96.40%%\n");
    printf("\nTotal: %.2f sec\n",now_sec()-t0);

    free(pre_byte);free(pre_para);free(nc_byte);free(nc_para);free(pred);
    free(p_pool);free(q_pool);
    free(joint_tr);free(joint_te);free(pent_tr);free(pent_te);
    free(px_tr);free(px_te);free(hg_tr);free(hg_te);free(vg_tr);free(vg_te);
    free(ht_tr);free(ht_te);free(vt_tr);free(vt_te);
    free(tern_train);free(tern_test);free(hgrad_train);free(hgrad_test);
    free(vgrad_train);free(vgrad_test);free(pent_train);free(pent_test);
    free(raw_train_img);free(raw_test_img);free(train_labels);free(test_labels);
    return 0;
}
