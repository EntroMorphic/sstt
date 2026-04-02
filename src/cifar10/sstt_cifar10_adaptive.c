/*
 * sstt_cifar10_adaptive.c — Adaptive Ternary Quantization
 *
 * Replace fixed 85/170 thresholds with per-image percentile thresholds.
 * Each image gets P33/P67 of its own pixel distribution as thresholds.
 * Result: exactly 1/3 of each trit per image. Brightness/contrast
 * variation eliminated. Ternary pattern encodes only relative structure.
 *
 * Tests:
 *   1. Adaptive on flattened RGB (Bayesian + cascade)
 *   2. Per-channel adaptive (R,G,B normalized independently)
 *   3. Adaptive + full stack (topo + Bayesian prior)
 *
 * Build: gcc -O3 -mavx2 -march=native -o sstt_cifar10_adaptive src/sstt_cifar10_adaptive.c -lm
 */

#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define TRAIN_N 50000
#define TEST_N  10000
#define IMG_W   96
#define IMG_H   32
#define PIXELS  3072
#define PADDED  3072
#define N_CLASSES 10
#define CLS_PAD 16
#define BLKS_PER_ROW 32
#define N_BLOCKS 1024
#define SIG_PAD  1024
#define BYTE_VALS 256
#define BG_TRANS 13
#define H_TRANS_PER_ROW 31
#define N_HTRANS (H_TRANS_PER_ROW*IMG_H)
#define V_TRANS_PER_COL 31
#define N_VTRANS (BLKS_PER_ROW*V_TRANS_PER_COL)
#define TRANS_PAD 992
#define IG_SCALE 16
#define TOP_K 200
#define SPATIAL_W 32
#define SPATIAL_H 32
#define GRAY_PIXELS 1024
#define GRAY_PADDED 1024
#define MAX_REGIONS 16

static const char *data_dir="data-cifar10/";
static const char *cn[]={"airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"};
static double now_sec(void){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return ts.tv_sec+ts.tv_nsec*1e-9;}

static uint8_t *raw_train,*raw_test,*raw_gray_tr,*raw_gray_te,*train_labels,*test_labels;

/* ================================================================
 *  Adaptive quantization: per-image percentile thresholds
 * ================================================================ */

static int cmp_u8(const void*a,const void*b){return *(const uint8_t*)a-*(const uint8_t*)b;}

static void adaptive_quantize(const uint8_t *src, int8_t *dst, int n, int pixels, int padded) {
    uint8_t *sorted = malloc(pixels);
    for (int img = 0; img < n; img++) {
        const uint8_t *s = src + (size_t)img * pixels;
        int8_t *d = dst + (size_t)img * padded;
        /* Find P33 and P67 for this image */
        memcpy(sorted, s, pixels);
        qsort(sorted, pixels, 1, cmp_u8);
        uint8_t p33 = sorted[pixels / 3];
        uint8_t p67 = sorted[2 * pixels / 3];
        /* Handle degenerate case: all same value */
        if (p33 == p67) { p33 = p33 > 0 ? p33 - 1 : 0; p67 = p67 < 255 ? p67 + 1 : 255; }
        /* Quantize */
        for (int i = 0; i < pixels; i++)
            d[i] = s[i] > p67 ? 1 : s[i] < p33 ? -1 : 0;
        memset(d + pixels, 0, padded - pixels);
    }
    free(sorted);
}

/* Per-channel adaptive: compute percentiles for each RGB channel separately */
static void adaptive_quantize_perchannel(const uint8_t *src, int8_t *dst, int n) {
    /* src is flattened RGB: for each pixel position, (R,G,B) interleaved
       Each image has 32 rows × 32 pixels × 3 channels = 3072 bytes
       R values at positions 0,3,6,...; G at 1,4,7,...; B at 2,5,8,... */
    uint8_t *channel_vals = malloc(1024); /* max 1024 values per channel */
    for (int img = 0; img < n; img++) {
        const uint8_t *s = src + (size_t)img * PIXELS;
        int8_t *d = dst + (size_t)img * PADDED;
        /* Extract and sort each channel */
        uint8_t p33[3], p67[3];
        for (int ch = 0; ch < 3; ch++) {
            int cnt = 0;
            for (int y = 0; y < 32; y++)
                for (int x = 0; x < 32; x++)
                    channel_vals[cnt++] = s[y * 96 + x * 3 + ch];
            qsort(channel_vals, cnt, 1, cmp_u8);
            p33[ch] = channel_vals[cnt / 3];
            p67[ch] = channel_vals[2 * cnt / 3];
            if (p33[ch] == p67[ch]) { p33[ch] = p33[ch] > 0 ? p33[ch] - 1 : 0; p67[ch] = p67[ch] < 255 ? p67[ch] + 1 : 255; }
        }
        /* Quantize each pixel using its channel's thresholds */
        for (int y = 0; y < 32; y++)
            for (int x = 0; x < 32; x++)
                for (int ch = 0; ch < 3; ch++) {
                    int idx = y * 96 + x * 3 + ch;
                    d[idx] = s[idx] > p67[ch] ? 1 : s[idx] < p33[ch] ? -1 : 0;
                }
    }
    free(channel_vals);
}

/* Standard fixed quantization for comparison */
static void fixed_quantize(const uint8_t *src, int8_t *dst, int n, int pixels, int padded) {
    const __m256i bi=_mm256_set1_epi8((char)0x80),th=_mm256_set1_epi8((char)(170^0x80)),
        tl=_mm256_set1_epi8((char)(85^0x80)),on=_mm256_set1_epi8(1);
    for(int i2=0;i2<n;i2++){const uint8_t*si=src+(size_t)i2*pixels;int8_t*di=dst+(size_t)i2*padded;int i;
        for(i=0;i+32<=pixels;i+=32){__m256i p=_mm256_loadu_si256((const __m256i*)(si+i));__m256i sp=_mm256_xor_si256(p,bi);
            _mm256_storeu_si256((__m256i*)(di+i),_mm256_sub_epi8(_mm256_and_si256(_mm256_cmpgt_epi8(sp,th),on),
                _mm256_and_si256(_mm256_cmpgt_epi8(tl,sp),on)));}for(;i<pixels;i++)di[i]=si[i]>170?1:si[i]<85?-1:0;memset(di+pixels,0,padded-pixels);}}

/* ================================================================
 *  Core pipeline functions (same as other experiments)
 * ================================================================ */
static inline int8_t ct(int v){return v>0?1:v<0?-1:0;}
static void grads(const int8_t*t,int8_t*h,int8_t*v,int n,int w,int ht,int pd){
    for(int i2=0;i2<n;i2++){const int8_t*ti=t+(size_t)i2*pd;int8_t*hi=h+(size_t)i2*pd,*vi=v+(size_t)i2*pd;
        for(int y=0;y<ht;y++){for(int x=0;x<w-1;x++)hi[y*w+x]=ct(ti[y*w+x+1]-ti[y*w+x]);hi[y*w+w-1]=0;}
        for(int y=0;y<ht-1;y++)for(int x=0;x<w;x++)vi[y*w+x]=ct(ti[(y+1)*w+x]-ti[y*w+x]);memset(vi+(ht-1)*w,0,w);}}
static inline uint8_t be(int8_t a,int8_t b,int8_t c){return(uint8_t)((a+1)*9+(b+1)*3+(c+1));}
static void bsf(const int8_t*d,uint8_t*s,int n){for(int i=0;i<n;i++){const int8_t*im=d+(size_t)i*PADDED;
    uint8_t*si=s+(size_t)i*SIG_PAD;for(int y=0;y<IMG_H;y++)for(int s2=0;s2<BLKS_PER_ROW;s2++){
        int b2=y*IMG_W+s2*3;si[y*BLKS_PER_ROW+s2]=be(im[b2],im[b2+1],im[b2+2]);}}}
static inline uint8_t te2(uint8_t a,uint8_t b){int8_t a0=(a/9)-1,a1=((a/3)%3)-1,a2=(a%3)-1,
    b0=(b/9)-1,b1=((b/3)%3)-1,b2=(b%3)-1;return be(ct(b0-a0),ct(b1-a1),ct(b2-a2));}
static void trf(const uint8_t*bsi,uint8_t*ht,uint8_t*vt,int n){
    for(int i=0;i<n;i++){const uint8_t*s=bsi+(size_t)i*SIG_PAD;uint8_t*h=ht+(size_t)i*TRANS_PAD,*v=vt+(size_t)i*TRANS_PAD;
        for(int y=0;y<IMG_H;y++)for(int ss=0;ss<H_TRANS_PER_ROW;ss++)h[y*H_TRANS_PER_ROW+ss]=te2(s[y*BLKS_PER_ROW+ss],s[y*BLKS_PER_ROW+ss+1]);
        memset(h+N_HTRANS,0xFF,TRANS_PAD-N_HTRANS);
        for(int y=0;y<V_TRANS_PER_COL;y++)for(int ss=0;ss<BLKS_PER_ROW;ss++)v[y*BLKS_PER_ROW+ss]=te2(s[y*BLKS_PER_ROW+ss],s[(y+1)*BLKS_PER_ROW+ss]);
        memset(v+N_VTRANS,0xFF,TRANS_PAD-N_VTRANS);}}
static void jsf(uint8_t*o,int n,const uint8_t*px,const uint8_t*hg,const uint8_t*vg,const uint8_t*ht,const uint8_t*vt){
    for(int i=0;i<n;i++){const uint8_t*pi=px+(size_t)i*SIG_PAD,*hi=hg+(size_t)i*SIG_PAD,*vi=vg+(size_t)i*SIG_PAD;
        const uint8_t*hti=ht+(size_t)i*TRANS_PAD,*vti=vt+(size_t)i*TRANS_PAD;uint8_t*oi=o+(size_t)i*SIG_PAD;
        for(int y=0;y<IMG_H;y++)for(int s=0;s<BLKS_PER_ROW;s++){int k=y*BLKS_PER_ROW+s;
            uint8_t htb=(s>0)?hti[y*H_TRANS_PER_ROW+(s-1)]:BG_TRANS,vtb=(y>0)?vti[(y-1)*BLKS_PER_ROW+s]:BG_TRANS;
            int ps=((pi[k]/9)-1)+(((pi[k]/3)%3)-1)+((pi[k]%3)-1);
            int hs=((hi[k]/9)-1)+(((hi[k]/3)%3)-1)+((hi[k]%3)-1);
            int vs=((vi[k]/9)-1)+(((vi[k]/3)%3)-1)+((vi[k]%3)-1);
            uint8_t pc=ps<0?0:ps==0?1:ps<3?2:3,hc=hs<0?0:hs==0?1:hs<3?2:3,vc=vs<0?0:vs==0?1:vs<3?2:3;
            oi[k]=pc|(hc<<2)|(vc<<4)|((htb!=BG_TRANS)?1<<6:0)|((vtb!=BG_TRANS)?1<<7:0);}}}
static void divf(const int8_t*hg,const int8_t*vg,int16_t*ns,int16_t*cy){
    int neg=0,ny=0,nc2=0;for(int y=0;y<SPATIAL_H;y++)for(int x=0;x<SPATIAL_W;x++){
        int dh=(int)hg[y*SPATIAL_W+x]-(x>0?(int)hg[y*SPATIAL_W+x-1]:0);
        int dv=(int)vg[y*SPATIAL_W+x]-(y>0?(int)vg[(y-1)*SPATIAL_W+x]:0);
        int d=dh+dv;if(d<0){neg+=d;ny+=y;nc2++;}}
    *ns=(int16_t)(neg<-32767?-32767:neg);*cy=nc2>0?(int16_t)(ny/nc2):-1;}
static void gdf(const int8_t*hg,const int8_t*vg,int grow,int gcol,int16_t*o){
    int nr=grow*gcol,reg[MAX_REGIONS];memset(reg,0,nr*sizeof(int));
    for(int y=0;y<SPATIAL_H;y++)for(int x=0;x<SPATIAL_W;x++){
        int dh=(int)hg[y*SPATIAL_W+x]-(x>0?(int)hg[y*SPATIAL_W+x-1]:0);
        int dv=(int)vg[y*SPATIAL_W+x]-(y>0?(int)vg[(y-1)*SPATIAL_W+x]:0);
        int d=dh+dv;if(d<0){int ry=y*grow/SPATIAL_H,rx=x*gcol/SPATIAL_W;
            if(ry>=grow)ry=grow-1;if(rx>=gcol)rx=gcol-1;reg[ry*gcol+rx]+=d;}}
    for(int i=0;i<nr;i++)o[i]=(int16_t)(reg[i]<-32767?-32767:reg[i]);}
static int32_t tdot(const int8_t*a,const int8_t*b,int len){int32_t tot=0;
    for(int ch=0;ch<len;ch+=64){__m256i ac=_mm256_setzero_si256();int e=ch+64;if(e>len)e=len;
        for(int i=ch;i<e;i+=32)ac=_mm256_add_epi8(ac,_mm256_sign_epi8(
            _mm256_load_si256((const __m256i*)(a+i)),_mm256_load_si256((const __m256i*)(b+i))));
        __m256i lo=_mm256_cvtepi8_epi16(_mm256_castsi256_si128(ac));
        __m256i hi=_mm256_cvtepi8_epi16(_mm256_extracti128_si256(ac,1));
        __m256i s32=_mm256_madd_epi16(_mm256_add_epi16(lo,hi),_mm256_set1_epi16(1));
        __m128i s=_mm_add_epi32(_mm256_castsi256_si128(s32),_mm256_extracti128_si256(s32,1));
        s=_mm_hadd_epi32(s,s);s=_mm_hadd_epi32(s,s);tot+=_mm_cvtsi128_si32(s);}return tot;}
typedef struct{uint32_t id;uint32_t votes;int64_t score;}cand_t;
static int cmpv(const void*a,const void*b){return(int)((const cand_t*)b)->votes-(int)((const cand_t*)a)->votes;}
static int cmps(const void*a,const void*b){int64_t d=((const cand_t*)b)->score-((const cand_t*)a)->score;return(d>0)-(d<0);}
static int *ghist=NULL;static size_t ghcap=0;
static int topk(const uint32_t*v,int n,cand_t*o,int k){uint32_t mx=0;for(int i=0;i<n;i++)if(v[i]>mx)mx=v[i];if(!mx)return 0;
    if((size_t)(mx+1)>ghcap){ghcap=(size_t)(mx+1)+4096;free(ghist);ghist=malloc(ghcap*sizeof(int));}
    memset(ghist,0,(mx+1)*sizeof(int));for(int i=0;i<n;i++)if(v[i])ghist[v[i]]++;
    int cu=0,th;for(th=(int)mx;th>=1;th--){cu+=ghist[th];if(cu>=k)break;}if(th<1)th=1;
    int nc=0;for(int i=0;i<n&&nc<k;i++)if(v[i]>=(uint32_t)th){o[nc]=(cand_t){0};o[nc].id=i;o[nc].votes=v[i];nc++;}
    qsort(o,(size_t)nc,sizeof(cand_t),cmpv);return nc;}

/* ================================================================
 *  Generic experiment runner: build pipeline + classify
 * ================================================================ */

typedef struct {
    int8_t *tern_tr,*tern_te,*hg_tr,*hg_te,*vg_tr,*vg_te;
    uint8_t *joint_tr,*joint_te;
    uint32_t *hot;
    uint16_t ig[N_BLOCKS];
    uint8_t bg;
    uint32_t idx_off[N_BLOCKS][BYTE_VALS];
    uint16_t idx_sz[N_BLOCKS][BYTE_VALS];
    uint32_t *idx_pool;
    /* Grayscale topo (shared) */
    int16_t *divneg_tr,*divneg_te,*divcy_tr,*divcy_te,*gdiv_tr,*gdiv_te;
} pipeline_t;

static uint8_t nbr[BYTE_VALS][8];

static void build_pipeline(pipeline_t *pl) {
    /* Block sigs + transitions + Encoding D */
    uint8_t *pxt=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD),*pxe=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    uint8_t *hst=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD),*hse=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    uint8_t *vst=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD),*vse=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    bsf(pl->tern_tr,pxt,TRAIN_N);bsf(pl->tern_te,pxe,TEST_N);
    bsf(pl->hg_tr,hst,TRAIN_N);bsf(pl->hg_te,hse,TEST_N);
    bsf(pl->vg_tr,vst,TRAIN_N);bsf(pl->vg_te,vse,TEST_N);
    uint8_t *htt=aligned_alloc(32,(size_t)TRAIN_N*TRANS_PAD),*hte2=aligned_alloc(32,(size_t)TEST_N*TRANS_PAD);
    uint8_t *vtt=aligned_alloc(32,(size_t)TRAIN_N*TRANS_PAD),*vte2=aligned_alloc(32,(size_t)TEST_N*TRANS_PAD);
    trf(pxt,htt,vtt,TRAIN_N);trf(pxe,hte2,vte2,TEST_N);
    pl->joint_tr=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD);pl->joint_te=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    jsf(pl->joint_tr,TRAIN_N,pxt,hst,vst,htt,vtt);jsf(pl->joint_te,TEST_N,pxe,hse,vse,hte2,vte2);
    free(pxt);free(pxe);free(hst);free(hse);free(vst);free(vse);free(htt);free(hte2);free(vtt);free(vte2);

    /* Background */
    long vc[BYTE_VALS]={0};for(int i=0;i<TRAIN_N;i++){const uint8_t*sig=pl->joint_tr+(size_t)i*SIG_PAD;
        for(int k=0;k<N_BLOCKS;k++)vc[sig[k]]++;}
    pl->bg=0;long mc=0;for(int v=0;v<BYTE_VALS;v++)if(vc[v]>mc){mc=vc[v];pl->bg=(uint8_t)v;}

    /* Hot map + IG */
    pl->hot=aligned_alloc(32,(size_t)N_BLOCKS*BYTE_VALS*CLS_PAD*4);
    memset(pl->hot,0,(size_t)N_BLOCKS*BYTE_VALS*CLS_PAD*4);
    for(int i=0;i<TRAIN_N;i++){int l=train_labels[i];const uint8_t*sig=pl->joint_tr+(size_t)i*SIG_PAD;
        for(int k=0;k<N_BLOCKS;k++)pl->hot[(size_t)k*BYTE_VALS*CLS_PAD+(size_t)sig[k]*CLS_PAD+l]++;}
    {int cc[N_CLASSES]={0};for(int i=0;i<TRAIN_N;i++)cc[train_labels[i]]++;
     double hc=0;for(int c=0;c<N_CLASSES;c++){double p=(double)cc[c]/TRAIN_N;if(p>0)hc-=p*log2(p);}
     double raw2[N_BLOCKS],mx=0;
     for(int k=0;k<N_BLOCKS;k++){double hcond=0;
         for(int v=0;v<BYTE_VALS;v++){if((uint8_t)v==pl->bg)continue;
             const uint32_t*h=pl->hot+(size_t)k*BYTE_VALS*CLS_PAD+(size_t)v*CLS_PAD;
             int vt=0;for(int c=0;c<N_CLASSES;c++)vt+=(int)h[c];if(!vt)continue;
             double pv=(double)vt/TRAIN_N,hv=0;for(int c=0;c<N_CLASSES;c++){double pc2=(double)h[c]/vt;if(pc2>0)hv-=pc2*log2(pc2);}
             hcond+=pv*hv;}raw2[k]=hc-hcond;if(raw2[k]>mx)mx=raw2[k];}
     for(int k=0;k<N_BLOCKS;k++){pl->ig[k]=mx>0?(uint16_t)(raw2[k]/mx*IG_SCALE+0.5):1;if(!pl->ig[k])pl->ig[k]=1;}}

    /* Index */
    memset(pl->idx_sz,0,sizeof(pl->idx_sz));
    for(int i=0;i<TRAIN_N;i++){const uint8_t*sig=pl->joint_tr+(size_t)i*SIG_PAD;
        for(int k=0;k<N_BLOCKS;k++)if(sig[k]!=pl->bg)pl->idx_sz[k][sig[k]]++;}
    uint32_t tot=0;for(int k=0;k<N_BLOCKS;k++)for(int v=0;v<BYTE_VALS;v++){pl->idx_off[k][v]=tot;tot+=pl->idx_sz[k][v];}
    pl->idx_pool=malloc((size_t)tot*4);
    uint32_t(*wp)[BYTE_VALS]=malloc((size_t)N_BLOCKS*BYTE_VALS*4);memcpy(wp,pl->idx_off,(size_t)N_BLOCKS*BYTE_VALS*4);
    for(int i=0;i<TRAIN_N;i++){const uint8_t*sig=pl->joint_tr+(size_t)i*SIG_PAD;
        for(int k=0;k<N_BLOCKS;k++)if(sig[k]!=pl->bg)pl->idx_pool[wp[k][sig[k]]++]=(uint32_t)i;}free(wp);
    printf("    BG=%d (%.1f%%), index=%u entries\n",pl->bg,100.0*mc/((long)TRAIN_N*N_BLOCKS),tot);
}

static void run_experiment(const char *name, pipeline_t *pl, int use_topo) {
    printf("\n=== %s ===\n", name);
    uint32_t *vbuf = calloc(TRAIN_N, 4);
    int nreg = 9;

    int bay_correct = 0, cas_correct = 0, stack_correct = 0;

    for (int i = 0; i < TEST_N; i++) {
        /* Bayesian */
        {const uint8_t *sig = pl->joint_te + (size_t)i * SIG_PAD;
         double bp[N_CLASSES]; for (int c = 0; c < N_CLASSES; c++) bp[c] = 0.0;
         for (int k = 0; k < N_BLOCKS; k++) { uint8_t bv = sig[k]; if (bv == pl->bg) continue;
             const uint32_t *h = pl->hot + (size_t)k * BYTE_VALS * CLS_PAD + (size_t)bv * CLS_PAD;
             for (int c = 0; c < N_CLASSES; c++) bp[c] += log(h[c] + 0.5); }
         int pred = 0; for (int c = 1; c < N_CLASSES; c++) if (bp[c] > bp[pred]) pred = c;
         if (pred == test_labels[i]) bay_correct++;}

        /* Vote → cascade */
        memset(vbuf, 0, TRAIN_N * 4);
        {const uint8_t *sig = pl->joint_te + (size_t)i * SIG_PAD;
         for (int k = 0; k < N_BLOCKS; k++) { uint8_t bv = sig[k]; if (bv == pl->bg) continue;
             uint16_t w = pl->ig[k], wh = w > 1 ? w / 2 : 1;
             { uint32_t off = pl->idx_off[k][bv]; uint16_t sz = pl->idx_sz[k][bv];
               for (uint16_t j = 0; j < sz; j++) vbuf[pl->idx_pool[off + j]] += w; }
             for (int nb = 0; nb < 8; nb++) { uint8_t nv = nbr[bv][nb]; if (nv == pl->bg) continue;
                 uint32_t noff = pl->idx_off[k][nv]; uint16_t nsz = pl->idx_sz[k][nv];
                 for (uint16_t j = 0; j < nsz; j++) vbuf[pl->idx_pool[noff + j]] += wh; }}}

        cand_t cands[TOP_K]; int nc = topk(vbuf, TRAIN_N, cands, TOP_K);

        /* Gradient dot (k=1) */
        for (int j = 0; j < nc; j++) { uint32_t id = cands[j].id;
            cands[j].score = tdot(pl->hg_te + (size_t)i * PADDED, pl->hg_tr + (size_t)id * PADDED, PADDED)
                           + tdot(pl->vg_te + (size_t)i * PADDED, pl->vg_tr + (size_t)id * PADDED, PADDED); }
        qsort(cands, (size_t)nc, sizeof(cand_t), cmps);
        if (nc > 0 && train_labels[cands[0].id] == test_labels[i]) cas_correct++;

        /* Full stack: gradient dot + topo + Bayesian prior */
        if (use_topo && pl->divneg_tr) {
            const uint8_t *sig = pl->joint_te + (size_t)i * SIG_PAD;
            double bp[N_CLASSES]; for (int c = 0; c < N_CLASSES; c++) bp[c] = 0.0;
            for (int k = 0; k < N_BLOCKS; k++) { uint8_t bv = sig[k]; if (bv == pl->bg) continue;
                const uint32_t *h = pl->hot + (size_t)k * BYTE_VALS * CLS_PAD + (size_t)bv * CLS_PAD;
                for (int c = 0; c < N_CLASSES; c++) bp[c] += log(h[c] + 0.5); }
            double mx2 = -1e30; for (int c = 0; c < N_CLASSES; c++) if (bp[c] > mx2) mx2 = bp[c];
            for (int c = 0; c < N_CLASSES; c++) bp[c] -= mx2;

            int16_t q_dn = pl->divneg_te[i], q_dc = pl->divcy_te[i];
            const int16_t *q_gd = pl->gdiv_te + (size_t)i * MAX_REGIONS;
            for (int j = 0; j < nc; j++) { uint32_t id = cands[j].id;
                int32_t gd = tdot(pl->hg_te + (size_t)i * PADDED, pl->hg_tr + (size_t)id * PADDED, PADDED)
                           + tdot(pl->vg_te + (size_t)i * PADDED, pl->vg_tr + (size_t)id * PADDED, PADDED);
                int16_t cdn = pl->divneg_tr[id], cdc = pl->divcy_tr[id];
                int32_t ds = -(int32_t)abs(q_dn - cdn);
                if (q_dc >= 0 && cdc >= 0) ds -= (int32_t)abs(q_dc - cdc) * 2;
                else if ((q_dc < 0) != (cdc < 0)) ds -= 10;
                const int16_t *cg = pl->gdiv_tr + (size_t)id * MAX_REGIONS;
                int32_t gl = 0; for (int r = 0; r < nreg; r++) gl += abs(q_gd[r] - cg[r]);
                uint8_t lbl = train_labels[id];
                cands[j].score = (int64_t)512 * gd + (int64_t)200 * ds + (int64_t)200 * (-gl)
                               + (int64_t)50 * (int64_t)(bp[lbl] * 1000); }
            qsort(cands, (size_t)nc, sizeof(cand_t), cmps);
            if (nc > 0 && train_labels[cands[0].id] == test_labels[i]) stack_correct++;
        }

        if ((i + 1) % 2000 == 0) fprintf(stderr, "  %s: %d/%d\r", name, i + 1, TEST_N);
    }
    fprintf(stderr, "\n");
    free(vbuf);

    printf("  Bayesian: %.2f%%\n", 100.0 * bay_correct / TEST_N);
    printf("  Cascade (grad dot, k=1): %.2f%%\n", 100.0 * cas_correct / TEST_N);
    if (use_topo) printf("  Full stack (dot+topo+bay): %.2f%%\n", 100.0 * stack_correct / TEST_N);

    /* Per-class Bayesian */
    printf("  Per-class Bayesian:\n");
    int pc[N_CLASSES] = {0}, pt[N_CLASSES] = {0};
    for (int i = 0; i < TEST_N; i++) {
        pt[test_labels[i]]++;
        const uint8_t *sig = pl->joint_te + (size_t)i * SIG_PAD;
        double bp[N_CLASSES]; for (int c = 0; c < N_CLASSES; c++) bp[c] = 0.0;
        for (int k = 0; k < N_BLOCKS; k++) { uint8_t bv = sig[k]; if (bv == pl->bg) continue;
            const uint32_t *h = pl->hot + (size_t)k * BYTE_VALS * CLS_PAD + (size_t)bv * CLS_PAD;
            for (int c = 0; c < N_CLASSES; c++) bp[c] += log(h[c] + 0.5); }
        int pred = 0; for (int c = 1; c < N_CLASSES; c++) if (bp[c] > bp[pred]) pred = c;
        if (pred == test_labels[i]) pc[test_labels[i]]++;
    }
    for (int c = 0; c < N_CLASSES; c++)
        printf("    %d %-10s %.1f%%\n", c, cn[c], 100.0 * pc[c] / pt[c]);
}

static void load_data(void) {
    raw_train=malloc((size_t)TRAIN_N*PIXELS);raw_test=malloc((size_t)TEST_N*PIXELS);
    raw_gray_tr=malloc((size_t)TRAIN_N*GRAY_PIXELS);raw_gray_te=malloc((size_t)TEST_N*GRAY_PIXELS);
    train_labels=malloc(TRAIN_N);test_labels=malloc(TEST_N);
    char p[512];uint8_t rec[3073];
    for(int b2=1;b2<=5;b2++){snprintf(p,sizeof(p),"%sdata_batch_%d.bin",data_dir,b2);
        FILE*f=fopen(p,"rb");for(int i=0;i<10000;i++){if(fread(rec,1,3073,f)!=3073){fclose(f);exit(1);}
            int idx=(b2-1)*10000+i;train_labels[idx]=rec[0];
            uint8_t*d=raw_train+(size_t)idx*PIXELS;const uint8_t*r=rec+1,*g=rec+1+1024,*b3=rec+1+2048;
            for(int y=0;y<32;y++)for(int x=0;x<32;x++){int si=y*32+x,di=y*96+x*3;d[di]=r[si];d[di+1]=g[si];d[di+2]=b3[si];}
            uint8_t*gd=raw_gray_tr+(size_t)idx*GRAY_PIXELS;
            for(int p2=0;p2<1024;p2++)gd[p2]=(uint8_t)((77*(int)r[p2]+150*(int)g[p2]+29*(int)b3[p2])>>8);}fclose(f);}
    snprintf(p,sizeof(p),"%stest_batch.bin",data_dir);
    FILE*f=fopen(p,"rb");for(int i=0;i<10000;i++){if(fread(rec,1,3073,f)!=3073){fclose(f);exit(1);}
        test_labels[i]=rec[0];uint8_t*d=raw_test+(size_t)i*PIXELS;const uint8_t*r=rec+1,*g=rec+1+1024,*b3=rec+1+2048;
        for(int y=0;y<32;y++)for(int x=0;x<32;x++){int si=y*32+x,di=y*96+x*3;d[di]=r[si];d[di+1]=g[si];d[di+2]=b3[si];}
        uint8_t*gd=raw_gray_te+(size_t)i*GRAY_PIXELS;
        for(int p2=0;p2<1024;p2++)gd[p2]=(uint8_t)((77*(int)r[p2]+150*(int)g[p2]+29*(int)b3[p2])>>8);}fclose(f);}

int main(int argc, char **argv) {
    double t0 = now_sec();
    if(argc>1){data_dir=argv[1];size_t l=strlen(data_dir);
        if(l&&data_dir[l-1]!='/'){char*b2=malloc(l+2);memcpy(b2,data_dir,l);b2[l]='/';b2[l+1]=0;data_dir=b2;}}
    printf("=== SSTT CIFAR-10: Adaptive Quantization ===\n\n");
    load_data();
    for(int v=0;v<BYTE_VALS;v++)for(int b3=0;b3<8;b3++)nbr[v][b3]=(uint8_t)(v^(1<<b3));

    /* Shared: grayscale topo features (adaptive on grayscale too) */
    int8_t *gray_tern_tr=aligned_alloc(32,(size_t)TRAIN_N*GRAY_PADDED);
    int8_t *gray_tern_te=aligned_alloc(32,(size_t)TEST_N*GRAY_PADDED);
    int8_t *gray_hg_tr=aligned_alloc(32,(size_t)TRAIN_N*GRAY_PADDED);
    int8_t *gray_hg_te=aligned_alloc(32,(size_t)TEST_N*GRAY_PADDED);
    int8_t *gray_vg_tr=aligned_alloc(32,(size_t)TRAIN_N*GRAY_PADDED);
    int8_t *gray_vg_te=aligned_alloc(32,(size_t)TEST_N*GRAY_PADDED);
    adaptive_quantize(raw_gray_tr,gray_tern_tr,TRAIN_N,GRAY_PIXELS,GRAY_PADDED);
    adaptive_quantize(raw_gray_te,gray_tern_te,TEST_N,GRAY_PIXELS,GRAY_PADDED);
    grads(gray_tern_tr,gray_hg_tr,gray_vg_tr,TRAIN_N,SPATIAL_W,SPATIAL_H,GRAY_PADDED);
    grads(gray_tern_te,gray_hg_te,gray_vg_te,TEST_N,SPATIAL_W,SPATIAL_H,GRAY_PADDED);
    int16_t *divneg_tr=malloc((size_t)TRAIN_N*2),*divneg_te=malloc((size_t)TEST_N*2);
    int16_t *divcy_tr=malloc((size_t)TRAIN_N*2),*divcy_te=malloc((size_t)TEST_N*2);
    for(int i=0;i<TRAIN_N;i++)divf(gray_hg_tr+(size_t)i*GRAY_PADDED,gray_vg_tr+(size_t)i*GRAY_PADDED,&divneg_tr[i],&divcy_tr[i]);
    for(int i=0;i<TEST_N;i++)divf(gray_hg_te+(size_t)i*GRAY_PADDED,gray_vg_te+(size_t)i*GRAY_PADDED,&divneg_te[i],&divcy_te[i]);
    int16_t *gdiv_tr=malloc((size_t)TRAIN_N*MAX_REGIONS*2),*gdiv_te=malloc((size_t)TEST_N*MAX_REGIONS*2);
    for(int i=0;i<TRAIN_N;i++)gdf(gray_hg_tr+(size_t)i*GRAY_PADDED,gray_vg_tr+(size_t)i*GRAY_PADDED,3,3,gdiv_tr+(size_t)i*MAX_REGIONS);
    for(int i=0;i<TEST_N;i++)gdf(gray_hg_te+(size_t)i*GRAY_PADDED,gray_vg_te+(size_t)i*GRAY_PADDED,3,3,gdiv_te+(size_t)i*MAX_REGIONS);
    free(gray_tern_tr);free(gray_tern_te);free(gray_hg_tr);free(gray_hg_te);free(gray_vg_tr);free(gray_vg_te);

    /* ================================================================
     * EXPERIMENT 1: Fixed thresholds (baseline)
     * ================================================================ */
    printf("Building Experiment 1: Fixed 85/170...\n");
    pipeline_t pl_fixed = {0};
    pl_fixed.tern_tr=aligned_alloc(32,(size_t)TRAIN_N*PADDED);pl_fixed.tern_te=aligned_alloc(32,(size_t)TEST_N*PADDED);
    pl_fixed.hg_tr=aligned_alloc(32,(size_t)TRAIN_N*PADDED);pl_fixed.hg_te=aligned_alloc(32,(size_t)TEST_N*PADDED);
    pl_fixed.vg_tr=aligned_alloc(32,(size_t)TRAIN_N*PADDED);pl_fixed.vg_te=aligned_alloc(32,(size_t)TEST_N*PADDED);
    fixed_quantize(raw_train,pl_fixed.tern_tr,TRAIN_N,PIXELS,PADDED);
    fixed_quantize(raw_test,pl_fixed.tern_te,TEST_N,PIXELS,PADDED);
    grads(pl_fixed.tern_tr,pl_fixed.hg_tr,pl_fixed.vg_tr,TRAIN_N,IMG_W,IMG_H,PADDED);
    grads(pl_fixed.tern_te,pl_fixed.hg_te,pl_fixed.vg_te,TEST_N,IMG_W,IMG_H,PADDED);
    pl_fixed.divneg_tr=divneg_tr;pl_fixed.divneg_te=divneg_te;
    pl_fixed.divcy_tr=divcy_tr;pl_fixed.divcy_te=divcy_te;
    pl_fixed.gdiv_tr=gdiv_tr;pl_fixed.gdiv_te=gdiv_te;
    build_pipeline(&pl_fixed);
    run_experiment("Exp 1: Fixed 85/170 (baseline)", &pl_fixed, 1);

    /* Free fixed pipeline data to save memory */
    free(pl_fixed.tern_tr);free(pl_fixed.tern_te);
    free(pl_fixed.hg_tr);free(pl_fixed.hg_te);free(pl_fixed.vg_tr);free(pl_fixed.vg_te);
    free(pl_fixed.joint_tr);free(pl_fixed.joint_te);free(pl_fixed.hot);free(pl_fixed.idx_pool);

    /* ================================================================
     * EXPERIMENT 2: Adaptive per-image quantization
     * ================================================================ */
    printf("\nBuilding Experiment 2: Adaptive per-image P33/P67...\n");
    pipeline_t pl_adapt = {0};
    pl_adapt.tern_tr=aligned_alloc(32,(size_t)TRAIN_N*PADDED);pl_adapt.tern_te=aligned_alloc(32,(size_t)TEST_N*PADDED);
    pl_adapt.hg_tr=aligned_alloc(32,(size_t)TRAIN_N*PADDED);pl_adapt.hg_te=aligned_alloc(32,(size_t)TEST_N*PADDED);
    pl_adapt.vg_tr=aligned_alloc(32,(size_t)TRAIN_N*PADDED);pl_adapt.vg_te=aligned_alloc(32,(size_t)TEST_N*PADDED);
    adaptive_quantize(raw_train,pl_adapt.tern_tr,TRAIN_N,PIXELS,PADDED);
    adaptive_quantize(raw_test,pl_adapt.tern_te,TEST_N,PIXELS,PADDED);
    grads(pl_adapt.tern_tr,pl_adapt.hg_tr,pl_adapt.vg_tr,TRAIN_N,IMG_W,IMG_H,PADDED);
    grads(pl_adapt.tern_te,pl_adapt.hg_te,pl_adapt.vg_te,TEST_N,IMG_W,IMG_H,PADDED);
    pl_adapt.divneg_tr=divneg_tr;pl_adapt.divneg_te=divneg_te;
    pl_adapt.divcy_tr=divcy_tr;pl_adapt.divcy_te=divcy_te;
    pl_adapt.gdiv_tr=gdiv_tr;pl_adapt.gdiv_te=gdiv_te;
    build_pipeline(&pl_adapt);
    run_experiment("Exp 2: Adaptive per-image P33/P67", &pl_adapt, 1);
    free(pl_adapt.tern_tr);free(pl_adapt.tern_te);
    free(pl_adapt.hg_tr);free(pl_adapt.hg_te);free(pl_adapt.vg_tr);free(pl_adapt.vg_te);
    free(pl_adapt.joint_tr);free(pl_adapt.joint_te);free(pl_adapt.hot);free(pl_adapt.idx_pool);

    /* ================================================================
     * EXPERIMENT 3: Per-channel adaptive quantization
     * ================================================================ */
    printf("\nBuilding Experiment 3: Per-channel adaptive P33/P67...\n");
    pipeline_t pl_perch = {0};
    pl_perch.tern_tr=aligned_alloc(32,(size_t)TRAIN_N*PADDED);pl_perch.tern_te=aligned_alloc(32,(size_t)TEST_N*PADDED);
    pl_perch.hg_tr=aligned_alloc(32,(size_t)TRAIN_N*PADDED);pl_perch.hg_te=aligned_alloc(32,(size_t)TEST_N*PADDED);
    pl_perch.vg_tr=aligned_alloc(32,(size_t)TRAIN_N*PADDED);pl_perch.vg_te=aligned_alloc(32,(size_t)TEST_N*PADDED);
    adaptive_quantize_perchannel(raw_train,pl_perch.tern_tr,TRAIN_N);
    adaptive_quantize_perchannel(raw_test,pl_perch.tern_te,TEST_N);
    grads(pl_perch.tern_tr,pl_perch.hg_tr,pl_perch.vg_tr,TRAIN_N,IMG_W,IMG_H,PADDED);
    grads(pl_perch.tern_te,pl_perch.hg_te,pl_perch.vg_te,TEST_N,IMG_W,IMG_H,PADDED);
    pl_perch.divneg_tr=divneg_tr;pl_perch.divneg_te=divneg_te;
    pl_perch.divcy_tr=divcy_tr;pl_perch.divcy_te=divcy_te;
    pl_perch.gdiv_tr=gdiv_tr;pl_perch.gdiv_te=gdiv_te;
    build_pipeline(&pl_perch);
    run_experiment("Exp 3: Per-channel adaptive P33/P67", &pl_perch, 1);

    printf("\n=== SUMMARY ===\n");
     printf("  If adaptive > fixed: brightness was the confound.\n");

    printf("  If adaptive <= fixed: representation ceiling is scale/shape, not lighting.\n");

    printf("\nTotal: %.1f sec\n", now_sec() - t0);
    return 0;
}
