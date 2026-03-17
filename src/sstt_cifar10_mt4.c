/*
 * sstt_cifar10_mt4.c — Multi-Trit-4 Dot Product on CIFAR-10
 *
 * The Mode C diagnostic showed the ternary dot product can't discriminate:
 * correct/wrong neighbor scores differ by only 4.6%. More resolution
 * might break the tie.
 *
 * Multi-Trit-4: decompose each pixel into 4 balanced ternary planes.
 *   bin = pixel * 81 / 256  (maps [0,255] to [0,80])
 *   t3 = (bin/27) - 1       (coarsest, weight 27)
 *   t2 = ((bin%27)/9) - 1   (weight 9)
 *   t1 = ((bin%9)/3) - 1    (weight 3)
 *   t0 = (bin%3) - 1        (finest, weight 1)
 *
 * Combined dot = 27*dot(t3) + 9*dot(t2) + 3*dot(t1) + dot(t0)
 *
 * This gives 81-level quantization resolution while preserving the
 * _mm256_sign_epi8 ternary ALU on each plane independently.
 *
 * Voting uses the coarsest plane (same as current ternary).
 * Only the dot product changes.
 *
 * Build: gcc -O3 -mavx2 -march=native -o sstt_cifar10_mt4 src/sstt_cifar10_mt4.c -lm
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

static const char *data_dir="data-cifar10/";
static const char *cn[]={"airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"};
static double now_sec(void){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return ts.tv_sec+ts.tv_nsec*1e-9;}

static uint8_t *raw_train,*raw_test,*train_labels,*test_labels;

/* 4 ternary planes per image */
static int8_t *t3_tr,*t3_te; /* coarsest (weight 27) */
static int8_t *t2_tr,*t2_te; /* weight 9 */
static int8_t *t1_tr,*t1_te; /* weight 3 */
static int8_t *t0_tr,*t0_te; /* finest (weight 1) */

/* Gradients on each plane */
static int8_t *hg3_tr,*hg3_te,*vg3_tr,*vg3_te;
static int8_t *hg2_tr,*hg2_te,*vg2_tr,*vg2_te;
static int8_t *hg1_tr,*hg1_te,*vg1_tr,*vg1_te;
static int8_t *hg0_tr,*hg0_te,*vg0_tr,*vg0_te;

/* Voting infrastructure (coarsest plane) */
static uint8_t *joint_tr,*joint_te;
static uint32_t *joint_hot;
static uint16_t ig_w[N_BLOCKS];
static uint8_t nbr[BYTE_VALS][8];
static uint32_t idx_off[N_BLOCKS][BYTE_VALS];
static uint16_t idx_sz[N_BLOCKS][BYTE_VALS];
static uint32_t *idx_pool;
static uint8_t bg_val;

/* ================================================================
 *  Multi-Trit-4 quantization
 * ================================================================ */

static void mt4_quantize(const uint8_t *src, int n,
                          int8_t *p3, int8_t *p2, int8_t *p1, int8_t *p0) {
    for (int img = 0; img < n; img++) {
        const uint8_t *s = src + (size_t)img * PIXELS;
        int8_t *d3 = p3 + (size_t)img * PADDED;
        int8_t *d2 = p2 + (size_t)img * PADDED;
        int8_t *d1 = p1 + (size_t)img * PADDED;
        int8_t *d0 = p0 + (size_t)img * PADDED;
        for (int i = 0; i < PIXELS; i++) {
            int bin = (int)s[i] * 81 / 256;  /* 0..80 */
            d3[i] = (int8_t)((bin / 27) - 1);
            int r = bin % 27;
            d2[i] = (int8_t)((r / 9) - 1);
            r = r % 9;
            d1[i] = (int8_t)((r / 3) - 1);
            d0[i] = (int8_t)((r % 3) - 1);
        }
    }
}

static inline int8_t ct(int v){return v>0?1:v<0?-1:0;}

static void grads(const int8_t *t, int8_t *h, int8_t *v, int n) {
    for(int img=0;img<n;img++){
        const int8_t *ti=t+(size_t)img*PADDED;
        int8_t *hi=h+(size_t)img*PADDED, *vi=v+(size_t)img*PADDED;
        for(int y=0;y<IMG_H;y++){
            for(int x=0;x<IMG_W-1;x++) hi[y*IMG_W+x]=ct(ti[y*IMG_W+x+1]-ti[y*IMG_W+x]);
            hi[y*IMG_W+IMG_W-1]=0;
        }
        for(int y=0;y<IMG_H-1;y++)
            for(int x=0;x<IMG_W;x++) vi[y*IMG_W+x]=ct(ti[(y+1)*IMG_W+x]-ti[y*IMG_W+x]);
        memset(vi+(IMG_H-1)*IMG_W,0,IMG_W);
    }
}

/* Standard ternary quantize for coarsest plane block sigs */
static void quant_tern(const uint8_t *src, int8_t *dst, int n) {
    const __m256i bi=_mm256_set1_epi8((char)0x80),th=_mm256_set1_epi8((char)(170^0x80)),
                 tl=_mm256_set1_epi8((char)(85^0x80)),on=_mm256_set1_epi8(1);
    for(int i2=0;i2<n;i2++){const uint8_t*s=src+(size_t)i2*PIXELS;int8_t*d=dst+(size_t)i2*PADDED;
        for(int i=0;i<PIXELS;i+=32){__m256i p=_mm256_loadu_si256((const __m256i*)(s+i));
            __m256i sp=_mm256_xor_si256(p,bi);
            _mm256_storeu_si256((__m256i*)(d+i),_mm256_sub_epi8(
                _mm256_and_si256(_mm256_cmpgt_epi8(sp,th),on),
                _mm256_and_si256(_mm256_cmpgt_epi8(tl,sp),on)));}}
}

/* ================================================================
 *  Block sigs, transitions, Encoding D (on coarsest ternary for voting)
 * ================================================================ */

static inline uint8_t be(int8_t a,int8_t b,int8_t c){return(uint8_t)((a+1)*9+(b+1)*3+(c+1));}
static void bsigs(const int8_t *d, uint8_t *s, int n) {
    for(int i=0;i<n;i++){const int8_t *im=d+(size_t)i*PADDED;uint8_t *si=s+(size_t)i*SIG_PAD;
        for(int y=0;y<IMG_H;y++)for(int s2=0;s2<BLKS_PER_ROW;s2++){
            int b2=y*IMG_W+s2*3;si[y*BLKS_PER_ROW+s2]=be(im[b2],im[b2+1],im[b2+2]);}}}
static inline uint8_t te(uint8_t a,uint8_t b){int8_t a0=(a/9)-1,a1=((a/3)%3)-1,a2=(a%3)-1,
    b0=(b/9)-1,b1=((b/3)%3)-1,b2=(b%3)-1;return be(ct(b0-a0),ct(b1-a1),ct(b2-a2));}
static void trans(const uint8_t *bs,uint8_t *ht,uint8_t *vt,int n){
    for(int i=0;i<n;i++){const uint8_t *s=bs+(size_t)i*SIG_PAD;
        uint8_t *h=ht+(size_t)i*TRANS_PAD,*v=vt+(size_t)i*TRANS_PAD;
        for(int y=0;y<IMG_H;y++)for(int ss=0;ss<H_TRANS_PER_ROW;ss++)
            h[y*H_TRANS_PER_ROW+ss]=te(s[y*BLKS_PER_ROW+ss],s[y*BLKS_PER_ROW+ss+1]);
        memset(h+N_HTRANS,0xFF,TRANS_PAD-N_HTRANS);
        for(int y=0;y<V_TRANS_PER_COL;y++)for(int ss=0;ss<BLKS_PER_ROW;ss++)
            v[y*BLKS_PER_ROW+ss]=te(s[y*BLKS_PER_ROW+ss],s[(y+1)*BLKS_PER_ROW+ss]);
        memset(v+N_VTRANS,0xFF,TRANS_PAD-N_VTRANS);}}
static void jsigs(uint8_t *o,int n,const uint8_t *px,const uint8_t *hg,const uint8_t *vg,
                  const uint8_t *ht,const uint8_t *vt){
    for(int i=0;i<n;i++){const uint8_t *pi=px+(size_t)i*SIG_PAD,*hi=hg+(size_t)i*SIG_PAD,
        *vi=vg+(size_t)i*SIG_PAD;const uint8_t *hti=ht+(size_t)i*TRANS_PAD,*vti=vt+(size_t)i*TRANS_PAD;
        uint8_t *oi=o+(size_t)i*SIG_PAD;
        for(int y=0;y<IMG_H;y++)for(int s=0;s<BLKS_PER_ROW;s++){int k=y*BLKS_PER_ROW+s;
            uint8_t htb=(s>0)?hti[y*H_TRANS_PER_ROW+(s-1)]:BG_TRANS,vtb=(y>0)?vti[(y-1)*BLKS_PER_ROW+s]:BG_TRANS;
            int ps=((pi[k]/9)-1)+(((pi[k]/3)%3)-1)+((pi[k]%3)-1);
            int hs=((hi[k]/9)-1)+(((hi[k]/3)%3)-1)+((hi[k]%3)-1);
            int vs=((vi[k]/9)-1)+(((vi[k]/3)%3)-1)+((vi[k]%3)-1);
            uint8_t pc=ps<0?0:ps==0?1:ps<3?2:3,hc=hs<0?0:hs==0?1:hs<3?2:3,vc=vs<0?0:vs==0?1:vs<3?2:3;
            oi[k]=pc|(hc<<2)|(vc<<4)|((htb!=BG_TRANS)?1<<6:0)|((vtb!=BG_TRANS)?1<<7:0);}}}

/* ================================================================
 *  Safe ternary dot for long vectors
 * ================================================================ */

static int32_t tdot(const int8_t *a, const int8_t *b) {
    int32_t tot=0;
    for(int ch=0;ch<PADDED;ch+=64){__m256i ac=_mm256_setzero_si256();
        int e=ch+64;if(e>PADDED)e=PADDED;
        for(int i=ch;i<e;i+=32)ac=_mm256_add_epi8(ac,_mm256_sign_epi8(
            _mm256_load_si256((const __m256i*)(a+i)),_mm256_load_si256((const __m256i*)(b+i))));
        __m256i lo=_mm256_cvtepi8_epi16(_mm256_castsi256_si128(ac));
        __m256i hi=_mm256_cvtepi8_epi16(_mm256_extracti128_si256(ac,1));
        __m256i s32=_mm256_madd_epi16(_mm256_add_epi16(lo,hi),_mm256_set1_epi16(1));
        __m128i s=_mm_add_epi32(_mm256_castsi256_si128(s32),_mm256_extracti128_si256(s32,1));
        s=_mm_hadd_epi32(s,s);s=_mm_hadd_epi32(s,s);tot+=_mm_cvtsi128_si32(s);}
    return tot;
}

/* Multi-Trit-4 dot: 27*dot3 + 9*dot2 + 3*dot1 + dot0 */
static int32_t mt4_dot(const int8_t *a3, const int8_t *a2, const int8_t *a1, const int8_t *a0,
                        const int8_t *b3, const int8_t *b2, const int8_t *b1, const int8_t *b0) {
    return 27*tdot(a3,b3) + 9*tdot(a2,b2) + 3*tdot(a1,b1) + tdot(a0,b0);
}

/* MT4 gradient dot: combine h-grad and v-grad across all 4 planes */
static int32_t mt4_grad_dot(int img_te, int img_tr) {
    return 27*(tdot(hg3_te+(size_t)img_te*PADDED, hg3_tr+(size_t)img_tr*PADDED)
              +tdot(vg3_te+(size_t)img_te*PADDED, vg3_tr+(size_t)img_tr*PADDED))
          + 9*(tdot(hg2_te+(size_t)img_te*PADDED, hg2_tr+(size_t)img_tr*PADDED)
              +tdot(vg2_te+(size_t)img_te*PADDED, vg2_tr+(size_t)img_tr*PADDED))
          + 3*(tdot(hg1_te+(size_t)img_te*PADDED, hg1_tr+(size_t)img_tr*PADDED)
              +tdot(vg1_te+(size_t)img_te*PADDED, vg1_tr+(size_t)img_tr*PADDED))
          +    (tdot(hg0_te+(size_t)img_te*PADDED, hg0_tr+(size_t)img_tr*PADDED)
              +tdot(vg0_te+(size_t)img_te*PADDED, vg0_tr+(size_t)img_tr*PADDED));
}

/* ================================================================ */

typedef struct{uint32_t id;uint32_t votes;int32_t dot;}cand_t;
static int cmpv(const void*a,const void*b){return(int)((const cand_t*)b)->votes-(int)((const cand_t*)a)->votes;}
static int cmpd(const void*a,const void*b){int32_t d=((const cand_t*)b)->dot-((const cand_t*)a)->dot;return(d>0)-(d<0);}
static int *ghist=NULL;static size_t ghcap=0;
static int topk(const uint32_t*v,int n,cand_t*o,int k){
    uint32_t mx=0;for(int i=0;i<n;i++)if(v[i]>mx)mx=v[i];if(!mx)return 0;
    if((size_t)(mx+1)>ghcap){ghcap=(size_t)(mx+1)+4096;free(ghist);ghist=malloc(ghcap*sizeof(int));}
    memset(ghist,0,(mx+1)*sizeof(int));for(int i=0;i<n;i++)if(v[i])ghist[v[i]]++;
    int cu=0,th;for(th=(int)mx;th>=1;th--){cu+=ghist[th];if(cu>=k)break;}if(th<1)th=1;
    int nc=0;for(int i=0;i<n&&nc<k;i++)if(v[i]>=(uint32_t)th){o[nc]=(cand_t){0};o[nc].id=i;o[nc].votes=v[i];nc++;}
    qsort(o,(size_t)nc,sizeof(cand_t),cmpv);return nc;}
static int knn(const cand_t*c,int nc,int k){int v[N_CLASSES]={0};if(k>nc)k=nc;
    for(int i=0;i<k;i++)v[train_labels[c[i].id]]++;int b=0;for(int c2=1;c2<N_CLASSES;c2++)if(v[c2]>v[b])b=c2;return b;}

static void load_data(void){
    raw_train=malloc((size_t)TRAIN_N*PIXELS);raw_test=malloc((size_t)TEST_N*PIXELS);
    train_labels=malloc(TRAIN_N);test_labels=malloc(TEST_N);
    char p[512];uint8_t rec[3073];
    for(int b2=1;b2<=5;b2++){snprintf(p,sizeof(p),"%sdata_batch_%d.bin",data_dir,b2);
        FILE*f=fopen(p,"rb");for(int i=0;i<10000;i++){
            if(fread(rec,1,3073,f)!=3073){fclose(f);exit(1);}
            int idx=(b2-1)*10000+i;train_labels[idx]=rec[0];
            uint8_t*d=raw_train+(size_t)idx*PIXELS;const uint8_t*r=rec+1,*g=rec+1+1024,*b3=rec+1+2048;
            for(int y=0;y<32;y++)for(int x=0;x<32;x++){int si=y*32+x,di=y*96+x*3;
                d[di]=r[si];d[di+1]=g[si];d[di+2]=b3[si];}}fclose(f);}
    snprintf(p,sizeof(p),"%stest_batch.bin",data_dir);
    FILE*f=fopen(p,"rb");for(int i=0;i<10000;i++){
        if(fread(rec,1,3073,f)!=3073){fclose(f);exit(1);}
        test_labels[i]=rec[0];uint8_t*d=raw_test+(size_t)i*PIXELS;
        const uint8_t*r=rec+1,*g=rec+1+1024,*b3=rec+1+2048;
        for(int y=0;y<32;y++)for(int x=0;x<32;x++){int si=y*32+x,di=y*96+x*3;
            d[di]=r[si];d[di+1]=g[si];d[di+2]=b3[si];}}fclose(f);
    printf("  Loaded\n");}

int main(int argc,char**argv){
    double t0=now_sec();
    if(argc>1){data_dir=argv[1];size_t l=strlen(data_dir);
        if(l&&data_dir[l-1]!='/'){char*b2=malloc(l+2);memcpy(b2,data_dir,l);b2[l]='/';b2[l+1]=0;data_dir=b2;}}
    printf("=== SSTT CIFAR-10: Multi-Trit-4 (81-level dot product) ===\n\n");

    load_data();

    /* MT4 quantization */
    printf("Computing MT4 planes...\n");
    t3_tr=aligned_alloc(32,(size_t)TRAIN_N*PADDED);t3_te=aligned_alloc(32,(size_t)TEST_N*PADDED);
    t2_tr=aligned_alloc(32,(size_t)TRAIN_N*PADDED);t2_te=aligned_alloc(32,(size_t)TEST_N*PADDED);
    t1_tr=aligned_alloc(32,(size_t)TRAIN_N*PADDED);t1_te=aligned_alloc(32,(size_t)TEST_N*PADDED);
    t0_tr=aligned_alloc(32,(size_t)TRAIN_N*PADDED);t0_te=aligned_alloc(32,(size_t)TEST_N*PADDED);
    mt4_quantize(raw_train,TRAIN_N,t3_tr,t2_tr,t1_tr,t0_tr);
    mt4_quantize(raw_test,TEST_N,t3_te,t2_te,t1_te,t0_te);

    /* Trit distributions per plane */
    for(int plane=0;plane<4;plane++){
        const int8_t *d=(plane==0)?t3_tr:(plane==1)?t2_tr:(plane==2)?t1_tr:t0_tr;
        long tn[3]={0};for(int i=0;i<TRAIN_N;i++){const int8_t*t=d+(size_t)i*PADDED;
            for(int p2=0;p2<PIXELS;p2++)tn[t[p2]+1]++;}
        long total=(long)TRAIN_N*PIXELS;
        printf("  Plane %d (w=%d): -1=%.1f%% 0=%.1f%% +1=%.1f%%\n",
               plane,plane==0?27:plane==1?9:plane==2?3:1,
               100.0*tn[0]/total,100.0*tn[1]/total,100.0*tn[2]/total);
    }

    /* Gradients on each plane */
    printf("Computing gradients on all 4 planes...\n");
    hg3_tr=aligned_alloc(32,(size_t)TRAIN_N*PADDED);hg3_te=aligned_alloc(32,(size_t)TEST_N*PADDED);
    vg3_tr=aligned_alloc(32,(size_t)TRAIN_N*PADDED);vg3_te=aligned_alloc(32,(size_t)TEST_N*PADDED);
    hg2_tr=aligned_alloc(32,(size_t)TRAIN_N*PADDED);hg2_te=aligned_alloc(32,(size_t)TEST_N*PADDED);
    vg2_tr=aligned_alloc(32,(size_t)TRAIN_N*PADDED);vg2_te=aligned_alloc(32,(size_t)TEST_N*PADDED);
    hg1_tr=aligned_alloc(32,(size_t)TRAIN_N*PADDED);hg1_te=aligned_alloc(32,(size_t)TEST_N*PADDED);
    vg1_tr=aligned_alloc(32,(size_t)TRAIN_N*PADDED);vg1_te=aligned_alloc(32,(size_t)TEST_N*PADDED);
    hg0_tr=aligned_alloc(32,(size_t)TRAIN_N*PADDED);hg0_te=aligned_alloc(32,(size_t)TEST_N*PADDED);
    vg0_tr=aligned_alloc(32,(size_t)TRAIN_N*PADDED);vg0_te=aligned_alloc(32,(size_t)TEST_N*PADDED);
    grads(t3_tr,hg3_tr,vg3_tr,TRAIN_N);grads(t3_te,hg3_te,vg3_te,TEST_N);
    grads(t2_tr,hg2_tr,vg2_tr,TRAIN_N);grads(t2_te,hg2_te,vg2_te,TEST_N);
    grads(t1_tr,hg1_tr,vg1_tr,TRAIN_N);grads(t1_te,hg1_te,vg1_te,TEST_N);
    grads(t0_tr,hg0_tr,vg0_tr,TRAIN_N);grads(t0_te,hg0_te,vg0_te,TEST_N);

    /* Voting: use coarsest ternary plane (same as standard) */
    printf("Building vote index (coarsest plane)...\n");
    int8_t *tern_tr=aligned_alloc(32,(size_t)TRAIN_N*PADDED),*tern_te=aligned_alloc(32,(size_t)TEST_N*PADDED);
    quant_tern(raw_train,tern_tr,TRAIN_N);quant_tern(raw_test,tern_te,TEST_N);
    int8_t *hgc_tr=aligned_alloc(32,(size_t)TRAIN_N*PADDED),*hgc_te=aligned_alloc(32,(size_t)TEST_N*PADDED);
    int8_t *vgc_tr=aligned_alloc(32,(size_t)TRAIN_N*PADDED),*vgc_te=aligned_alloc(32,(size_t)TEST_N*PADDED);
    grads(tern_tr,hgc_tr,vgc_tr,TRAIN_N);grads(tern_te,hgc_te,vgc_te,TEST_N);

    uint8_t *pxt=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD),*pxe=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    uint8_t *hst=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD),*hse=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    uint8_t *vst=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD),*vse=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    bsigs(tern_tr,pxt,TRAIN_N);bsigs(tern_te,pxe,TEST_N);
    bsigs(hgc_tr,hst,TRAIN_N);bsigs(hgc_te,hse,TEST_N);
    bsigs(vgc_tr,vst,TRAIN_N);bsigs(vgc_te,vse,TEST_N);
    uint8_t *htt=aligned_alloc(32,(size_t)TRAIN_N*TRANS_PAD),*hte2=aligned_alloc(32,(size_t)TEST_N*TRANS_PAD);
    uint8_t *vtt=aligned_alloc(32,(size_t)TRAIN_N*TRANS_PAD),*vte2=aligned_alloc(32,(size_t)TEST_N*TRANS_PAD);
    trans(pxt,htt,vtt,TRAIN_N);trans(pxe,hte2,vte2,TEST_N);
    joint_tr=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD);joint_te=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    jsigs(joint_tr,TRAIN_N,pxt,hst,vst,htt,vtt);jsigs(joint_te,TEST_N,pxe,hse,vse,hte2,vte2);

    /* Background + hot map + IG + index */
    {long vc[BYTE_VALS]={0};for(int i=0;i<TRAIN_N;i++){const uint8_t*sig=joint_tr+(size_t)i*SIG_PAD;
        for(int k=0;k<N_BLOCKS;k++)vc[sig[k]]++;}bg_val=0;long mc=0;
        for(int v=0;v<BYTE_VALS;v++)if(vc[v]>mc){mc=vc[v];bg_val=(uint8_t)v;}
        printf("  BG=%d (%.1f%%)\n",bg_val,100.0*mc/((long)TRAIN_N*N_BLOCKS));}
    joint_hot=aligned_alloc(32,(size_t)N_BLOCKS*BYTE_VALS*CLS_PAD*4);
    memset(joint_hot,0,(size_t)N_BLOCKS*BYTE_VALS*CLS_PAD*4);
    for(int i=0;i<TRAIN_N;i++){int l=train_labels[i];const uint8_t*sig=joint_tr+(size_t)i*SIG_PAD;
        for(int k=0;k<N_BLOCKS;k++)joint_hot[(size_t)k*BYTE_VALS*CLS_PAD+(size_t)sig[k]*CLS_PAD+l]++;}
    {int cc[N_CLASSES]={0};for(int i=0;i<TRAIN_N;i++)cc[train_labels[i]]++;
     double hc=0;for(int c=0;c<N_CLASSES;c++){double p=(double)cc[c]/TRAIN_N;if(p>0)hc-=p*log2(p);}
     double raw[N_BLOCKS],mx=0;
     for(int k=0;k<N_BLOCKS;k++){double hcond=0;
         for(int v=0;v<BYTE_VALS;v++){if((uint8_t)v==bg_val)continue;
             const uint32_t*h=joint_hot+(size_t)k*BYTE_VALS*CLS_PAD+(size_t)v*CLS_PAD;
             int vt=0;for(int c=0;c<N_CLASSES;c++)vt+=(int)h[c];if(!vt)continue;
             double pv=(double)vt/TRAIN_N,hv=0;for(int c=0;c<N_CLASSES;c++){double pc2=(double)h[c]/vt;if(pc2>0)hv-=pc2*log2(pc2);}
             hcond+=pv*hv;}raw[k]=hc-hcond;if(raw[k]>mx)mx=raw[k];}
     for(int k=0;k<N_BLOCKS;k++){ig_w[k]=mx>0?(uint16_t)(raw[k]/mx*IG_SCALE+0.5):1;if(!ig_w[k])ig_w[k]=1;}}
    for(int v=0;v<BYTE_VALS;v++)for(int b2=0;b2<8;b2++)nbr[v][b2]=(uint8_t)(v^(1<<b2));
    memset(idx_sz,0,sizeof(idx_sz));
    for(int i=0;i<TRAIN_N;i++){const uint8_t*sig=joint_tr+(size_t)i*SIG_PAD;
        for(int k=0;k<N_BLOCKS;k++)if(sig[k]!=bg_val)idx_sz[k][sig[k]]++;}
    uint32_t tot=0;for(int k=0;k<N_BLOCKS;k++)for(int v=0;v<BYTE_VALS;v++){idx_off[k][v]=tot;tot+=idx_sz[k][v];}
    idx_pool=malloc((size_t)tot*4);
    uint32_t(*wp)[BYTE_VALS]=malloc((size_t)N_BLOCKS*BYTE_VALS*4);memcpy(wp,idx_off,(size_t)N_BLOCKS*BYTE_VALS*4);
    for(int i=0;i<TRAIN_N;i++){const uint8_t*sig=joint_tr+(size_t)i*SIG_PAD;
        for(int k=0;k<N_BLOCKS;k++)if(sig[k]!=bg_val)idx_pool[wp[k][sig[k]]++]=(uint32_t)i;}free(wp);
    printf("  Index: %u entries\n",tot);
    printf("  Built (%.1f sec)\n\n",now_sec()-t0);

    /* ================================================================
     * Test: compare ternary dot vs MT4 dot vs MT4 grad dot
     * ================================================================ */
    printf("Running cascade with 3 dot variants...\n");

    uint32_t *vbuf=calloc(TRAIN_N,4);
    cand_t *cands=malloc(TOP_K*sizeof(cand_t));

    int tern_px_correct[4]={0};   /* ternary pixel dot, k=1,3,5,7 */
    int tern_grad_correct[4]={0}; /* ternary gradient dot */
    int mt4_px_correct[4]={0};    /* MT4 pixel dot */
    int mt4_grad_correct[4]={0};  /* MT4 gradient dot */
    int mt4_both_correct[4]={0};  /* MT4 pixel + gradient dot */
    int knn_v[]={1,3,5,7};

    double tc=now_sec();
    for(int i=0;i<TEST_N;i++){
        /* Vote */
        memset(vbuf,0,TRAIN_N*4);const uint8_t*sig=joint_te+(size_t)i*SIG_PAD;
        for(int k=0;k<N_BLOCKS;k++){uint8_t bv=sig[k];if(bv==bg_val)continue;
            uint16_t w=ig_w[k],wh=w>1?w/2:1;
            {uint32_t off=idx_off[k][bv];uint16_t sz=idx_sz[k][bv];for(uint16_t j=0;j<sz;j++)vbuf[idx_pool[off+j]]+=w;}
            for(int nb=0;nb<8;nb++){uint8_t nv=nbr[bv][nb];if(nv==bg_val)continue;
                uint32_t no=idx_off[k][nv];uint16_t ns=idx_sz[k][nv];for(uint16_t j=0;j<ns;j++)vbuf[idx_pool[no+j]]+=wh;}}

        int nc=topk(vbuf,TRAIN_N,cands,TOP_K);

        /* --- Variant 1: ternary pixel dot (baseline) --- */
        for(int j=0;j<nc;j++)cands[j].dot=tdot(tern_te+(size_t)i*PADDED,tern_tr+(size_t)cands[j].id*PADDED);
        qsort(cands,(size_t)nc,sizeof(cand_t),cmpd);
        for(int ki=0;ki<4;ki++)if(knn(cands,nc,knn_v[ki])==test_labels[i])tern_px_correct[ki]++;

        /* --- Variant 2: ternary gradient dot --- */
        for(int j=0;j<nc;j++){uint32_t id=cands[j].id;
            cands[j].dot=tdot(hgc_te+(size_t)i*PADDED,hgc_tr+(size_t)id*PADDED)
                        +tdot(vgc_te+(size_t)i*PADDED,vgc_tr+(size_t)id*PADDED);}
        qsort(cands,(size_t)nc,sizeof(cand_t),cmpd);
        for(int ki=0;ki<4;ki++)if(knn(cands,nc,knn_v[ki])==test_labels[i])tern_grad_correct[ki]++;

        /* --- Variant 3: MT4 pixel dot (81-level) --- */
        for(int j=0;j<nc;j++){uint32_t id=cands[j].id;
            cands[j].dot=mt4_dot(t3_te+(size_t)i*PADDED,t2_te+(size_t)i*PADDED,
                                  t1_te+(size_t)i*PADDED,t0_te+(size_t)i*PADDED,
                                  t3_tr+(size_t)id*PADDED,t2_tr+(size_t)id*PADDED,
                                  t1_tr+(size_t)id*PADDED,t0_tr+(size_t)id*PADDED);}
        qsort(cands,(size_t)nc,sizeof(cand_t),cmpd);
        for(int ki=0;ki<4;ki++)if(knn(cands,nc,knn_v[ki])==test_labels[i])mt4_px_correct[ki]++;

        /* --- Variant 4: MT4 gradient dot (81-level gradients) --- */
        for(int j=0;j<nc;j++){cands[j].dot=mt4_grad_dot(i,cands[j].id);}
        qsort(cands,(size_t)nc,sizeof(cand_t),cmpd);
        for(int ki=0;ki<4;ki++)if(knn(cands,nc,knn_v[ki])==test_labels[i])mt4_grad_correct[ki]++;

        /* --- Variant 5: MT4 pixel + gradient combined --- */
        for(int j=0;j<nc;j++){uint32_t id=cands[j].id;
            cands[j].dot=mt4_dot(t3_te+(size_t)i*PADDED,t2_te+(size_t)i*PADDED,
                                  t1_te+(size_t)i*PADDED,t0_te+(size_t)i*PADDED,
                                  t3_tr+(size_t)id*PADDED,t2_tr+(size_t)id*PADDED,
                                  t1_tr+(size_t)id*PADDED,t0_tr+(size_t)id*PADDED)
                         +mt4_grad_dot(i,id);}
        qsort(cands,(size_t)nc,sizeof(cand_t),cmpd);
        for(int ki=0;ki<4;ki++)if(knn(cands,nc,knn_v[ki])==test_labels[i])mt4_both_correct[ki]++;

        if((i+1)%1000==0)fprintf(stderr,"  %d/%d\r",i+1,TEST_N);
    }
    fprintf(stderr,"\n");

    printf("\n=== RESULTS ===\n\n");
    printf("  %-30s  k=1     k=3     k=5     k=7\n","Method");
    printf("  %-30s","Ternary pixel dot (3-level)");
    for(int ki=0;ki<4;ki++)printf("  %5.2f%%",100.0*tern_px_correct[ki]/TEST_N);printf("\n");
    printf("  %-30s","Ternary grad dot (3-level)");
    for(int ki=0;ki<4;ki++)printf("  %5.2f%%",100.0*tern_grad_correct[ki]/TEST_N);printf("\n");
    printf("  %-30s","MT4 pixel dot (81-level)");
    for(int ki=0;ki<4;ki++)printf("  %5.2f%%",100.0*mt4_px_correct[ki]/TEST_N);printf("\n");
    printf("  %-30s","MT4 grad dot (81-level)");
    for(int ki=0;ki<4;ki++)printf("  %5.2f%%",100.0*mt4_grad_correct[ki]/TEST_N);printf("\n");
    printf("  %-30s","MT4 pixel+grad (81-level)");
    for(int ki=0;ki<4;ki++)printf("  %5.2f%%",100.0*mt4_both_correct[ki]/TEST_N);printf("\n");

    printf("\n  Bayesian baseline (no dot): 36.58%% (from sstt_cifar10_flat)\n");
    printf("\nTotal: %.1f sec\n",now_sec()-t0);

    free(vbuf);free(cands);
    return 0;
}
