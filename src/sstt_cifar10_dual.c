/*
 * sstt_cifar10_dual.c — Dual Quantization: Fixed + Adaptive Combined
 *
 * Fixed thresholds are strong for: airplane (+50%), ship (+50%), frog (+59%)
 * Adaptive thresholds are strong for: cat (+18%), automobile (+46%), deer (+32%)
 *
 * Combine both: run both pipelines, merge votes, combine Bayesian posteriors.
 * Each pipeline contributes where it's strongest.
 *
 * Build: gcc -O3 -mavx2 -march=native -o sstt_cifar10_dual src/sstt_cifar10_dual.c -lm
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

/* Core functions — identical to all other experiments */
static inline int8_t ct(int v){return v>0?1:v<0?-1:0;}
static int cmp_u8(const void*a,const void*b){return *(const uint8_t*)a-*(const uint8_t*)b;}
static void fixed_qt(const uint8_t*s,int8_t*d,int n){
    const __m256i bi=_mm256_set1_epi8((char)0x80),th=_mm256_set1_epi8((char)(170^0x80)),
        tl=_mm256_set1_epi8((char)(85^0x80)),on=_mm256_set1_epi8(1);
    for(int i2=0;i2<n;i2++){const uint8_t*si=s+(size_t)i2*PIXELS;int8_t*di=d+(size_t)i2*PADDED;
        for(int i=0;i<PIXELS;i+=32){__m256i p=_mm256_loadu_si256((const __m256i*)(si+i));__m256i sp=_mm256_xor_si256(p,bi);
            _mm256_storeu_si256((__m256i*)(di+i),_mm256_sub_epi8(_mm256_and_si256(_mm256_cmpgt_epi8(sp,th),on),
                _mm256_and_si256(_mm256_cmpgt_epi8(tl,sp),on)));}}}
static void adaptive_qt(const uint8_t*s,int8_t*d,int n){
    uint8_t*sorted=malloc(PIXELS);
    for(int img=0;img<n;img++){const uint8_t*si=s+(size_t)img*PIXELS;int8_t*di=d+(size_t)img*PADDED;
        memcpy(sorted,si,PIXELS);qsort(sorted,PIXELS,1,cmp_u8);
        uint8_t p33=sorted[PIXELS/3],p67=sorted[2*PIXELS/3];
        if(p33==p67){p33=p33>0?p33-1:0;p67=p67<255?p67+1:255;}
        for(int i=0;i<PIXELS;i++)di[i]=si[i]>p67?1:si[i]<p33?-1:0;}free(sorted);}
static void gr(const int8_t*t,int8_t*h,int8_t*v,int n){
    for(int i2=0;i2<n;i2++){const int8_t*ti=t+(size_t)i2*PADDED;int8_t*hi=h+(size_t)i2*PADDED,*vi=v+(size_t)i2*PADDED;
        for(int y=0;y<IMG_H;y++){for(int x=0;x<IMG_W-1;x++)hi[y*IMG_W+x]=ct(ti[y*IMG_W+x+1]-ti[y*IMG_W+x]);hi[y*IMG_W+IMG_W-1]=0;}
        for(int y=0;y<IMG_H-1;y++)for(int x=0;x<IMG_W;x++)vi[y*IMG_W+x]=ct(ti[(y+1)*IMG_W+x]-ti[y*IMG_W+x]);memset(vi+(IMG_H-1)*IMG_W,0,IMG_W);}}
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
static int32_t tdot(const int8_t*a,const int8_t*b){int32_t tot=0;
    for(int ch=0;ch<PADDED;ch+=64){__m256i ac=_mm256_setzero_si256();int e=ch+64;
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

/* Pipeline state */
typedef struct {
    int8_t *hg_tr,*hg_te,*vg_tr,*vg_te;
    uint8_t *joint_tr,*joint_te;
    uint32_t *hot;
    uint16_t ig[N_BLOCKS];
    uint8_t bg,nbr[BYTE_VALS][8];
    uint32_t idx_off[N_BLOCKS][BYTE_VALS];
    uint16_t idx_sz[N_BLOCKS][BYTE_VALS];
    uint32_t *idx_pool;
} pipe_t;

static void build_pipe(pipe_t *p, const int8_t *tern_tr, const int8_t *tern_te) {
    p->hg_tr=aligned_alloc(32,(size_t)TRAIN_N*PADDED);p->hg_te=aligned_alloc(32,(size_t)TEST_N*PADDED);
    p->vg_tr=aligned_alloc(32,(size_t)TRAIN_N*PADDED);p->vg_te=aligned_alloc(32,(size_t)TEST_N*PADDED);
    gr(tern_tr,p->hg_tr,p->vg_tr,TRAIN_N);gr(tern_te,p->hg_te,p->vg_te,TEST_N);
    uint8_t *pxt=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD),*pxe=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    uint8_t *hst=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD),*hse=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    uint8_t *vst=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD),*vse=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    bsf(tern_tr,pxt,TRAIN_N);bsf(tern_te,pxe,TEST_N);
    bsf(p->hg_tr,hst,TRAIN_N);bsf(p->hg_te,hse,TEST_N);
    bsf(p->vg_tr,vst,TRAIN_N);bsf(p->vg_te,vse,TEST_N);
    uint8_t *htt=aligned_alloc(32,(size_t)TRAIN_N*TRANS_PAD),*hte=aligned_alloc(32,(size_t)TEST_N*TRANS_PAD);
    uint8_t *vtt=aligned_alloc(32,(size_t)TRAIN_N*TRANS_PAD),*vte=aligned_alloc(32,(size_t)TEST_N*TRANS_PAD);
    trf(pxt,htt,vtt,TRAIN_N);trf(pxe,hte,vte,TEST_N);
    p->joint_tr=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD);p->joint_te=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    jsf(p->joint_tr,TRAIN_N,pxt,hst,vst,htt,vtt);jsf(p->joint_te,TEST_N,pxe,hse,vse,hte,vte);
    free(pxt);free(pxe);free(hst);free(hse);free(vst);free(vse);free(htt);free(hte);free(vtt);free(vte);
    long vc[BYTE_VALS]={0};for(int i=0;i<TRAIN_N;i++){const uint8_t*sig=p->joint_tr+(size_t)i*SIG_PAD;
        for(int k=0;k<N_BLOCKS;k++)vc[sig[k]]++;}
    p->bg=0;long mc=0;for(int v=0;v<BYTE_VALS;v++)if(vc[v]>mc){mc=vc[v];p->bg=(uint8_t)v;}
    p->hot=aligned_alloc(32,(size_t)N_BLOCKS*BYTE_VALS*CLS_PAD*4);
    memset(p->hot,0,(size_t)N_BLOCKS*BYTE_VALS*CLS_PAD*4);
    for(int i=0;i<TRAIN_N;i++){int l=train_labels[i];const uint8_t*sig=p->joint_tr+(size_t)i*SIG_PAD;
        for(int k=0;k<N_BLOCKS;k++)p->hot[(size_t)k*BYTE_VALS*CLS_PAD+(size_t)sig[k]*CLS_PAD+l]++;}
    {int cc[N_CLASSES]={0};for(int i=0;i<TRAIN_N;i++)cc[train_labels[i]]++;
     double hc=0;for(int c=0;c<N_CLASSES;c++){double pp=(double)cc[c]/TRAIN_N;if(pp>0)hc-=pp*log2(pp);}
     double raw2[N_BLOCKS],mx=0;
     for(int k=0;k<N_BLOCKS;k++){double hcond=0;
         for(int v=0;v<BYTE_VALS;v++){if((uint8_t)v==p->bg)continue;
             const uint32_t*h=p->hot+(size_t)k*BYTE_VALS*CLS_PAD+(size_t)v*CLS_PAD;
             int vt=0;for(int c=0;c<N_CLASSES;c++)vt+=(int)h[c];if(!vt)continue;
             double pv=(double)vt/TRAIN_N,hv=0;for(int c=0;c<N_CLASSES;c++){double pc=(double)h[c]/vt;if(pc>0)hv-=pc*log2(pc);}
             hcond+=pv*hv;}raw2[k]=hc-hcond;if(raw2[k]>mx)mx=raw2[k];}
     for(int k=0;k<N_BLOCKS;k++){p->ig[k]=mx>0?(uint16_t)(raw2[k]/mx*IG_SCALE+0.5):1;if(!p->ig[k])p->ig[k]=1;}}
    for(int v=0;v<BYTE_VALS;v++)for(int b3=0;b3<8;b3++)p->nbr[v][b3]=(uint8_t)(v^(1<<b3));
    memset(p->idx_sz,0,sizeof(p->idx_sz));
    for(int i=0;i<TRAIN_N;i++){const uint8_t*sig=p->joint_tr+(size_t)i*SIG_PAD;
        for(int k=0;k<N_BLOCKS;k++)if(sig[k]!=p->bg)p->idx_sz[k][sig[k]]++;}
    uint32_t tot=0;for(int k=0;k<N_BLOCKS;k++)for(int v=0;v<BYTE_VALS;v++){p->idx_off[k][v]=tot;tot+=p->idx_sz[k][v];}
    p->idx_pool=malloc((size_t)tot*4);
    uint32_t(*wp)[BYTE_VALS]=malloc((size_t)N_BLOCKS*BYTE_VALS*4);memcpy(wp,p->idx_off,(size_t)N_BLOCKS*BYTE_VALS*4);
    for(int i=0;i<TRAIN_N;i++){const uint8_t*sig=p->joint_tr+(size_t)i*SIG_PAD;
        for(int k=0;k<N_BLOCKS;k++)if(sig[k]!=p->bg)p->idx_pool[wp[k][sig[k]]++]=(uint32_t)i;}free(wp);
    printf("    BG=%d (%.1f%%), %u entries\n",p->bg,100.0*mc/((long)TRAIN_N*N_BLOCKS),tot);
}

static void vote_pipe(const pipe_t*p,int img,uint32_t*votes,int weight){
    const uint8_t*sig=p->joint_te+(size_t)img*SIG_PAD;
    for(int k=0;k<N_BLOCKS;k++){uint8_t bv=sig[k];if(bv==p->bg)continue;
        uint16_t w=(uint16_t)(p->ig[k]*weight),wh=w>1?w/2:1;
        {uint32_t off=p->idx_off[k][bv];uint16_t sz=p->idx_sz[k][bv];
         for(uint16_t j=0;j<sz;j++)votes[p->idx_pool[off+j]]+=w;}
        for(int nb=0;nb<8;nb++){uint8_t nv=p->nbr[bv][nb];if(nv==p->bg)continue;
            uint32_t noff=p->idx_off[k][nv];uint16_t nsz=p->idx_sz[k][nv];
            for(uint16_t j=0;j<nsz;j++)votes[p->idx_pool[noff+j]]+=wh;}}}

static void bay_pipe(const pipe_t*p,int img,double*bp){
    const uint8_t*sig=p->joint_te+(size_t)img*SIG_PAD;
    for(int k=0;k<N_BLOCKS;k++){uint8_t bv=sig[k];if(bv==p->bg)continue;
        const uint32_t*h=p->hot+(size_t)k*BYTE_VALS*CLS_PAD+(size_t)bv*CLS_PAD;
        for(int c=0;c<N_CLASSES;c++)bp[c]+=log(h[c]+0.5);}}

static void load_data(void){
    raw_train=malloc((size_t)TRAIN_N*PIXELS);raw_test=malloc((size_t)TEST_N*PIXELS);
    train_labels=malloc(TRAIN_N);test_labels=malloc(TEST_N);
    char p[512];uint8_t rec[3073];
    for(int b2=1;b2<=5;b2++){snprintf(p,sizeof(p),"%sdata_batch_%d.bin",data_dir,b2);
        FILE*f=fopen(p,"rb");for(int i=0;i<10000;i++){if(fread(rec,1,3073,f)!=3073){fclose(f);exit(1);}
            int idx=(b2-1)*10000+i;train_labels[idx]=rec[0];
            uint8_t*d=raw_train+(size_t)idx*PIXELS;const uint8_t*r=rec+1,*g=rec+1+1024,*b3=rec+1+2048;
            for(int y=0;y<32;y++)for(int x=0;x<32;x++){int si=y*32+x,di=y*96+x*3;d[di]=r[si];d[di+1]=g[si];d[di+2]=b3[si];}}fclose(f);}
    snprintf(p,sizeof(p),"%stest_batch.bin",data_dir);
    FILE*f=fopen(p,"rb");for(int i=0;i<10000;i++){if(fread(rec,1,3073,f)!=3073){fclose(f);exit(1);}
        test_labels[i]=rec[0];uint8_t*d=raw_test+(size_t)i*PIXELS;const uint8_t*r=rec+1,*g=rec+1+1024,*b3=rec+1+2048;
        for(int y=0;y<32;y++)for(int x=0;x<32;x++){int si=y*32+x,di=y*96+x*3;d[di]=r[si];d[di+1]=g[si];d[di+2]=b3[si];}}fclose(f);}

int main(int argc,char**argv){
    double t0=now_sec();
    if(argc>1){data_dir=argv[1];size_t l=strlen(data_dir);
        if(l&&data_dir[l-1]!='/'){char*b2=malloc(l+2);memcpy(b2,data_dir,l);b2[l]='/';b2[l+1]=0;data_dir=b2;}}
    printf("=== SSTT CIFAR-10: Dual Quantization (Fixed + Adaptive) ===\n\n");
    load_data();

    /* Build both pipelines */
    printf("Building fixed pipeline...\n");
    int8_t *fix_tr=aligned_alloc(32,(size_t)TRAIN_N*PADDED),*fix_te=aligned_alloc(32,(size_t)TEST_N*PADDED);
    fixed_qt(raw_train,fix_tr,TRAIN_N);fixed_qt(raw_test,fix_te,TEST_N);
    pipe_t pf={0};build_pipe(&pf,fix_tr,fix_te);free(fix_tr);free(fix_te);

    printf("Building adaptive pipeline...\n");
    int8_t *adp_tr=aligned_alloc(32,(size_t)TRAIN_N*PADDED),*adp_te=aligned_alloc(32,(size_t)TEST_N*PADDED);
    adaptive_qt(raw_train,adp_tr,TRAIN_N);adaptive_qt(raw_test,adp_te,TEST_N);
    pipe_t pa={0};build_pipe(&pa,adp_tr,adp_te);free(adp_tr);free(adp_te);
    printf("  Built (%.1f sec)\n\n",now_sec()-t0);

    /* ================================================================ */
    uint32_t *vbuf=calloc(TRAIN_N,4);

    /* Test 1: Combined Bayesian (sum log-posteriors from both) */
    printf("--- Test 1: Combined Bayesian (fixed + adaptive posteriors) ---\n");
    {int correct=0;int pc[N_CLASSES]={0},pt[N_CLASSES]={0};
     for(int i=0;i<TEST_N;i++){pt[test_labels[i]]++;
         double bp[N_CLASSES];memset(bp,0,sizeof(bp));
         bay_pipe(&pf,i,bp);bay_pipe(&pa,i,bp); /* sum both */
         double mx=-1e30;for(int c=0;c<N_CLASSES;c++)if(bp[c]>mx)mx=bp[c];
         for(int c=0;c<N_CLASSES;c++)bp[c]-=mx;
         int pred=0;for(int c=1;c<N_CLASSES;c++)if(bp[c]>bp[pred])pred=c;
         if(pred==test_labels[i]){correct++;pc[test_labels[i]]++;}}
     printf("  Combined Bayesian: %.2f%%\n",100.0*correct/TEST_N);
     for(int c=0;c<N_CLASSES;c++)printf("    %d %-10s %.1f%%\n",c,cn[c],100.0*pc[c]/pt[c]);}

    /* Test 2: Combined vote → gradient dot from BOTH pipelines → k=1 */
    printf("\n--- Test 2: Dual vote + dual gradient dot ---\n");
    {int correct=0;int pc[N_CLASSES]={0},pt[N_CLASSES]={0};
     for(int i=0;i<TEST_N;i++){pt[test_labels[i]]++;
         memset(vbuf,0,TRAIN_N*4);
         vote_pipe(&pf,i,vbuf,1);vote_pipe(&pa,i,vbuf,1);
         cand_t cands[TOP_K];int nc=topk(vbuf,TRAIN_N,cands,TOP_K);
         /* Dual gradient dot */
         for(int j=0;j<nc;j++){uint32_t id=cands[j].id;
             cands[j].score=(int64_t)(tdot(pf.hg_te+(size_t)i*PADDED,pf.hg_tr+(size_t)id*PADDED)
                                     +tdot(pf.vg_te+(size_t)i*PADDED,pf.vg_tr+(size_t)id*PADDED))
                           +(int64_t)(tdot(pa.hg_te+(size_t)i*PADDED,pa.hg_tr+(size_t)id*PADDED)
                                     +tdot(pa.vg_te+(size_t)i*PADDED,pa.vg_tr+(size_t)id*PADDED));}
         qsort(cands,(size_t)nc,sizeof(cand_t),cmps);
         int pred=(nc>0)?train_labels[cands[0].id]:0;
         if(pred==test_labels[i]){correct++;pc[test_labels[i]]++;}
         if((i+1)%2000==0)fprintf(stderr,"  %d/%d\r",i+1,TEST_N);}
     fprintf(stderr,"\n");
     printf("  Dual cascade: %.2f%%\n",100.0*correct/TEST_N);
     for(int c=0;c<N_CLASSES;c++)printf("    %d %-10s %.1f%%\n",c,cn[c],100.0*pc[c]/pt[c]);}

    /* Test 3: Dual vote + dual dot + dual Bayesian prior */
    printf("\n--- Test 3: Full dual stack (vote + dot + Bayesian prior) ---\n");
    {int correct=0;int pc[N_CLASSES]={0},pt[N_CLASSES]={0};
     for(int i=0;i<TEST_N;i++){pt[test_labels[i]]++;
         memset(vbuf,0,TRAIN_N*4);
         vote_pipe(&pf,i,vbuf,1);vote_pipe(&pa,i,vbuf,1);
         cand_t cands[TOP_K];int nc=topk(vbuf,TRAIN_N,cands,TOP_K);
         /* Dual Bayesian posterior */
         double bp[N_CLASSES];memset(bp,0,sizeof(bp));
         bay_pipe(&pf,i,bp);bay_pipe(&pa,i,bp);
         double mx=-1e30;for(int c=0;c<N_CLASSES;c++)if(bp[c]>mx)mx=bp[c];
         for(int c=0;c<N_CLASSES;c++)bp[c]-=mx;
         /* Dual dot + Bayesian prior */
         for(int j=0;j<nc;j++){uint32_t id=cands[j].id;
             int64_t d=(int64_t)(tdot(pf.hg_te+(size_t)i*PADDED,pf.hg_tr+(size_t)id*PADDED)
                                +tdot(pf.vg_te+(size_t)i*PADDED,pf.vg_tr+(size_t)id*PADDED))
                      +(int64_t)(tdot(pa.hg_te+(size_t)i*PADDED,pa.hg_tr+(size_t)id*PADDED)
                                +tdot(pa.vg_te+(size_t)i*PADDED,pa.vg_tr+(size_t)id*PADDED));
             uint8_t lbl=train_labels[id];
             cands[j].score=(int64_t)512*d+(int64_t)50*(int64_t)(bp[lbl]*1000);}
         qsort(cands,(size_t)nc,sizeof(cand_t),cmps);
         int pred=(nc>0)?train_labels[cands[0].id]:0;
         if(pred==test_labels[i]){correct++;pc[test_labels[i]]++;}
         if((i+1)%2000==0)fprintf(stderr,"  %d/%d\r",i+1,TEST_N);}
     fprintf(stderr,"\n");
     printf("  Full dual stack: %.2f%%\n",100.0*correct/TEST_N);
     for(int c=0;c<N_CLASSES;c++)printf("    %d %-10s %.1f%%\n",c,cn[c],100.0*pc[c]/pt[c]);}

    printf("\n=== BASELINES ===\n");
    printf("  Fixed only Bayesian:     36.58%%\n");
    printf("  Adaptive only Bayesian:  36.95%%\n");
    printf("  MT4 full stack (best):   42.05%%\n");

    printf("\nTotal: %.1f sec\n",now_sec()-t0);
    free(vbuf);return 0;
}
