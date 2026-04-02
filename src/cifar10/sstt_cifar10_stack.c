/*
 * sstt_cifar10_stack.c — Full Stacked Pipeline on CIFAR-10
 *
 * Everything we have, applied in the right order:
 *
 *   1. VOTE: Flattened RGB Encoding D, multi-probe IG (97.96% recall)
 *   2. BAYESIAN PRIOR: P(class|query) from hot map — fed into ranking
 *   3. RANK: MT4 pixel+grad dot (81-level) + per-RGB-channel divergence
 *            + grid div (gray + RGB) + Bayesian prior boost
 *   4. DECIDE: k=1 (kNN dilutes on CIFAR) or score-weighted
 *
 * The Bayesian posterior is a FEATURE for ranking, not just a baseline.
 * Per-channel RGB divergence captures color-specific topology:
 * green loops (frogs), blue boundaries (sky/ships), red contours.
 *
 * Build: gcc -O3 -mavx2 -march=native -o sstt_cifar10_stack src/sstt_cifar10_stack.c -lm
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
#define MAX_REGIONS 16
#define SPATIAL_W 32
#define SPATIAL_H 32
#define FG_W 34
#define FG_H 34
#define FG_SZ (FG_W*FG_H)
#define GRAY_PIXELS 1024
#define GRAY_PADDED 1024
#define CENT_BOTH_OPEN 50
#define CENT_DISAGREE 80

static const char *data_dir="data-cifar10/";
static const char *cn[]={"airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"};
static double now_sec(void){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return ts.tv_sec+ts.tv_nsec*1e-9;}

static uint8_t *raw_train,*raw_test,*raw_gray_tr,*raw_gray_te,*train_labels,*test_labels;
static uint8_t *raw_r_tr,*raw_r_te,*raw_g_tr,*raw_g_te,*raw_b_tr,*raw_b_te;

/* MT4 planes */
static int8_t *t3_tr,*t3_te,*t2_tr,*t2_te,*t1_tr,*t1_te,*t0_tr,*t0_te;
/* MT4 gradients (coarsest + finest for efficiency) */
static int8_t *hg3_tr,*hg3_te,*vg3_tr,*vg3_te;
static int8_t *hg0_tr,*hg0_te,*vg0_tr,*vg0_te;
/* Full ternary for voting */
static int8_t *tern_tr,*tern_te,*hgc_tr,*hgc_te,*vgc_tr,*vgc_te;
/* Grayscale topo features */
static int8_t *ghg_tr,*ghg_te,*gvg_tr,*gvg_te;
static int16_t *cent_tr,*cent_te,*divneg_tr,*divneg_te,*divcy_tr,*divcy_te;
static int16_t *gdiv_tr,*gdiv_te;
/* Per-RGB-channel divergence */
static int16_t *rdiv_tr,*rdiv_te,*gdivr_tr,*gdivr_te;  /* R channel div + grid div */
static int16_t *gdiv_g_tr,*gdiv_g_te,*gdivg_tr,*gdivg_te;  /* G channel */
static int16_t *bdiv_tr,*bdiv_te,*gdivb_tr,*gdivb_te;  /* B channel */
/* Per-channel ternary + gradients */
static int8_t *r_hg_tr,*r_hg_te,*r_vg_tr,*r_vg_te;
static int8_t *g_hg_tr,*g_hg_te,*g_vg_tr,*g_vg_te;
static int8_t *b_hg_tr,*b_hg_te,*b_vg_tr,*b_vg_te;
/* Voting infra */
static uint8_t *joint_tr,*joint_te;
static uint32_t *joint_hot;
static uint16_t ig_w[N_BLOCKS];
static uint8_t nbr[BYTE_VALS][8];
static uint32_t idx_off[N_BLOCKS][BYTE_VALS];
static uint16_t idx_sz[N_BLOCKS][BYTE_VALS];
static uint32_t *idx_pool;
static uint8_t bg_val;

/* ================================================================ */
static inline int8_t ct(int v){return v>0?1:v<0?-1:0;}

static void mt4_quant(const uint8_t *src,int n,int8_t *p3,int8_t *p2,int8_t *p1,int8_t *p0){
    for(int img=0;img<n;img++){const uint8_t*s=src+(size_t)img*PIXELS;
        int8_t *d3=p3+(size_t)img*PADDED,*d2=p2+(size_t)img*PADDED,
               *d1=p1+(size_t)img*PADDED,*d0=p0+(size_t)img*PADDED;
        for(int i=0;i<PIXELS;i++){int bin=(int)s[i]*81/256;
            d3[i]=(int8_t)(bin/27-1);int r=bin%27;d2[i]=(int8_t)(r/9-1);r%=9;d1[i]=(int8_t)(r/3-1);d0[i]=(int8_t)(r%3-1);}}}

static void quant_tern(const uint8_t *s,int8_t *d,int n,int px,int pd){
    const __m256i bi=_mm256_set1_epi8((char)0x80),th=_mm256_set1_epi8((char)(170^0x80)),
                 tl=_mm256_set1_epi8((char)(85^0x80)),on=_mm256_set1_epi8(1);
    for(int i2=0;i2<n;i2++){const uint8_t*si=s+(size_t)i2*px;int8_t*di=d+(size_t)i2*pd;int i;
        for(i=0;i+32<=px;i+=32){__m256i p=_mm256_loadu_si256((const __m256i*)(si+i));__m256i sp=_mm256_xor_si256(p,bi);
            _mm256_storeu_si256((__m256i*)(di+i),_mm256_sub_epi8(_mm256_and_si256(_mm256_cmpgt_epi8(sp,th),on),
                _mm256_and_si256(_mm256_cmpgt_epi8(tl,sp),on)));}
        for(;i<px;i++)di[i]=si[i]>170?1:si[i]<85?-1:0;memset(di+px,0,pd-px);}}

static void grads(const int8_t*t,int8_t*h,int8_t*v,int n,int w,int ht,int pd){
    for(int i2=0;i2<n;i2++){const int8_t*ti=t+(size_t)i2*pd;int8_t*hi=h+(size_t)i2*pd,*vi=v+(size_t)i2*pd;
        for(int y=0;y<ht;y++){for(int x=0;x<w-1;x++)hi[y*w+x]=ct(ti[y*w+x+1]-ti[y*w+x]);hi[y*w+w-1]=0;}
        for(int y=0;y<ht-1;y++)for(int x=0;x<w;x++)vi[y*w+x]=ct(ti[(y+1)*w+x]-ti[y*w+x]);memset(vi+(ht-1)*w,0,w);}}

static inline uint8_t be(int8_t a,int8_t b,int8_t c){return(uint8_t)((a+1)*9+(b+1)*3+(c+1));}
static void bsigs(const int8_t*d,uint8_t*s,int n){for(int i=0;i<n;i++){const int8_t*im=d+(size_t)i*PADDED;
    uint8_t*si=s+(size_t)i*SIG_PAD;for(int y=0;y<IMG_H;y++)for(int s2=0;s2<BLKS_PER_ROW;s2++){
        int b2=y*IMG_W+s2*3;si[y*BLKS_PER_ROW+s2]=be(im[b2],im[b2+1],im[b2+2]);}}}
static inline uint8_t te2(uint8_t a,uint8_t b){int8_t a0=(a/9)-1,a1=((a/3)%3)-1,a2=(a%3)-1,
    b0=(b/9)-1,b1=((b/3)%3)-1,b2=(b%3)-1;return be(ct(b0-a0),ct(b1-a1),ct(b2-a2));}
static void trans(const uint8_t*bs,uint8_t*ht,uint8_t*vt,int n){
    for(int i=0;i<n;i++){const uint8_t*s=bs+(size_t)i*SIG_PAD;uint8_t*h=ht+(size_t)i*TRANS_PAD,*v=vt+(size_t)i*TRANS_PAD;
        for(int y=0;y<IMG_H;y++)for(int ss=0;ss<H_TRANS_PER_ROW;ss++)h[y*H_TRANS_PER_ROW+ss]=te2(s[y*BLKS_PER_ROW+ss],s[y*BLKS_PER_ROW+ss+1]);
        memset(h+N_HTRANS,0xFF,TRANS_PAD-N_HTRANS);
        for(int y=0;y<V_TRANS_PER_COL;y++)for(int ss=0;ss<BLKS_PER_ROW;ss++)v[y*BLKS_PER_ROW+ss]=te2(s[y*BLKS_PER_ROW+ss],s[(y+1)*BLKS_PER_ROW+ss]);
        memset(v+N_VTRANS,0xFF,TRANS_PAD-N_VTRANS);}}
static void jsigs(uint8_t*o,int n,const uint8_t*px,const uint8_t*hg,const uint8_t*vg,const uint8_t*ht,const uint8_t*vt){
    for(int i=0;i<n;i++){const uint8_t*pi=px+(size_t)i*SIG_PAD,*hi=hg+(size_t)i*SIG_PAD,*vi=vg+(size_t)i*SIG_PAD;
        const uint8_t*hti=ht+(size_t)i*TRANS_PAD,*vti=vt+(size_t)i*TRANS_PAD;uint8_t*oi=o+(size_t)i*SIG_PAD;
        for(int y=0;y<IMG_H;y++)for(int s=0;s<BLKS_PER_ROW;s++){int k=y*BLKS_PER_ROW+s;
            uint8_t htb=(s>0)?hti[y*H_TRANS_PER_ROW+(s-1)]:BG_TRANS,vtb=(y>0)?vti[(y-1)*BLKS_PER_ROW+s]:BG_TRANS;
            int ps=((pi[k]/9)-1)+(((pi[k]/3)%3)-1)+((pi[k]%3)-1);
            int hs=((hi[k]/9)-1)+(((hi[k]/3)%3)-1)+((hi[k]%3)-1);
            int vs=((vi[k]/9)-1)+(((vi[k]/3)%3)-1)+((vi[k]%3)-1);
            uint8_t pc=ps<0?0:ps==0?1:ps<3?2:3,hc=hs<0?0:hs==0?1:hs<3?2:3,vc=vs<0?0:vs==0?1:vs<3?2:3;
            oi[k]=pc|(hc<<2)|(vc<<4)|((htb!=BG_TRANS)?1<<6:0)|((vtb!=BG_TRANS)?1<<7:0);}}}

/* Topo features on grayscale */
static int16_t enclosed_centroid(const int8_t*tern){
    uint8_t grid[FG_SZ];memset(grid,0,sizeof(grid));
    for(int y=0;y<SPATIAL_H;y++)for(int x=0;x<SPATIAL_W;x++)grid[(y+1)*FG_W+(x+1)]=(tern[y*SPATIAL_W+x]>0)?1:0;
    uint8_t vis[FG_SZ];memset(vis,0,sizeof(vis));int stk[FG_SZ];int sp=0;
    for(int y=0;y<FG_H;y++)for(int x=0;x<FG_W;x++){if(y==0||y==FG_H-1||x==0||x==FG_W-1){
        int pos=y*FG_W+x;if(!grid[pos]&&!vis[pos]){vis[pos]=1;stk[sp++]=pos;}}}
    while(sp>0){int pos=stk[--sp];int py=pos/FG_W,px2=pos%FG_W;const int dx[]={0,0,1,-1},dy[]={1,-1,0,0};
        for(int d=0;d<4;d++){int ny=py+dy[d],nx=px2+dx[d];if(ny<0||ny>=FG_H||nx<0||nx>=FG_W)continue;
            int np=ny*FG_W+nx;if(!vis[np]&&!grid[np]){vis[np]=1;stk[sp++]=np;}}}
    int sy=0,cnt=0;for(int y=1;y<FG_H-1;y++)for(int x=1;x<FG_W-1;x++){
        int pos=y*FG_W+x;if(!grid[pos]&&!vis[pos]){sy+=(y-1);cnt++;}}
    return cnt>0?(int16_t)(sy/cnt):-1;}

static void divf(const int8_t*hg,const int8_t*vg,int16_t*ns,int16_t*cy){
    int neg=0,ny=0,nc2=0;for(int y=0;y<SPATIAL_H;y++)for(int x=0;x<SPATIAL_W;x++){
        int dh=(int)hg[y*SPATIAL_W+x]-(x>0?(int)hg[y*SPATIAL_W+x-1]:0);
        int dv=(int)vg[y*SPATIAL_W+x]-(y>0?(int)vg[(y-1)*SPATIAL_W+x]:0);
        int d=dh+dv;if(d<0){neg+=d;ny+=y;nc2++;}}
    *ns=(int16_t)(neg<-32767?-32767:neg);*cy=nc2>0?(int16_t)(ny/nc2):-1;}

static void gdiv(const int8_t*hg,const int8_t*vg,int gr,int gc,int16_t*o){
    int nr=gr*gc,reg[MAX_REGIONS];memset(reg,0,nr*sizeof(int));
    for(int y=0;y<SPATIAL_H;y++)for(int x=0;x<SPATIAL_W;x++){
        int dh=(int)hg[y*SPATIAL_W+x]-(x>0?(int)hg[y*SPATIAL_W+x-1]:0);
        int dv=(int)vg[y*SPATIAL_W+x]-(y>0?(int)vg[(y-1)*SPATIAL_W+x]:0);
        int d=dh+dv;if(d<0){int ry=y*gr/SPATIAL_H,rx=x*gc/SPATIAL_W;
            if(ry>=gr)ry=gr-1;if(rx>=gc)rx=gc-1;reg[ry*gc+rx]+=d;}}
    for(int i=0;i<nr;i++)o[i]=(int16_t)(reg[i]<-32767?-32767:reg[i]);}

/* Safe dot */
static int32_t tdot(const int8_t*a,const int8_t*b,int len){
    int32_t tot=0;for(int ch=0;ch<len;ch+=64){__m256i ac=_mm256_setzero_si256();int e=ch+64;if(e>len)e=len;
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
static int topk(const uint32_t*v,int n,cand_t*o,int k){
    uint32_t mx=0;for(int i=0;i<n;i++)if(v[i]>mx)mx=v[i];if(!mx)return 0;
    if((size_t)(mx+1)>ghcap){ghcap=(size_t)(mx+1)+4096;free(ghist);ghist=malloc(ghcap*sizeof(int));}
    memset(ghist,0,(mx+1)*sizeof(int));for(int i=0;i<n;i++)if(v[i])ghist[v[i]]++;
    int cu=0,th;for(th=(int)mx;th>=1;th--){cu+=ghist[th];if(cu>=k)break;}if(th<1)th=1;
    int nc=0;for(int i=0;i<n&&nc<k;i++)if(v[i]>=(uint32_t)th){o[nc]=(cand_t){0};o[nc].id=i;o[nc].votes=v[i];nc++;}
    qsort(o,(size_t)nc,sizeof(cand_t),cmpv);return nc;}

/* Bayesian posterior: compute log P(class|query) for all classes */
static void bayesian_posterior(const uint8_t *sig, double *log_post) {
    for(int c=0;c<N_CLASSES;c++) log_post[c]=0.0;
    for(int k=0;k<N_BLOCKS;k++){
        uint8_t bv=sig[k]; if(bv==bg_val) continue;
        const uint32_t *h=joint_hot+(size_t)k*BYTE_VALS*CLS_PAD+(size_t)bv*CLS_PAD;
        for(int c=0;c<N_CLASSES;c++) log_post[c]+=log(h[c]+0.5);
    }
    /* Normalize to prevent overflow when converting to int */
    double mx=-1e30;for(int c=0;c<N_CLASSES;c++)if(log_post[c]>mx)mx=log_post[c];
    for(int c=0;c<N_CLASSES;c++) log_post[c]-=mx;
}

/* MAD for Kalman */
static int cmp_i16(const void*a,const void*b){return *(const int16_t*)a-*(const int16_t*)b;}
static int compute_mad(const cand_t*c,int nc){if(nc<3)return 1000;
    int16_t vals[TOP_K];for(int j=0;j<nc;j++)vals[j]=divneg_tr[c[j].id];
    qsort(vals,(size_t)nc,sizeof(int16_t),cmp_i16);int16_t med=vals[nc/2],devs[TOP_K];
    for(int j=0;j<nc;j++)devs[j]=(int16_t)abs(vals[j]-med);
    qsort(devs,(size_t)nc,sizeof(int16_t),cmp_i16);return devs[nc/2]>0?devs[nc/2]:1;}

/* ================================================================ */

static void load_data(void){
    raw_train=malloc((size_t)TRAIN_N*PIXELS);raw_test=malloc((size_t)TEST_N*PIXELS);
    raw_gray_tr=malloc((size_t)TRAIN_N*GRAY_PIXELS);raw_gray_te=malloc((size_t)TEST_N*GRAY_PIXELS);
    raw_r_tr=malloc((size_t)TRAIN_N*GRAY_PIXELS);raw_r_te=malloc((size_t)TEST_N*GRAY_PIXELS);
    raw_g_tr=malloc((size_t)TRAIN_N*GRAY_PIXELS);raw_g_te=malloc((size_t)TEST_N*GRAY_PIXELS);
    raw_b_tr=malloc((size_t)TRAIN_N*GRAY_PIXELS);raw_b_te=malloc((size_t)TEST_N*GRAY_PIXELS);
    train_labels=malloc(TRAIN_N);test_labels=malloc(TEST_N);
    char p[512];uint8_t rec[3073];
    for(int b2=1;b2<=5;b2++){snprintf(p,sizeof(p),"%sdata_batch_%d.bin",data_dir,b2);
        FILE*f=fopen(p,"rb");for(int i=0;i<10000;i++){if(fread(rec,1,3073,f)!=3073){fclose(f);exit(1);}
            int idx=(b2-1)*10000+i;train_labels[idx]=rec[0];
            uint8_t*d=raw_train+(size_t)idx*PIXELS;const uint8_t*r=rec+1,*g=rec+1+1024,*b3=rec+1+2048;
            for(int y=0;y<32;y++)for(int x=0;x<32;x++){int si=y*32+x,di=y*96+x*3;d[di]=r[si];d[di+1]=g[si];d[di+2]=b3[si];}
            uint8_t*gd=raw_gray_tr+(size_t)idx*GRAY_PIXELS;
            for(int p2=0;p2<1024;p2++)gd[p2]=(uint8_t)((77*(int)r[p2]+150*(int)g[p2]+29*(int)b3[p2])>>8);
            memcpy(raw_r_tr+(size_t)idx*GRAY_PIXELS,r,1024);
            memcpy(raw_g_tr+(size_t)idx*GRAY_PIXELS,g,1024);
            memcpy(raw_b_tr+(size_t)idx*GRAY_PIXELS,b3,1024);}fclose(f);}
    snprintf(p,sizeof(p),"%stest_batch.bin",data_dir);
    FILE*f=fopen(p,"rb");for(int i=0;i<10000;i++){if(fread(rec,1,3073,f)!=3073){fclose(f);exit(1);}
        test_labels[i]=rec[0];uint8_t*d=raw_test+(size_t)i*PIXELS;const uint8_t*r=rec+1,*g=rec+1+1024,*b3=rec+1+2048;
        for(int y=0;y<32;y++)for(int x=0;x<32;x++){int si=y*32+x,di=y*96+x*3;d[di]=r[si];d[di+1]=g[si];d[di+2]=b3[si];}
        uint8_t*gd=raw_gray_te+(size_t)i*GRAY_PIXELS;
        for(int p2=0;p2<1024;p2++)gd[p2]=(uint8_t)((77*(int)r[p2]+150*(int)g[p2]+29*(int)b3[p2])>>8);
        memcpy(raw_r_te+(size_t)i*GRAY_PIXELS,r,1024);
        memcpy(raw_g_te+(size_t)i*GRAY_PIXELS,g,1024);
        memcpy(raw_b_te+(size_t)i*GRAY_PIXELS,b3,1024);}fclose(f);}

int main(int argc,char**argv){
    double t0=now_sec();
    if(argc>1){data_dir=argv[1];size_t l=strlen(data_dir);
        if(l&&data_dir[l-1]!='/'){char*b2=malloc(l+2);memcpy(b2,data_dir,l);b2[l]='/';b2[l+1]=0;data_dir=b2;}}
    printf("=== SSTT CIFAR-10: Full Stacked Pipeline ===\n\n");
    load_data();printf("  Loaded\n");

    /* MT4 */
    t3_tr=aligned_alloc(32,(size_t)TRAIN_N*PADDED);t3_te=aligned_alloc(32,(size_t)TEST_N*PADDED);
    t2_tr=aligned_alloc(32,(size_t)TRAIN_N*PADDED);t2_te=aligned_alloc(32,(size_t)TEST_N*PADDED);
    t1_tr=aligned_alloc(32,(size_t)TRAIN_N*PADDED);t1_te=aligned_alloc(32,(size_t)TEST_N*PADDED);
    t0_tr=aligned_alloc(32,(size_t)TRAIN_N*PADDED);t0_te=aligned_alloc(32,(size_t)TEST_N*PADDED);
    mt4_quant(raw_train,TRAIN_N,t3_tr,t2_tr,t1_tr,t0_tr);mt4_quant(raw_test,TEST_N,t3_te,t2_te,t1_te,t0_te);

    /* MT4 gradients (coarsest + finest — bracket the resolution) */
    hg3_tr=aligned_alloc(32,(size_t)TRAIN_N*PADDED);hg3_te=aligned_alloc(32,(size_t)TEST_N*PADDED);
    vg3_tr=aligned_alloc(32,(size_t)TRAIN_N*PADDED);vg3_te=aligned_alloc(32,(size_t)TEST_N*PADDED);
    hg0_tr=aligned_alloc(32,(size_t)TRAIN_N*PADDED);hg0_te=aligned_alloc(32,(size_t)TEST_N*PADDED);
    vg0_tr=aligned_alloc(32,(size_t)TRAIN_N*PADDED);vg0_te=aligned_alloc(32,(size_t)TEST_N*PADDED);
    grads(t3_tr,hg3_tr,vg3_tr,TRAIN_N,IMG_W,IMG_H,PADDED);grads(t3_te,hg3_te,vg3_te,TEST_N,IMG_W,IMG_H,PADDED);
    grads(t0_tr,hg0_tr,vg0_tr,TRAIN_N,IMG_W,IMG_H,PADDED);grads(t0_te,hg0_te,vg0_te,TEST_N,IMG_W,IMG_H,PADDED);

    /* Ternary for voting */
    tern_tr=aligned_alloc(32,(size_t)TRAIN_N*PADDED);tern_te=aligned_alloc(32,(size_t)TEST_N*PADDED);
    hgc_tr=aligned_alloc(32,(size_t)TRAIN_N*PADDED);hgc_te=aligned_alloc(32,(size_t)TEST_N*PADDED);
    vgc_tr=aligned_alloc(32,(size_t)TRAIN_N*PADDED);vgc_te=aligned_alloc(32,(size_t)TEST_N*PADDED);
    quant_tern(raw_train,tern_tr,TRAIN_N,PIXELS,PADDED);quant_tern(raw_test,tern_te,TEST_N,PIXELS,PADDED);
    grads(tern_tr,hgc_tr,vgc_tr,TRAIN_N,IMG_W,IMG_H,PADDED);grads(tern_te,hgc_te,vgc_te,TEST_N,IMG_W,IMG_H,PADDED);

    /* Grayscale for topo */
    int8_t *gt_tr=aligned_alloc(32,(size_t)TRAIN_N*GRAY_PADDED),*gt_te=aligned_alloc(32,(size_t)TEST_N*GRAY_PADDED);
    ghg_tr=aligned_alloc(32,(size_t)TRAIN_N*GRAY_PADDED);ghg_te=aligned_alloc(32,(size_t)TEST_N*GRAY_PADDED);
    gvg_tr=aligned_alloc(32,(size_t)TRAIN_N*GRAY_PADDED);gvg_te=aligned_alloc(32,(size_t)TEST_N*GRAY_PADDED);
    quant_tern(raw_gray_tr,gt_tr,TRAIN_N,GRAY_PIXELS,GRAY_PADDED);quant_tern(raw_gray_te,gt_te,TEST_N,GRAY_PIXELS,GRAY_PADDED);
    grads(gt_tr,ghg_tr,gvg_tr,TRAIN_N,SPATIAL_W,SPATIAL_H,GRAY_PADDED);
    grads(gt_te,ghg_te,gvg_te,TEST_N,SPATIAL_W,SPATIAL_H,GRAY_PADDED);

    /* Topo features */
    cent_tr=malloc((size_t)TRAIN_N*2);cent_te=malloc((size_t)TEST_N*2);
    for(int i=0;i<TRAIN_N;i++)cent_tr[i]=enclosed_centroid(gt_tr+(size_t)i*GRAY_PADDED);
    for(int i=0;i<TEST_N;i++)cent_te[i]=enclosed_centroid(gt_te+(size_t)i*GRAY_PADDED);
    divneg_tr=malloc((size_t)TRAIN_N*2);divneg_te=malloc((size_t)TEST_N*2);
    divcy_tr=malloc((size_t)TRAIN_N*2);divcy_te=malloc((size_t)TEST_N*2);
    for(int i=0;i<TRAIN_N;i++)divf(ghg_tr+(size_t)i*GRAY_PADDED,gvg_tr+(size_t)i*GRAY_PADDED,&divneg_tr[i],&divcy_tr[i]);
    for(int i=0;i<TEST_N;i++)divf(ghg_te+(size_t)i*GRAY_PADDED,gvg_te+(size_t)i*GRAY_PADDED,&divneg_te[i],&divcy_te[i]);
    gdiv_tr=malloc((size_t)TRAIN_N*MAX_REGIONS*2);gdiv_te=malloc((size_t)TEST_N*MAX_REGIONS*2);
    for(int i=0;i<TRAIN_N;i++)gdiv(ghg_tr+(size_t)i*GRAY_PADDED,gvg_tr+(size_t)i*GRAY_PADDED,3,3,gdiv_tr+(size_t)i*MAX_REGIONS);
    for(int i=0;i<TEST_N;i++)gdiv(ghg_te+(size_t)i*GRAY_PADDED,gvg_te+(size_t)i*GRAY_PADDED,3,3,gdiv_te+(size_t)i*MAX_REGIONS);
    free(gt_tr);free(gt_te);

    /* Per-RGB-channel divergence (Green's theorem on each color channel) */
    printf("  Computing per-RGB divergence...\n");
    {
        /* R channel */
        int8_t *rt_tr=aligned_alloc(32,(size_t)TRAIN_N*GRAY_PADDED),*rt_te=aligned_alloc(32,(size_t)TEST_N*GRAY_PADDED);
        r_hg_tr=aligned_alloc(32,(size_t)TRAIN_N*GRAY_PADDED);r_hg_te=aligned_alloc(32,(size_t)TEST_N*GRAY_PADDED);
        r_vg_tr=aligned_alloc(32,(size_t)TRAIN_N*GRAY_PADDED);r_vg_te=aligned_alloc(32,(size_t)TEST_N*GRAY_PADDED);
        quant_tern(raw_r_tr,rt_tr,TRAIN_N,GRAY_PIXELS,GRAY_PADDED);quant_tern(raw_r_te,rt_te,TEST_N,GRAY_PIXELS,GRAY_PADDED);
        grads(rt_tr,r_hg_tr,r_vg_tr,TRAIN_N,SPATIAL_W,SPATIAL_H,GRAY_PADDED);
        grads(rt_te,r_hg_te,r_vg_te,TEST_N,SPATIAL_W,SPATIAL_H,GRAY_PADDED);
        rdiv_tr=malloc((size_t)TRAIN_N*2);rdiv_te=malloc((size_t)TEST_N*2);
        int16_t *rcy_tr=malloc((size_t)TRAIN_N*2),*rcy_te=malloc((size_t)TEST_N*2);
        for(int i=0;i<TRAIN_N;i++)divf(r_hg_tr+(size_t)i*GRAY_PADDED,r_vg_tr+(size_t)i*GRAY_PADDED,&rdiv_tr[i],&rcy_tr[i]);
        for(int i=0;i<TEST_N;i++)divf(r_hg_te+(size_t)i*GRAY_PADDED,r_vg_te+(size_t)i*GRAY_PADDED,&rdiv_te[i],&rcy_te[i]);
        gdivr_tr=malloc((size_t)TRAIN_N*MAX_REGIONS*2);gdivr_te=malloc((size_t)TEST_N*MAX_REGIONS*2);
        for(int i=0;i<TRAIN_N;i++)gdiv(r_hg_tr+(size_t)i*GRAY_PADDED,r_vg_tr+(size_t)i*GRAY_PADDED,3,3,gdivr_tr+(size_t)i*MAX_REGIONS);
        for(int i=0;i<TEST_N;i++)gdiv(r_hg_te+(size_t)i*GRAY_PADDED,r_vg_te+(size_t)i*GRAY_PADDED,3,3,gdivr_te+(size_t)i*MAX_REGIONS);
        free(rt_tr);free(rt_te);free(rcy_tr);free(rcy_te);

        /* G channel */
        int8_t *gt2_tr=aligned_alloc(32,(size_t)TRAIN_N*GRAY_PADDED),*gt2_te=aligned_alloc(32,(size_t)TEST_N*GRAY_PADDED);
        g_hg_tr=aligned_alloc(32,(size_t)TRAIN_N*GRAY_PADDED);g_hg_te=aligned_alloc(32,(size_t)TEST_N*GRAY_PADDED);
        g_vg_tr=aligned_alloc(32,(size_t)TRAIN_N*GRAY_PADDED);g_vg_te=aligned_alloc(32,(size_t)TEST_N*GRAY_PADDED);
        quant_tern(raw_g_tr,gt2_tr,TRAIN_N,GRAY_PIXELS,GRAY_PADDED);quant_tern(raw_g_te,gt2_te,TEST_N,GRAY_PIXELS,GRAY_PADDED);
        grads(gt2_tr,g_hg_tr,g_vg_tr,TRAIN_N,SPATIAL_W,SPATIAL_H,GRAY_PADDED);
        grads(gt2_te,g_hg_te,g_vg_te,TEST_N,SPATIAL_W,SPATIAL_H,GRAY_PADDED);
        gdiv_g_tr=malloc((size_t)TRAIN_N*2);gdiv_g_te=malloc((size_t)TEST_N*2);
        int16_t *gcy_tr=malloc((size_t)TRAIN_N*2),*gcy_te=malloc((size_t)TEST_N*2);
        for(int i=0;i<TRAIN_N;i++)divf(g_hg_tr+(size_t)i*GRAY_PADDED,g_vg_tr+(size_t)i*GRAY_PADDED,&gdiv_g_tr[i],&gcy_tr[i]);
        for(int i=0;i<TEST_N;i++)divf(g_hg_te+(size_t)i*GRAY_PADDED,g_vg_te+(size_t)i*GRAY_PADDED,&gdiv_g_te[i],&gcy_te[i]);
        gdivg_tr=malloc((size_t)TRAIN_N*MAX_REGIONS*2);gdivg_te=malloc((size_t)TEST_N*MAX_REGIONS*2);
        for(int i=0;i<TRAIN_N;i++)gdiv(g_hg_tr+(size_t)i*GRAY_PADDED,g_vg_tr+(size_t)i*GRAY_PADDED,3,3,gdivg_tr+(size_t)i*MAX_REGIONS);
        for(int i=0;i<TEST_N;i++)gdiv(g_hg_te+(size_t)i*GRAY_PADDED,g_vg_te+(size_t)i*GRAY_PADDED,3,3,gdivg_te+(size_t)i*MAX_REGIONS);
        free(gt2_tr);free(gt2_te);free(gcy_tr);free(gcy_te);

        /* B channel */
        int8_t *bt_tr=aligned_alloc(32,(size_t)TRAIN_N*GRAY_PADDED),*bt_te=aligned_alloc(32,(size_t)TEST_N*GRAY_PADDED);
        b_hg_tr=aligned_alloc(32,(size_t)TRAIN_N*GRAY_PADDED);b_hg_te=aligned_alloc(32,(size_t)TEST_N*GRAY_PADDED);
        b_vg_tr=aligned_alloc(32,(size_t)TRAIN_N*GRAY_PADDED);b_vg_te=aligned_alloc(32,(size_t)TEST_N*GRAY_PADDED);
        quant_tern(raw_b_tr,bt_tr,TRAIN_N,GRAY_PIXELS,GRAY_PADDED);quant_tern(raw_b_te,bt_te,TEST_N,GRAY_PIXELS,GRAY_PADDED);
        grads(bt_tr,b_hg_tr,b_vg_tr,TRAIN_N,SPATIAL_W,SPATIAL_H,GRAY_PADDED);
        grads(bt_te,b_hg_te,b_vg_te,TEST_N,SPATIAL_W,SPATIAL_H,GRAY_PADDED);
        bdiv_tr=malloc((size_t)TRAIN_N*2);bdiv_te=malloc((size_t)TEST_N*2);
        int16_t *bcy_tr=malloc((size_t)TRAIN_N*2),*bcy_te=malloc((size_t)TEST_N*2);
        for(int i=0;i<TRAIN_N;i++)divf(b_hg_tr+(size_t)i*GRAY_PADDED,b_vg_tr+(size_t)i*GRAY_PADDED,&bdiv_tr[i],&bcy_tr[i]);
        for(int i=0;i<TEST_N;i++)divf(b_hg_te+(size_t)i*GRAY_PADDED,b_vg_te+(size_t)i*GRAY_PADDED,&bdiv_te[i],&bcy_te[i]);
        gdivb_tr=malloc((size_t)TRAIN_N*MAX_REGIONS*2);gdivb_te=malloc((size_t)TEST_N*MAX_REGIONS*2);
        for(int i=0;i<TRAIN_N;i++)gdiv(b_hg_tr+(size_t)i*GRAY_PADDED,b_vg_tr+(size_t)i*GRAY_PADDED,3,3,gdivb_tr+(size_t)i*MAX_REGIONS);
        for(int i=0;i<TEST_N;i++)gdiv(b_hg_te+(size_t)i*GRAY_PADDED,b_vg_te+(size_t)i*GRAY_PADDED,3,3,gdivb_te+(size_t)i*MAX_REGIONS);
        free(bt_tr);free(bt_te);free(bcy_tr);free(bcy_te);
    }

    /* Voting infra */
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

    {long vc[BYTE_VALS]={0};for(int i=0;i<TRAIN_N;i++){const uint8_t*sig=joint_tr+(size_t)i*SIG_PAD;
        for(int k=0;k<N_BLOCKS;k++)vc[sig[k]]++;}bg_val=0;long mc=0;
        for(int v=0;v<BYTE_VALS;v++)if(vc[v]>mc){mc=vc[v];bg_val=(uint8_t)v;}}
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
    for(int v=0;v<BYTE_VALS;v++)for(int b3=0;b3<8;b3++)nbr[v][b3]=(uint8_t)(v^(1<<b3));
    memset(idx_sz,0,sizeof(idx_sz));
    for(int i=0;i<TRAIN_N;i++){const uint8_t*sig=joint_tr+(size_t)i*SIG_PAD;
        for(int k=0;k<N_BLOCKS;k++)if(sig[k]!=bg_val)idx_sz[k][sig[k]]++;}
    uint32_t tot=0;for(int k=0;k<N_BLOCKS;k++)for(int v=0;v<BYTE_VALS;v++){idx_off[k][v]=tot;tot+=idx_sz[k][v];}
    idx_pool=malloc((size_t)tot*4);
    uint32_t(*wp)[BYTE_VALS]=malloc((size_t)N_BLOCKS*BYTE_VALS*4);memcpy(wp,idx_off,(size_t)N_BLOCKS*BYTE_VALS*4);
    for(int i=0;i<TRAIN_N;i++){const uint8_t*sig=joint_tr+(size_t)i*SIG_PAD;
        for(int k=0;k<N_BLOCKS;k++)if(sig[k]!=bg_val)idx_pool[wp[k][sig[k]]++]=(uint32_t)i;}free(wp);

    printf("  Built (%.1f sec)\n\n",now_sec()-t0);

    /* ================================================================
     * RUN: Full stacked pipeline with weight sweep
     * ================================================================ */
    printf("Running stacked pipeline...\n\n");
    uint32_t *vbuf=calloc(TRAIN_N,4);cand_t *cands=malloc(TOP_K*sizeof(cand_t));
    int nreg=9;

    /* Precompute Bayesian posterior + candidate features */
    /* Weight sweep: mt4_dot, topo, bayesian prior */
    int w_dot[]={512};        /* MT4 px+grad dot (best from prior sweep) */
    int w_div[]={0,200};      /* grayscale divergence */
    int w_gdv[]={0,200};      /* grayscale grid div */
    int w_cen[]={0};           /* centroid (was 0 in best) */
    int w_bay[]={0,50,100};   /* bayesian prior */
    int w_rgb[]={0,100,200};  /* per-RGB-channel divergence (scalar) */
    int w_rgdv[]={0,100,200}; /* per-RGB grid divergence */

    int best_acc=0;
    int bwd=0,bwg=0,bwc=0,bwb=0,bsc=0,bwdot=0,bwrgb=0,bwrgdv=0;

    /* First: precompute candidate features */
    typedef struct{int32_t mt4_dot;int32_t div_sim;int32_t cent_sim;int32_t gdiv_sim;
                   int32_t rdiv_sim;int32_t gdivr_sim;  /* R channel div + grid */
                   int32_t gdiv_sim_g;int32_t gdivg_sim; /* G channel */
                   int32_t bdiv_sim;int32_t gdivb_sim;  /* B channel */
                  }feat_t;
    feat_t *feats=malloc((size_t)TEST_N*TOP_K*sizeof(feat_t));
    int *nc_arr=malloc(TEST_N*sizeof(int));
    uint32_t *cand_ids=malloc((size_t)TEST_N*TOP_K*sizeof(uint32_t));
    double *bay_post=malloc((size_t)TEST_N*N_CLASSES*sizeof(double));

    printf("  Precomputing...\n");
    for(int i=0;i<TEST_N;i++){
        /* Vote */
        memset(vbuf,0,TRAIN_N*4);const uint8_t*sig=joint_te+(size_t)i*SIG_PAD;
        for(int k=0;k<N_BLOCKS;k++){uint8_t bv=sig[k];if(bv==bg_val)continue;
            uint16_t w=ig_w[k],wh=w>1?w/2:1;
            {uint32_t off=idx_off[k][bv];uint16_t sz=idx_sz[k][bv];for(uint16_t j=0;j<sz;j++)vbuf[idx_pool[off+j]]+=w;}
            for(int nb=0;nb<8;nb++){uint8_t nv=nbr[bv][nb];if(nv==bg_val)continue;
                uint32_t no=idx_off[k][nv];uint16_t ns=idx_sz[k][nv];for(uint16_t j=0;j<ns;j++)vbuf[idx_pool[no+j]]+=wh;}}

        int nc=topk(vbuf,TRAIN_N,cands,TOP_K);nc_arr[i]=nc;

        /* Bayesian posterior */
        bayesian_posterior(sig, bay_post+(size_t)i*N_CLASSES);

        /* Per-candidate features */
        int16_t q_dn=divneg_te[i],q_dc=divcy_te[i],q_cent=cent_te[i];
        const int16_t *q_gd=gdiv_te+(size_t)i*MAX_REGIONS;
        for(int j=0;j<nc;j++){
            uint32_t id=cands[j].id;
            cand_ids[i*TOP_K+j]=id;
            feat_t *ft=&feats[i*TOP_K+j];
            /* MT4 pixel+grad dot */
            ft->mt4_dot= 27*(tdot(t3_te+(size_t)i*PADDED,t3_tr+(size_t)id*PADDED,PADDED)
                             +tdot(hg3_te+(size_t)i*PADDED,hg3_tr+(size_t)id*PADDED,PADDED)
                             +tdot(vg3_te+(size_t)i*PADDED,vg3_tr+(size_t)id*PADDED,PADDED))
                         +   (tdot(t0_te+(size_t)i*PADDED,t0_tr+(size_t)id*PADDED,PADDED)
                             +tdot(hg0_te+(size_t)i*PADDED,hg0_tr+(size_t)id*PADDED,PADDED)
                             +tdot(vg0_te+(size_t)i*PADDED,vg0_tr+(size_t)id*PADDED,PADDED));
            /* Divergence */
            int16_t cdn=divneg_tr[id],cdc=divcy_tr[id];
            int32_t ds=-(int32_t)abs(q_dn-cdn);
            if(q_dc>=0&&cdc>=0)ds-=(int32_t)abs(q_dc-cdc)*2;
            else if((q_dc<0)!=(cdc<0))ds-=10;
            ft->div_sim=ds;
            /* Centroid */
            int16_t cc2=cent_tr[id];
            ft->cent_sim=(q_cent<0&&cc2<0)?CENT_BOTH_OPEN:(q_cent<0||cc2<0)?-CENT_DISAGREE:-(int32_t)abs(q_cent-cc2);
            /* Grid divergence (grayscale) */
            const int16_t *cg=gdiv_tr+(size_t)id*MAX_REGIONS;
            int32_t gl=0;for(int r=0;r<nreg;r++)gl+=abs(q_gd[r]-cg[r]);ft->gdiv_sim=-gl;
            /* Per-RGB-channel divergence */
            ft->rdiv_sim=-(int32_t)abs(rdiv_te[i]-rdiv_tr[id]);
            ft->bdiv_sim=-(int32_t)abs(bdiv_te[i]-bdiv_tr[id]);
            ft->gdiv_sim_g=-(int32_t)abs(gdiv_g_te[i]-gdiv_g_tr[id]);
            /* Per-RGB grid divergence */
            {const int16_t *cr=gdivr_tr+(size_t)id*MAX_REGIONS,*qr=gdivr_te+(size_t)i*MAX_REGIONS;
             int32_t l=0;for(int r=0;r<nreg;r++)l+=abs(qr[r]-cr[r]);ft->gdivr_sim=-l;}
            {const int16_t *cgg=gdivg_tr+(size_t)id*MAX_REGIONS,*qg=gdivg_te+(size_t)i*MAX_REGIONS;
             int32_t l=0;for(int r=0;r<nreg;r++)l+=abs(qg[r]-cgg[r]);ft->gdivg_sim=-l;}
            {const int16_t *cb=gdivb_tr+(size_t)id*MAX_REGIONS,*qb=gdivb_te+(size_t)i*MAX_REGIONS;
             int32_t l=0;for(int r=0;r<nreg;r++)l+=abs(qb[r]-cb[r]);ft->gdivb_sim=-l;}
        }
        if((i+1)%2000==0)fprintf(stderr,"  %d/%d\r",i+1,TEST_N);
    }
    fprintf(stderr,"\n");free(vbuf);

    printf("  Sweeping weights...\n");
    int sweep_count=0;
    for(int doti=0;doti<1;doti++)
    for(int di=0;di<2;di++)for(int gi=0;gi<2;gi++)for(int ci=0;ci<1;ci++)
    for(int bi=0;bi<3;bi++)for(int ri=0;ri<3;ri++)for(int rgi=0;rgi<3;rgi++){
        int correct=0;
        for(int i=0;i<TEST_N;i++){
            int nc=nc_arr[i];
            cand_t cs[TOP_K];
            const double *bp=bay_post+(size_t)i*N_CLASSES;
            for(int j=0;j<nc;j++){
                cs[j].id=cand_ids[i*TOP_K+j];cs[j].votes=0;
                const feat_t *ft=&feats[i*TOP_K+j];
                uint8_t lbl=train_labels[cs[j].id];
                int64_t score=(int64_t)w_dot[doti]*ft->mt4_dot
                             +(int64_t)w_div[di]*ft->div_sim
                             +(int64_t)w_gdv[gi]*ft->gdiv_sim
                             +(int64_t)w_cen[ci]*ft->cent_sim
                             +(int64_t)w_bay[bi]*(int64_t)(bp[lbl]*1000)
                             +(int64_t)w_rgb[ri]*(ft->rdiv_sim+ft->gdiv_sim_g+ft->bdiv_sim)
                             +(int64_t)w_rgdv[rgi]*(ft->gdivr_sim+ft->gdivg_sim+ft->gdivb_sim);
                cs[j].score=score;
            }
            qsort(cs,(size_t)nc,sizeof(cand_t),cmps);
            if(nc>0&&train_labels[cs[0].id]==test_labels[i])correct++;
        }
        sweep_count++;
        if(correct>best_acc){best_acc=correct;bwdot=w_dot[doti];bwd=w_div[di];bwg=w_gdv[gi];
            bwc=w_cen[ci];bwb=w_bay[bi];bwrgb=w_rgb[ri];bwrgdv=w_rgdv[rgi];}
    }
    printf("  Swept %d configurations\n",sweep_count);

    printf("\n=== RESULTS ===\n\n");
    printf("  Best: dot=%d div=%d gdiv=%d cent=%d bayes=%d rgb_div=%d rgb_gdiv=%d\n",
           bwdot,bwd,bwg,bwc,bwb,bwrgb,bwrgdv);
    printf("  k=1 accuracy: %.2f%% (%d/%d)\n\n",100.0*best_acc/TEST_N,best_acc,TEST_N);

    /* Also test k=1,3,5,7 with best weights */
    printf("  kNN sweep with best weights:\n");
    for(int kk=1;kk<=7;kk+=2){
        int correct=0;
        for(int i=0;i<TEST_N;i++){
            int nc=nc_arr[i];cand_t cs[TOP_K];
            const double *bp=bay_post+(size_t)i*N_CLASSES;
            for(int j=0;j<nc;j++){cs[j].id=cand_ids[i*TOP_K+j];
                const feat_t *ft=&feats[i*TOP_K+j];uint8_t lbl=train_labels[cs[j].id];
                cs[j].score=(int64_t)bwdot*ft->mt4_dot+(int64_t)bwd*ft->div_sim
                           +(int64_t)bwg*ft->gdiv_sim+(int64_t)bwc*ft->cent_sim
                           +(int64_t)bwb*(int64_t)(bp[lbl]*1000)
                           +(int64_t)bwrgb*(ft->rdiv_sim+ft->gdiv_sim_g+ft->bdiv_sim)
                           +(int64_t)bwrgdv*(ft->gdivr_sim+ft->gdivg_sim+ft->gdivb_sim);}
            qsort(cs,(size_t)nc,sizeof(cand_t),cmps);
            int v[N_CLASSES]={0};int ek=kk<nc?kk:nc;
            for(int j=0;j<ek;j++)v[train_labels[cs[j].id]]++;
            int pred=0;for(int c=1;c<N_CLASSES;c++)if(v[c]>v[pred])pred=c;
            if(pred==test_labels[i])correct++;
        }
        printf("    k=%d: %.2f%%\n",kk,100.0*correct/TEST_N);
    }

    /* Baselines for comparison */
    printf("\n  Baselines:\n");
    printf("    Bayesian (no cascade):    36.58%%\n");
    printf("    MT4 px+grad k=1:          36.42%%\n");
    printf("    Ternary cascade k=7:      33.18%%\n");
    printf("    Full topo k=7:            33.94%%\n");

    /* Per-class with best config */
    printf("\n  Per-class recall (best config, k=1):\n");
    {int pc[N_CLASSES]={0},pt[N_CLASSES]={0};
     for(int i=0;i<TEST_N;i++){pt[test_labels[i]]++;int nc=nc_arr[i];cand_t cs[TOP_K];
         const double *bp=bay_post+(size_t)i*N_CLASSES;
         for(int j=0;j<nc;j++){cs[j].id=cand_ids[i*TOP_K+j];const feat_t *ft=&feats[i*TOP_K+j];
             uint8_t lbl=train_labels[cs[j].id];
             cs[j].score=(int64_t)bwdot*ft->mt4_dot+(int64_t)bwd*ft->div_sim
                        +(int64_t)bwg*ft->gdiv_sim+(int64_t)bwc*ft->cent_sim
                        +(int64_t)bwb*(int64_t)(bp[lbl]*1000);}
         qsort(cs,(size_t)nc,sizeof(cand_t),cmps);
         if(nc>0&&train_labels[cs[0].id]==test_labels[i])pc[test_labels[i]]++;}
     for(int c=0;c<N_CLASSES;c++)printf("    %d %-12s %.1f%%\n",c,cn[c],100.0*pc[c]/pt[c]);}

    printf("\nTotal: %.1f sec\n",now_sec()-t0);
    free(feats);free(nc_arr);free(cand_ids);free(bay_post);free(cands);
    return 0;
}
