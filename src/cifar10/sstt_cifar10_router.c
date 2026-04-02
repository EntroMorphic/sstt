/*
 * sstt_cifar10_router.c — Meta-System Router: Per-Image System Selection
 *
 * We have 6+ classifiers, each excelling on different images.
 * Oracle ceiling: ~58% (per-class best across all systems).
 * Current single-system best: 50.18%.
 *
 * The router: for each test image, predict WHICH system will succeed
 * using the training set's success/failure patterns. Route to that system.
 *
 * Method: for each training image, pre-classify with all systems,
 * record which succeed. For each test image, find its nearest training
 * neighbors (via Gauss map), check which system succeeded on those
 * neighbors, route to the system with the highest success rate among
 * similar images.
 *
 * Zero learned parameters. The router is a conditional frequency lookup.
 *
 * Build: gcc -O3 -mavx2 -march=native -o sstt_cifar10_router src/sstt_cifar10_router.c -lm
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
#define N_CLASSES 10
#define CLS_PAD 16
#define SP_W 32
#define SP_H 32
#define SP_PX 1024
#define IMG_W 96
#define IMG_H 32
#define PIXELS 3072
#define PADDED 3072
#define BLKS_PER_ROW 32
#define N_BLOCKS 1024
#define SIG_PAD 1024
#define BYTE_VALS 256
#define BG_TRANS 13
#define H_TRANS_PER_ROW 31
#define N_HTRANS (H_TRANS_PER_ROW*IMG_H)
#define V_TRANS_PER_COL 31
#define N_VTRANS (BLKS_PER_ROW*V_TRANS_PER_COL)
#define TRANS_PAD 992
#define IG_SCALE 16
#define TOP_K 200
#define GM_DIR 3
#define GM_MAG 5
#define GM_BINS (GM_DIR*GM_DIR*GM_MAG)
#define G4 4
#define G4_FEAT (G4*G4*GM_BINS)
#define NCH 4
#define GM_RGB_FEAT (NCH*G4_FEAT)

#define N_SYSTEMS 4 /* Systems to route between */

static const char *data_dir="data-cifar10/";
static const char *cn[]={"airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"};
static const char *sysnames[]={"Cascade Gauss RGB","Brute Gauss RGB","Block 3-eye Bayesian","Edge-masked Bayesian"};
static double now_sec(void){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return ts.tv_sec+ts.tv_nsec*1e-9;}
static int cmp_u8(const void*a,const void*b){return *(const uint8_t*)a-*(const uint8_t*)b;}
static inline int8_t ct(int v){return v>0?1:v<0?-1:0;}

static uint8_t *raw_train,*raw_test,*train_labels,*test_labels;
static uint8_t *raw_r_tr,*raw_r_te,*raw_g_tr,*raw_g_te,*raw_b_tr,*raw_b_te,*raw_gray_tr,*raw_gray_te;

/* ================================================================
 *  Gauss map (shared by routing + System 1 + System 2)
 * ================================================================ */
static int16_t *gm_tr, *gm_te; /* RGB 4×4 grid Gauss maps */

static void gauss_map_grid(const uint8_t *img, int16_t *hist) {
    memset(hist,0,G4_FEAT*sizeof(int16_t));
    for(int y=0;y<SP_H;y++)for(int x=0;x<SP_W;x++){
        int gx=(x<SP_W-1)?(int)img[y*SP_W+x+1]-(int)img[y*SP_W+x]:0;
        int gy=(y<SP_H-1)?(int)img[(y+1)*SP_W+x]-(int)img[y*SP_W+x]:0;
        int dx=(gx>10)?2:(gx<-10)?0:1,dy=(gy>10)?2:(gy<-10)?0:1;
        int mag=abs(gx)+abs(gy),mb=(mag<5)?0:(mag<20)?1:(mag<50)?2:(mag<100)?3:4;
        int bin=dx*GM_DIR*GM_MAG+dy*GM_MAG+mb;
        int ry=y*G4/SP_H,rx=x*G4/SP_W;if(ry>=G4)ry=G4-1;if(rx>=G4)rx=G4-1;
        hist[(ry*G4+rx)*GM_BINS+bin]++;}
}

static int32_t hist_l1(const int16_t*a,const int16_t*b,int len){int32_t d=0;for(int i=0;i<len;i++)d+=abs(a[i]-b[i]);return d;}

/* ================================================================
 *  Block-based voting (3-eye stereo)
 * ================================================================ */
typedef struct{uint8_t *joint_tr,*joint_te;uint32_t *hot;uint16_t ig[N_BLOCKS];uint8_t bg;
    uint32_t idx_off[N_BLOCKS][BYTE_VALS];uint16_t idx_sz[N_BLOCKS][BYTE_VALS];uint32_t *idx_pool;
    uint8_t nbr[BYTE_VALS][8];}eye_t;
static eye_t eyes[3];

static inline uint8_t be(int8_t a,int8_t b,int8_t c){return(uint8_t)((a+1)*9+(b+1)*3+(c+1));}
static inline uint8_t te2(uint8_t a,uint8_t b){int8_t a0=(a/9)-1,a1=((a/3)%3)-1,a2=(a%3)-1,b0=(b/9)-1,b1=((b/3)%3)-1,b2=(b%3)-1;return be(ct(b0-a0),ct(b1-a1),ct(b2-a2));}

/* Build eye from raw data using quantization function */
static void build_eye(eye_t*e,void(*qfn)(const uint8_t*,int8_t*,int)){
    int8_t *ttr=aligned_alloc(32,(size_t)TRAIN_N*PADDED),*tte=aligned_alloc(32,(size_t)TEST_N*PADDED);
    qfn(raw_train,ttr,TRAIN_N);qfn(raw_test,tte,TEST_N);
    int8_t *hgr=aligned_alloc(32,(size_t)TRAIN_N*PADDED),*vgr=aligned_alloc(32,(size_t)TRAIN_N*PADDED);
    int8_t *hge=aligned_alloc(32,(size_t)TEST_N*PADDED),*vge=aligned_alloc(32,(size_t)TEST_N*PADDED);
    #define GRAD(t,h,v,n) for(int i=0;i<n;i++){const int8_t*ti=t+(size_t)i*PADDED;int8_t*hi=h+(size_t)i*PADDED,*vi=v+(size_t)i*PADDED;\
        for(int y=0;y<IMG_H;y++){for(int x=0;x<IMG_W-1;x++)hi[y*IMG_W+x]=ct(ti[y*IMG_W+x+1]-ti[y*IMG_W+x]);hi[y*IMG_W+IMG_W-1]=0;}\
        for(int y=0;y<IMG_H-1;y++)for(int x=0;x<IMG_W;x++)vi[y*IMG_W+x]=ct(ti[(y+1)*IMG_W+x]-ti[y*IMG_W+x]);memset(vi+(IMG_H-1)*IMG_W,0,IMG_W);}
    GRAD(ttr,hgr,vgr,TRAIN_N);GRAD(tte,hge,vge,TEST_N);
    uint8_t *pxt=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD),*pxe=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    uint8_t *hst=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD),*hse=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    uint8_t *vst=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD),*vse=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    #define BSIG(src,dst,n) for(int i=0;i<n;i++){const int8_t*im=src+(size_t)i*PADDED;uint8_t*si=dst+(size_t)i*SIG_PAD;\
        for(int y=0;y<IMG_H;y++)for(int s=0;s<BLKS_PER_ROW;s++){int b2=y*IMG_W+s*3;si[y*BLKS_PER_ROW+s]=be(im[b2],im[b2+1],im[b2+2]);}}
    BSIG(ttr,pxt,TRAIN_N);BSIG(tte,pxe,TEST_N);BSIG(hgr,hst,TRAIN_N);BSIG(hge,hse,TEST_N);BSIG(vgr,vst,TRAIN_N);BSIG(vge,vse,TEST_N);
    uint8_t *htt=aligned_alloc(32,(size_t)TRAIN_N*TRANS_PAD),*hte=aligned_alloc(32,(size_t)TEST_N*TRANS_PAD);
    uint8_t *vtt=aligned_alloc(32,(size_t)TRAIN_N*TRANS_PAD),*vte=aligned_alloc(32,(size_t)TEST_N*TRANS_PAD);
    #define TRANS(bs,ht2,vt2,n) for(int i=0;i<n;i++){const uint8_t*s=bs+(size_t)i*SIG_PAD;uint8_t*hh=ht2+(size_t)i*TRANS_PAD,*vv=vt2+(size_t)i*TRANS_PAD;\
        for(int y=0;y<IMG_H;y++)for(int ss=0;ss<H_TRANS_PER_ROW;ss++)hh[y*H_TRANS_PER_ROW+ss]=te2(s[y*BLKS_PER_ROW+ss],s[y*BLKS_PER_ROW+ss+1]);\
        memset(hh+N_HTRANS,0xFF,TRANS_PAD-N_HTRANS);\
        for(int y=0;y<V_TRANS_PER_COL;y++)for(int ss=0;ss<BLKS_PER_ROW;ss++)vv[y*BLKS_PER_ROW+ss]=te2(s[y*BLKS_PER_ROW+ss],s[(y+1)*BLKS_PER_ROW+ss]);\
        memset(vv+N_VTRANS,0xFF,TRANS_PAD-N_VTRANS);}
    TRANS(pxt,htt,vtt,TRAIN_N);TRANS(pxe,hte,vte,TEST_N);
    e->joint_tr=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD);e->joint_te=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    #define ENCD(pi,hi,vi,hti,vti,oi,n) for(int i=0;i<n;i++){const uint8_t*_p=pi+(size_t)i*SIG_PAD,*_h=hi+(size_t)i*SIG_PAD,*_v=vi+(size_t)i*SIG_PAD;\
        const uint8_t*_ht=hti+(size_t)i*TRANS_PAD,*_vt=vti+(size_t)i*TRANS_PAD;uint8_t*_o=oi+(size_t)i*SIG_PAD;\
        for(int y=0;y<IMG_H;y++)for(int s=0;s<BLKS_PER_ROW;s++){int k=y*BLKS_PER_ROW+s;\
            uint8_t htb=(s>0)?_ht[y*H_TRANS_PER_ROW+(s-1)]:BG_TRANS,vtb=(y>0)?_vt[(y-1)*BLKS_PER_ROW+s]:BG_TRANS;\
            int ps=((_p[k]/9)-1)+(((_p[k]/3)%3)-1)+((_p[k]%3)-1);int hs=((_h[k]/9)-1)+(((_h[k]/3)%3)-1)+((_h[k]%3)-1);\
            int vs=((_v[k]/9)-1)+(((_v[k]/3)%3)-1)+((_v[k]%3)-1);\
            uint8_t pc=ps<0?0:ps==0?1:ps<3?2:3,hc=hs<0?0:hs==0?1:hs<3?2:3,vc=vs<0?0:vs==0?1:vs<3?2:3;\
            _o[k]=pc|(hc<<2)|(vc<<4)|((htb!=BG_TRANS)?1<<6:0)|((vtb!=BG_TRANS)?1<<7:0);}}
    ENCD(pxt,hst,vst,htt,vtt,e->joint_tr,TRAIN_N);ENCD(pxe,hse,vse,hte,vte,e->joint_te,TEST_N);
    free(ttr);free(tte);free(hgr);free(vgr);free(hge);free(vge);free(pxt);free(pxe);free(hst);free(hse);free(vst);free(vse);free(htt);free(hte);free(vtt);free(vte);
    long vc[BYTE_VALS]={0};for(int i=0;i<TRAIN_N;i++){const uint8_t*sig=e->joint_tr+(size_t)i*SIG_PAD;for(int k=0;k<N_BLOCKS;k++)vc[sig[k]]++;}
    e->bg=0;long mc=0;for(int v=0;v<BYTE_VALS;v++)if(vc[v]>mc){mc=vc[v];e->bg=(uint8_t)v;}
    e->hot=aligned_alloc(32,(size_t)N_BLOCKS*BYTE_VALS*CLS_PAD*4);memset(e->hot,0,(size_t)N_BLOCKS*BYTE_VALS*CLS_PAD*4);
    for(int i=0;i<TRAIN_N;i++){int l=train_labels[i];const uint8_t*sig=e->joint_tr+(size_t)i*SIG_PAD;
        for(int k=0;k<N_BLOCKS;k++)e->hot[(size_t)k*BYTE_VALS*CLS_PAD+(size_t)sig[k]*CLS_PAD+l]++;}
    {int cc[N_CLASSES]={0};for(int i=0;i<TRAIN_N;i++)cc[train_labels[i]]++;
     double hc=0;for(int c=0;c<N_CLASSES;c++){double p=(double)cc[c]/TRAIN_N;if(p>0)hc-=p*log2(p);}
     double raw2[N_BLOCKS],mx=0;for(int k=0;k<N_BLOCKS;k++){double hcond=0;
         for(int v=0;v<BYTE_VALS;v++){if((uint8_t)v==e->bg)continue;const uint32_t*h=e->hot+(size_t)k*BYTE_VALS*CLS_PAD+(size_t)v*CLS_PAD;
             int vt=0;for(int c=0;c<N_CLASSES;c++)vt+=(int)h[c];if(!vt)continue;double pv=(double)vt/TRAIN_N,hv=0;
             for(int c=0;c<N_CLASSES;c++){double pc2=(double)h[c]/vt;if(pc2>0)hv-=pc2*log2(pc2);}hcond+=pv*hv;}
         raw2[k]=hc-hcond;if(raw2[k]>mx)mx=raw2[k];}
     for(int k=0;k<N_BLOCKS;k++){e->ig[k]=mx>0?(uint16_t)(raw2[k]/mx*IG_SCALE+0.5):1;if(!e->ig[k])e->ig[k]=1;}}
    for(int v=0;v<BYTE_VALS;v++)for(int b3=0;b3<8;b3++)e->nbr[v][b3]=(uint8_t)(v^(1<<b3));
    memset(e->idx_sz,0,sizeof(e->idx_sz));
    for(int i=0;i<TRAIN_N;i++){const uint8_t*sig=e->joint_tr+(size_t)i*SIG_PAD;for(int k=0;k<N_BLOCKS;k++)if(sig[k]!=e->bg)e->idx_sz[k][sig[k]]++;}
    uint32_t tot=0;for(int k=0;k<N_BLOCKS;k++)for(int v=0;v<BYTE_VALS;v++){e->idx_off[k][v]=tot;tot+=e->idx_sz[k][v];}
    e->idx_pool=malloc((size_t)tot*4);uint32_t(*wp)[BYTE_VALS]=malloc((size_t)N_BLOCKS*BYTE_VALS*4);memcpy(wp,e->idx_off,(size_t)N_BLOCKS*BYTE_VALS*4);
    for(int i=0;i<TRAIN_N;i++){const uint8_t*sig=e->joint_tr+(size_t)i*SIG_PAD;for(int k=0;k<N_BLOCKS;k++)if(sig[k]!=e->bg)e->idx_pool[wp[k][sig[k]]++]=(uint32_t)i;}free(wp);
}
static void vote_eye(const eye_t*e,int img,uint32_t*votes){
    const uint8_t*sig=e->joint_te+(size_t)img*SIG_PAD;
    for(int k=0;k<N_BLOCKS;k++){uint8_t bv=sig[k];if(bv==e->bg)continue;
        uint16_t w=e->ig[k],wh=w>1?w/2:1;
        {uint32_t off=e->idx_off[k][bv];uint16_t sz=e->idx_sz[k][bv];for(uint16_t j=0;j<sz;j++)votes[e->idx_pool[off+j]]+=w;}
        for(int nb=0;nb<8;nb++){uint8_t nv=e->nbr[bv][nb];if(nv==e->bg)continue;
            uint32_t noff=e->idx_off[k][nv];uint16_t nsz=e->idx_sz[k][nv];for(uint16_t j=0;j<nsz;j++)votes[e->idx_pool[noff+j]]+=wh;}}}
static void quant_fixed(const uint8_t*s,int8_t*d,int n){for(int img=0;img<n;img++){const uint8_t*si=s+(size_t)img*PIXELS;int8_t*di=d+(size_t)img*PADDED;for(int i=0;i<PIXELS;i++)di[i]=si[i]>170?1:si[i]<85?-1:0;}}
static void quant_adaptive(const uint8_t*s,int8_t*d,int n){uint8_t*sorted=malloc(PIXELS);for(int img=0;img<n;img++){const uint8_t*si=s+(size_t)img*PIXELS;int8_t*di=d+(size_t)img*PADDED;memcpy(sorted,si,PIXELS);qsort(sorted,PIXELS,1,cmp_u8);uint8_t p33=sorted[PIXELS/3],p67=sorted[2*PIXELS/3];if(p33==p67){p33=p33>0?p33-1:0;p67=p67<255?p67+1:255;}for(int i=0;i<PIXELS;i++)di[i]=si[i]>p67?1:si[i]<p33?-1:0;}free(sorted);}
static void quant_perchannel(const uint8_t*s,int8_t*d,int n){uint8_t*cv=malloc(1024);for(int img=0;img<n;img++){const uint8_t*si=s+(size_t)img*PIXELS;int8_t*di=d+(size_t)img*PADDED;uint8_t p33[3],p67[3];for(int ch=0;ch<3;ch++){int cnt=0;for(int y=0;y<32;y++)for(int x=0;x<32;x++)cv[cnt++]=si[y*96+x*3+ch];qsort(cv,cnt,1,cmp_u8);p33[ch]=cv[cnt/3];p67[ch]=cv[2*cnt/3];if(p33[ch]==p67[ch]){p33[ch]=p33[ch]>0?p33[ch]-1:0;p67[ch]=p67[ch]<255?p67[ch]+1:255;}}for(int y=0;y<32;y++)for(int x=0;x<32;x++)for(int ch=0;ch<3;ch++){int idx=y*96+x*3+ch;di[idx]=si[idx]>p67[ch]?1:si[idx]<p33[ch]?-1:0;}}free(cv);}
static void bay_3eye(int img,double*bp){memset(bp,0,N_CLASSES*sizeof(double));
    for(int ei=0;ei<3;ei++){const uint8_t*sig=eyes[ei].joint_te+(size_t)img*SIG_PAD;
        for(int k=0;k<N_BLOCKS;k++){uint8_t bv=sig[k];if(bv==eyes[ei].bg)continue;
            const uint32_t*h=eyes[ei].hot+(size_t)k*BYTE_VALS*CLS_PAD+(size_t)bv*CLS_PAD;
            for(int c=0;c<N_CLASSES;c++)bp[c]+=log(h[c]+0.5);}}
    double mx=-1e30;for(int c=0;c<N_CLASSES;c++)if(bp[c]>mx)mx=bp[c];for(int c=0;c<N_CLASSES;c++)bp[c]-=mx;}
/* Same but on training data */
static void bay_3eye_train(int img,double*bp){memset(bp,0,N_CLASSES*sizeof(double));
    for(int ei=0;ei<3;ei++){const uint8_t*sig=eyes[ei].joint_tr+(size_t)img*SIG_PAD;
        for(int k=0;k<N_BLOCKS;k++){uint8_t bv=sig[k];if(bv==eyes[ei].bg)continue;
            const uint32_t*h=eyes[ei].hot+(size_t)k*BYTE_VALS*CLS_PAD+(size_t)bv*CLS_PAD;
            for(int c=0;c<N_CLASSES;c++)bp[c]+=log(h[c]+0.5);}}
    double mx=-1e30;for(int c=0;c<N_CLASSES;c++)if(bp[c]>mx)mx=bp[c];for(int c=0;c<N_CLASSES;c++)bp[c]-=mx;}

typedef struct{uint32_t id;uint32_t votes;}cand_t;
static int cmpv(const void*a,const void*b){return(int)((const cand_t*)b)->votes-(int)((const cand_t*)a)->votes;}
static int *ghist=NULL;static size_t ghcap=0;
static int topk(const uint32_t*v,int n,cand_t*o,int k){uint32_t mx=0;for(int i=0;i<n;i++)if(v[i]>mx)mx=v[i];if(!mx)return 0;
    if((size_t)(mx+1)>ghcap){ghcap=(size_t)(mx+1)+4096;free(ghist);ghist=malloc(ghcap*sizeof(int));}
    memset(ghist,0,(mx+1)*sizeof(int));for(int i=0;i<n;i++)if(v[i])ghist[v[i]]++;
    int cu=0,th;for(th=(int)mx;th>=1;th--){cu+=ghist[th];if(cu>=k)break;}if(th<1)th=1;
    int nc=0;for(int i=0;i<n&&nc<k;i++)if(v[i]>=(uint32_t)th){o[nc].id=i;o[nc].votes=v[i];nc++;}
    qsort(o,(size_t)nc,sizeof(cand_t),cmpv);return nc;}

/* ================================================================
 *  Edge-masked Bayesian (System 4)
 * ================================================================ */
static eye_t edge_eye;
static void quant_edgemask(const uint8_t*s,int8_t*d,int n){
    for(int img=0;img<n;img++){const uint8_t*si=s+(size_t)img*PIXELS;int8_t*di=d+(size_t)img*PADDED;
        for(int y=0;y<IMG_H;y++)for(int x=0;x<IMG_W;x++){int idx=y*IMG_W+x;
            int gx=(x<IMG_W-1)?(int)si[idx+1]-(int)si[idx]:0;int gy=(y<IMG_H-1)?(int)si[idx+IMG_W]-(int)si[idx]:0;
            int mag=abs(gx)+abs(gy);if(mag<30)di[idx]=0;else di[idx]=si[idx]>170?1:si[idx]<85?-1:0;}}}

static void load_data(void){
    raw_train=malloc((size_t)TRAIN_N*PIXELS);raw_test=malloc((size_t)TEST_N*PIXELS);
    raw_r_tr=malloc((size_t)TRAIN_N*SP_PX);raw_r_te=malloc((size_t)TEST_N*SP_PX);
    raw_g_tr=malloc((size_t)TRAIN_N*SP_PX);raw_g_te=malloc((size_t)TEST_N*SP_PX);
    raw_b_tr=malloc((size_t)TRAIN_N*SP_PX);raw_b_te=malloc((size_t)TEST_N*SP_PX);
    raw_gray_tr=malloc((size_t)TRAIN_N*SP_PX);raw_gray_te=malloc((size_t)TEST_N*SP_PX);
    train_labels=malloc(TRAIN_N);test_labels=malloc(TEST_N);char p[512];uint8_t rec[3073];
    for(int b2=1;b2<=5;b2++){snprintf(p,sizeof(p),"%sdata_batch_%d.bin",data_dir,b2);
        FILE*f=fopen(p,"rb");for(int i=0;i<10000;i++){if(fread(rec,1,3073,f)!=3073){fclose(f);exit(1);}
            int idx=(b2-1)*10000+i;train_labels[idx]=rec[0];
            uint8_t*d=raw_train+(size_t)idx*PIXELS;const uint8_t*r=rec+1,*g=rec+1+1024,*b3=rec+1+2048;
            for(int y=0;y<32;y++)for(int x=0;x<32;x++){int si=y*32+x,di=y*96+x*3;d[di]=r[si];d[di+1]=g[si];d[di+2]=b3[si];}
            memcpy(raw_r_tr+(size_t)idx*SP_PX,r,SP_PX);memcpy(raw_g_tr+(size_t)idx*SP_PX,g,SP_PX);memcpy(raw_b_tr+(size_t)idx*SP_PX,b3,SP_PX);
            uint8_t*gd=raw_gray_tr+(size_t)idx*SP_PX;for(int p2=0;p2<SP_PX;p2++)gd[p2]=(uint8_t)((77*(int)r[p2]+150*(int)g[p2]+29*(int)b3[p2])>>8);}fclose(f);}
    snprintf(p,sizeof(p),"%stest_batch.bin",data_dir);
    FILE*f=fopen(p,"rb");for(int i=0;i<10000;i++){if(fread(rec,1,3073,f)!=3073){fclose(f);exit(1);}
        test_labels[i]=rec[0];uint8_t*d=raw_test+(size_t)i*PIXELS;const uint8_t*r=rec+1,*g=rec+1+1024,*b3=rec+1+2048;
        for(int y=0;y<32;y++)for(int x=0;x<32;x++){int si=y*32+x,di=y*96+x*3;d[di]=r[si];d[di+1]=g[si];d[di+2]=b3[si];}
        memcpy(raw_r_te+(size_t)i*SP_PX,r,SP_PX);memcpy(raw_g_te+(size_t)i*SP_PX,g,SP_PX);memcpy(raw_b_te+(size_t)i*SP_PX,b3,SP_PX);
        uint8_t*gd=raw_gray_te+(size_t)i*SP_PX;for(int p2=0;p2<SP_PX;p2++)gd[p2]=(uint8_t)((77*(int)r[p2]+150*(int)g[p2]+29*(int)b3[p2])>>8);}fclose(f);}

int main(int argc,char**argv){
    double t0=now_sec();
    if(argc>1){data_dir=argv[1];size_t l=strlen(data_dir);
        if(l&&data_dir[l-1]!='/'){char*b2=malloc(l+2);memcpy(b2,data_dir,l);b2[l]='/';b2[l+1]=0;data_dir=b2;}}
    printf("=== SSTT CIFAR-10: Meta-System Router ===\n\n");
    load_data();

    /* Build RGB Gauss maps for all train + test */
    printf("Computing Gauss maps...\n");
    gm_tr=malloc((size_t)TRAIN_N*GM_RGB_FEAT*2);gm_te=malloc((size_t)TEST_N*GM_RGB_FEAT*2);
    const uint8_t *ch_tr[]={raw_gray_tr,raw_r_tr,raw_g_tr,raw_b_tr};
    const uint8_t *ch_te[]={raw_gray_te,raw_r_te,raw_g_te,raw_b_te};
    for(int i=0;i<TRAIN_N;i++)for(int ch=0;ch<NCH;ch++)gauss_map_grid(ch_tr[ch]+(size_t)i*SP_PX,gm_tr+(size_t)i*GM_RGB_FEAT+ch*G4_FEAT);
    for(int i=0;i<TEST_N;i++)for(int ch=0;ch<NCH;ch++)gauss_map_grid(ch_te[ch]+(size_t)i*SP_PX,gm_te+(size_t)i*GM_RGB_FEAT+ch*G4_FEAT);

    /* Build 3-eye stereo + edge-masked eye */
    printf("Building 4 systems...\n");
    void(*qfns[])(const uint8_t*,int8_t*,int)={quant_fixed,quant_adaptive,quant_perchannel};
    for(int ei=0;ei<3;ei++){build_eye(&eyes[ei],qfns[ei]);printf("  Eye %d built\n",ei+1);}
    build_eye(&edge_eye,quant_edgemask);printf("  Edge eye built\n");
    printf("  Systems built (%.1f sec)\n\n",now_sec()-t0);

    /* ================================================================
     * PHASE 1: Pre-classify ALL training images with all 4 systems
     *          using leave-one-out style (classify train[i] against rest)
     *          Approximation: classify train[i] against full training set
     *          (self-inclusion bias but fast)
     * ================================================================ */
    printf("Phase 1: Pre-classifying training set with all 4 systems...\n");
    uint8_t *train_preds=malloc((size_t)TRAIN_N*N_SYSTEMS); /* pred per system */
    uint8_t *train_correct=malloc((size_t)TRAIN_N*N_SYSTEMS); /* 1 if correct */
    uint32_t *vbuf=calloc(TRAIN_N,4);

    for(int i=0;i<TRAIN_N;i++){
        /* System 0: Cascade Gauss (stereo vote → Gauss rank k=7) */
        memset(vbuf,0,TRAIN_N*4);
        /* Use training-set eyes for voting (self-inclusive, slight bias) */
        for(int ei=0;ei<3;ei++){const uint8_t*sig=eyes[ei].joint_tr+(size_t)i*SIG_PAD;
            for(int k=0;k<N_BLOCKS;k++){uint8_t bv=sig[k];if(bv==eyes[ei].bg)continue;
                uint16_t w=eyes[ei].ig[k];uint32_t off=eyes[ei].idx_off[k][bv];uint16_t sz=eyes[ei].idx_sz[k][bv];
                for(uint16_t j=0;j<sz;j++)vbuf[eyes[ei].idx_pool[off+j]]+=w;}}
        vbuf[i]=0; /* exclude self */
        cand_t cands[TOP_K];int nc=topk(vbuf,TRAIN_N,cands,TOP_K);
        /* Gauss rank among candidates */
        const int16_t*qi=gm_tr+(size_t)i*GM_RGB_FEAT;
        int32_t bd[TOP_K];for(int j=0;j<nc;j++)bd[j]=hist_l1(qi,gm_tr+(size_t)cands[j].id*GM_RGB_FEAT,GM_RGB_FEAT);
        for(int j=1;j<nc;j++){int32_t kd=bd[j];cand_t kc=cands[j];int x=j-1;
            while(x>=0&&bd[x]>kd){bd[x+1]=bd[x];cands[x+1]=cands[x];x--;}bd[x+1]=kd;cands[x+1]=kc;}
        {int v[N_CLASSES]={0};int kk=7<nc?7:nc;for(int j=0;j<kk;j++)v[train_labels[cands[j].id]]++;
         int pred=0;for(int c=1;c<N_CLASSES;c++)if(v[c]>v[pred])pred=c;
         train_preds[i*N_SYSTEMS+0]=(uint8_t)pred;train_correct[i*N_SYSTEMS+0]=(pred==train_labels[i])?1:0;}

        /* System 1: Brute Gauss kNN k=5 */
        {int32_t best_d[5];uint8_t best_l[5];for(int j=0;j<5;j++){best_d[j]=INT32_MAX;best_l[j]=0;}
         for(int j=0;j<TRAIN_N;j++){if(j==i)continue;
             int32_t d=hist_l1(qi,gm_tr+(size_t)j*GM_RGB_FEAT,GM_RGB_FEAT);
             if(d<best_d[4]){best_d[4]=d;best_l[4]=train_labels[j];
                 for(int x=3;x>=0;x--)if(best_d[x+1]<best_d[x]){int32_t td=best_d[x];best_d[x]=best_d[x+1];best_d[x+1]=td;
                     uint8_t tl=best_l[x];best_l[x]=best_l[x+1];best_l[x+1]=tl;}else break;}}
         int v[N_CLASSES]={0};for(int j=0;j<5;j++)v[best_l[j]]++;
         int pred=0;for(int c=1;c<N_CLASSES;c++)if(v[c]>v[pred])pred=c;
         train_preds[i*N_SYSTEMS+1]=(uint8_t)pred;train_correct[i*N_SYSTEMS+1]=(pred==train_labels[i])?1:0;}

        /* System 2: 3-eye Bayesian */
        {double bp[N_CLASSES];bay_3eye_train(i,bp);
         int pred=0;for(int c=1;c<N_CLASSES;c++)if(bp[c]>bp[pred])pred=c;
         train_preds[i*N_SYSTEMS+2]=(uint8_t)pred;train_correct[i*N_SYSTEMS+2]=(pred==train_labels[i])?1:0;}

        /* System 3: Edge-masked Bayesian */
        {const uint8_t*sig=edge_eye.joint_tr+(size_t)i*SIG_PAD;
         double bp[N_CLASSES];memset(bp,0,sizeof(bp));
         for(int k=0;k<N_BLOCKS;k++){uint8_t bv=sig[k];if(bv==edge_eye.bg)continue;
             const uint32_t*h=edge_eye.hot+(size_t)k*BYTE_VALS*CLS_PAD+(size_t)bv*CLS_PAD;
             for(int c=0;c<N_CLASSES;c++)bp[c]+=log(h[c]+0.5);}
         int pred=0;for(int c=1;c<N_CLASSES;c++)if(bp[c]>bp[pred])pred=c;
         train_preds[i*N_SYSTEMS+3]=(uint8_t)pred;train_correct[i*N_SYSTEMS+3]=(pred==train_labels[i])?1:0;}

        if((i+1)%5000==0)fprintf(stderr,"  Train pre-classify: %d/%d\r",i+1,TRAIN_N);
    }
    fprintf(stderr,"\n");

    /* Training set accuracy per system */
    printf("\n  Training set accuracy (self-inclusive, biased):\n");
    for(int si=0;si<N_SYSTEMS;si++){int c=0;for(int i=0;i<TRAIN_N;i++)c+=train_correct[i*N_SYSTEMS+si];
        printf("    System %d %-25s %.2f%%\n",si,sysnames[si],100.0*c/TRAIN_N);}

    /* Oracle: best system per training image */
    {int oracle=0;for(int i=0;i<TRAIN_N;i++){int any=0;for(int si=0;si<N_SYSTEMS;si++)if(train_correct[i*N_SYSTEMS+si])any=1;oracle+=any;}
     printf("    Oracle (any system correct):   %.2f%%\n",100.0*oracle/TRAIN_N);}

    /* ================================================================
     * PHASE 2: Classify test images with all 4 systems
     * ================================================================ */
    printf("\nPhase 2: Classifying test set with all 4 systems...\n");
    uint8_t *test_preds=malloc((size_t)TEST_N*N_SYSTEMS);
    uint8_t *test_correct=malloc((size_t)TEST_N*N_SYSTEMS);

    for(int i=0;i<TEST_N;i++){
        /* System 0: Cascade Gauss */
        memset(vbuf,0,TRAIN_N*4);for(int ei=0;ei<3;ei++)vote_eye(&eyes[ei],i,vbuf);
        cand_t cands[TOP_K];int nc=topk(vbuf,TRAIN_N,cands,TOP_K);
        const int16_t*qi=gm_te+(size_t)i*GM_RGB_FEAT;
        int32_t bd[TOP_K];for(int j=0;j<nc;j++)bd[j]=hist_l1(qi,gm_tr+(size_t)cands[j].id*GM_RGB_FEAT,GM_RGB_FEAT);
        for(int j=1;j<nc;j++){int32_t kd=bd[j];cand_t kc=cands[j];int x=j-1;
            while(x>=0&&bd[x]>kd){bd[x+1]=bd[x];cands[x+1]=cands[x];x--;}bd[x+1]=kd;cands[x+1]=kc;}
        {int v[N_CLASSES]={0};int kk=7<nc?7:nc;for(int j=0;j<kk;j++)v[train_labels[cands[j].id]]++;
         int pred=0;for(int c=1;c<N_CLASSES;c++)if(v[c]>v[pred])pred=c;
         test_preds[i*N_SYSTEMS+0]=(uint8_t)pred;test_correct[i*N_SYSTEMS+0]=(pred==test_labels[i])?1:0;}

        /* System 1: Brute Gauss kNN k=5 */
        {int32_t best_d[5];uint8_t best_l[5];for(int j=0;j<5;j++){best_d[j]=INT32_MAX;best_l[j]=0;}
         for(int j=0;j<TRAIN_N;j++){int32_t d=hist_l1(qi,gm_tr+(size_t)j*GM_RGB_FEAT,GM_RGB_FEAT);
             if(d<best_d[4]){best_d[4]=d;best_l[4]=train_labels[j];
                 for(int x=3;x>=0;x--)if(best_d[x+1]<best_d[x]){int32_t td=best_d[x];best_d[x]=best_d[x+1];best_d[x+1]=td;
                     uint8_t tl=best_l[x];best_l[x]=best_l[x+1];best_l[x+1]=tl;}else break;}}
         int v[N_CLASSES]={0};for(int j=0;j<5;j++)v[best_l[j]]++;
         int pred=0;for(int c=1;c<N_CLASSES;c++)if(v[c]>v[pred])pred=c;
         test_preds[i*N_SYSTEMS+1]=(uint8_t)pred;test_correct[i*N_SYSTEMS+1]=(pred==test_labels[i])?1:0;}

        /* System 2: 3-eye Bayesian */
        {double bp[N_CLASSES];bay_3eye(i,bp);
         int pred=0;for(int c=1;c<N_CLASSES;c++)if(bp[c]>bp[pred])pred=c;
         test_preds[i*N_SYSTEMS+2]=(uint8_t)pred;test_correct[i*N_SYSTEMS+2]=(pred==test_labels[i])?1:0;}

        /* System 3: Edge-masked Bayesian */
        {const uint8_t*sig=edge_eye.joint_te+(size_t)i*SIG_PAD;
         double bp[N_CLASSES];memset(bp,0,sizeof(bp));
         for(int k=0;k<N_BLOCKS;k++){uint8_t bv=sig[k];if(bv==edge_eye.bg)continue;
             const uint32_t*h=edge_eye.hot+(size_t)k*BYTE_VALS*CLS_PAD+(size_t)bv*CLS_PAD;
             for(int c=0;c<N_CLASSES;c++)bp[c]+=log(h[c]+0.5);}
         int pred=0;for(int c=1;c<N_CLASSES;c++)if(bp[c]>bp[pred])pred=c;
         test_preds[i*N_SYSTEMS+3]=(uint8_t)pred;test_correct[i*N_SYSTEMS+3]=(pred==test_labels[i])?1:0;}

        if((i+1)%1000==0)fprintf(stderr,"  Test classify: %d/%d\r",i+1,TEST_N);
    }
    fprintf(stderr,"\n");

    printf("\n  Test set accuracy per system:\n");
    for(int si=0;si<N_SYSTEMS;si++){int c=0;for(int i=0;i<TEST_N;i++)c+=test_correct[i*N_SYSTEMS+si];
        printf("    System %d %-25s %.2f%%\n",si,sysnames[si],100.0*c/TEST_N);}
    {int oracle=0;for(int i=0;i<TEST_N;i++){int any=0;for(int si=0;si<N_SYSTEMS;si++)if(test_correct[i*N_SYSTEMS+si])any=1;oracle+=any;}
     printf("    Oracle (any system correct):   %.2f%%\n",100.0*oracle/TEST_N);}

    /* ================================================================
     * PHASE 3: ROUTING — for each test image, find k nearest training
     * images (by Gauss map), check which system succeeded on those
     * neighbors, route to the system with highest neighbor success rate
     * ================================================================ */
    printf("\nPhase 3: Routing via neighbor system success...\n\n");

    int k_route_vals[] = {5, 10, 20, 50, 100};
    for (int ki = 0; ki < 5; ki++) {
        int k_route = k_route_vals[ki];
        int correct = 0;
        int system_chosen[N_SYSTEMS] = {0};

        for (int i = 0; i < TEST_N; i++) {
            const int16_t *qi = gm_te + (size_t)i * GM_RGB_FEAT;

            /* Find k nearest training images by Gauss map L1 */
            typedef struct{int32_t d;int idx;}nb_t;
            nb_t *best = malloc(k_route * sizeof(nb_t));
            for(int j=0;j<k_route;j++){best[j].d=INT32_MAX;best[j].idx=0;}
            for (int j = 0; j < TRAIN_N; j++) {
                int32_t d = hist_l1(qi, gm_tr + (size_t)j * GM_RGB_FEAT, GM_RGB_FEAT);
                if (d < best[k_route-1].d) {
                    best[k_route-1] = (nb_t){d, j};
                    for(int x=k_route-2;x>=0;x--)
                        if(best[x+1].d<best[x].d){nb_t t=best[x];best[x]=best[x+1];best[x+1]=t;}else break;
                }
            }

            /* Count neighbor success rate per system */
            int sys_success[N_SYSTEMS] = {0};
            for (int j = 0; j < k_route; j++)
                for (int si = 0; si < N_SYSTEMS; si++)
                    sys_success[si] += train_correct[best[j].idx * N_SYSTEMS + si];

            /* Route to system with highest neighbor success */
            int best_sys = 0;
            for (int si = 1; si < N_SYSTEMS; si++)
                if (sys_success[si] > sys_success[best_sys]) best_sys = si;

            system_chosen[best_sys]++;
            if (test_correct[i * N_SYSTEMS + best_sys]) correct++;
            free(best);
        }

        printf("  k_route=%3d: %.2f%% | chosen: ", k_route, 100.0 * correct / TEST_N);
        for (int si = 0; si < N_SYSTEMS; si++) printf("S%d=%d ", si, system_chosen[si]);
        printf("\n");
    }

    /* ================================================================
     * PHASE 4: Majority vote across all systems (ensemble)
     * ================================================================ */
    printf("\n=== ENSEMBLE: Majority vote across all 4 systems ===\n\n");
    {int correct=0;int pc[N_CLASSES]={0},pt[N_CLASSES]={0};
     for(int i=0;i<TEST_N;i++){pt[test_labels[i]]++;
         int votes[N_CLASSES]={0};
         for(int si=0;si<N_SYSTEMS;si++)votes[test_preds[i*N_SYSTEMS+si]]++;
         int pred=0;for(int c=1;c<N_CLASSES;c++)if(votes[c]>votes[pred])pred=c;
         if(pred==test_labels[i]){correct++;pc[test_labels[i]]++;}}
     printf("  Majority vote: %.2f%%\n",100.0*correct/TEST_N);
     for(int c=0;c<N_CLASSES;c++)printf("    %d %-10s %.1f%%\n",c,cn[c],100.0*pc[c]/pt[c]);}

    /* ================================================================
     * PHASE 5: Weighted ensemble (weight by training accuracy)
     * ================================================================ */
    printf("\n=== WEIGHTED ENSEMBLE ===\n\n");
    {
        /* Per-system training accuracy as weights */
        double sys_acc[N_SYSTEMS];
        for(int si=0;si<N_SYSTEMS;si++){int c=0;for(int i=0;i<TRAIN_N;i++)c+=train_correct[i*N_SYSTEMS+si];sys_acc[si]=(double)c/TRAIN_N;}

        int correct=0;
        for(int i=0;i<TEST_N;i++){
            double weighted[N_CLASSES];memset(weighted,0,sizeof(weighted));
            for(int si=0;si<N_SYSTEMS;si++)weighted[test_preds[i*N_SYSTEMS+si]]+=sys_acc[si];
            int pred=0;for(int c=1;c<N_CLASSES;c++)if(weighted[c]>weighted[pred])pred=c;
            if(pred==test_labels[i])correct++;}
        printf("  Weighted ensemble: %.2f%%\n",100.0*correct/TEST_N);
    }

    printf("\n=== REFERENCE ===\n");
    printf("  Cascade Gauss alone:   50.18%%\n");
    printf("  Oracle (any correct):  see above\n");
    printf("  Target:                55%%+\n");

    printf("\nTotal: %.1f sec\n",now_sec()-t0);
    free(vbuf);free(train_preds);free(train_correct);free(test_preds);free(test_correct);
    return 0;
}
