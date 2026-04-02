/*
 * sstt_cifar10_mt7vote.c — MT7 Voting: All 4 Ternary Planes in the Vote
 *
 * Currently: voting uses coarsest ternary plane only (27-val blocks).
 * This experiment: build block sigs + inverted index + IG weights on
 * ALL 4 MT7 planes. Vote across 4 planes. Each plane captures different
 * detail level. The combined vote should retrieve better candidates.
 *
 * Architecture:
 *   - 4 planes × 1024 blocks = 4096 block positions
 *   - Each with its own IG weights and inverted index
 *   - Joint Encoding D on each plane = 4 × 256-value signatures
 *   - Multi-probe on each plane
 *   - MT7 dot + topo + Bayesian for ranking (unchanged)
 *
 * Build: gcc -O3 -mavx2 -march=native -o sstt_cifar10_mt7vote src/sstt_cifar10_mt7vote.c -lm
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
#define N_PLANES 7
#define SPATIAL_W 32
#define SPATIAL_H 32
#define GRAY_PIXELS 1024
#define GRAY_PADDED 1024
#define MAX_REGIONS 16

static const char *data_dir="data-cifar10/";
static const char *cn[]={"airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"};
static double now_sec(void){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return ts.tv_sec+ts.tv_nsec*1e-9;}

static uint8_t *raw_train,*raw_test,*raw_gray_tr,*raw_gray_te,*train_labels,*test_labels;

/* 4 MT7 planes */
static int8_t *plane_tr[N_PLANES], *plane_te[N_PLANES];
static int8_t *plane_hg_tr[N_PLANES], *plane_hg_te[N_PLANES];
static int8_t *plane_vg_tr[N_PLANES], *plane_vg_te[N_PLANES];

/* Per-plane: block sigs, transitions, Encoding D, hot map, IG, index */
static uint8_t *p_px_tr[N_PLANES],*p_px_te[N_PLANES];
static uint8_t *p_hg_tr[N_PLANES],*p_hg_te[N_PLANES];
static uint8_t *p_vg_tr[N_PLANES],*p_vg_te[N_PLANES];
static uint8_t *p_ht_tr[N_PLANES],*p_ht_te[N_PLANES];
static uint8_t *p_vt_tr[N_PLANES],*p_vt_te[N_PLANES];
static uint8_t *p_joint_tr[N_PLANES],*p_joint_te[N_PLANES];
static uint32_t *p_hot[N_PLANES];
static uint16_t p_ig[N_PLANES][N_BLOCKS];
static uint8_t p_bg[N_PLANES];
static uint8_t p_nbr[BYTE_VALS][8];
/* Per-plane inverted index */
static uint32_t p_idx_off[N_PLANES][N_BLOCKS][BYTE_VALS];
static uint16_t p_idx_sz[N_PLANES][N_BLOCKS][BYTE_VALS];
static uint32_t *p_idx_pool[N_PLANES];

/* Topo features (grayscale) */
static int8_t *ghg_tr,*ghg_te,*gvg_tr,*gvg_te;
static int16_t *divneg_tr,*divneg_te,*divcy_tr,*divcy_te,*gdiv_tr,*gdiv_te;

/* Core functions */
static inline int8_t ct(int v){return v>0?1:v<0?-1:0;}
static void gr(const int8_t*t,int8_t*h,int8_t*v,int n,int w,int ht,int pd){
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

static void load_data(void){
    raw_train=malloc((size_t)TRAIN_N*PIXELS);raw_test=malloc((size_t)TEST_N*PIXELS);
    raw_gray_tr=malloc((size_t)TRAIN_N*GRAY_PIXELS);raw_gray_te=malloc((size_t)TEST_N*GRAY_PIXELS);
    train_labels=malloc(TRAIN_N);test_labels=malloc(TEST_N);
    char p[512];uint8_t rec[3073];
    for(int b2=1;b2<=5;b2++){snprintf(p,sizeof(p),"%sdata_batch_%d.bin",data_dir,b2);
        FILE*f=fopen(p,"rb");for(int i=0;i<10000;i++){if(fread(rec,1,3073,f)!=3073){fclose(f);exit(1);}
            int idx=(b2-1)*10000+i;train_labels[idx]=rec[0];
            uint8_t*d=raw_train+(size_t)idx*PIXELS;const uint8_t*r=rec+1,*g=rec+1+1024,*b3=rec+1+2048;
            for(int y=0;y<32;y++)for(int x=0;x<32;x++){int si=y*32+x,di=y*96+x*3;d[di]=r[si];d[di+1]=g[si];d[di+2]=b3[si];}
            uint8_t*gdd=raw_gray_tr+(size_t)idx*GRAY_PIXELS;
            for(int p2=0;p2<1024;p2++)gdd[p2]=(uint8_t)((77*(int)r[p2]+150*(int)g[p2]+29*(int)b3[p2])>>8);}fclose(f);}
    snprintf(p,sizeof(p),"%stest_batch.bin",data_dir);
    FILE*f=fopen(p,"rb");for(int i=0;i<10000;i++){if(fread(rec,1,3073,f)!=3073){fclose(f);exit(1);}
        test_labels[i]=rec[0];uint8_t*d=raw_test+(size_t)i*PIXELS;const uint8_t*r=rec+1,*g=rec+1+1024,*b3=rec+1+2048;
        for(int y=0;y<32;y++)for(int x=0;x<32;x++){int si=y*32+x,di=y*96+x*3;d[di]=r[si];d[di+1]=g[si];d[di+2]=b3[si];}
        uint8_t*gdd=raw_gray_te+(size_t)i*GRAY_PIXELS;
        for(int p2=0;p2<1024;p2++)gdd[p2]=(uint8_t)((77*(int)r[p2]+150*(int)g[p2]+29*(int)b3[p2])>>8);}fclose(f);}

/* Build one plane's voting infra: block sigs, transitions, Encoding D, hot map, IG, index */
static void build_plane(int pi) {
    /* Block sigs */
    p_px_tr[pi]=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD);p_px_te[pi]=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    p_hg_tr[pi]=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD);p_hg_te[pi]=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    p_vg_tr[pi]=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD);p_vg_te[pi]=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    bsf(plane_tr[pi],p_px_tr[pi],TRAIN_N);bsf(plane_te[pi],p_px_te[pi],TEST_N);
    bsf(plane_hg_tr[pi],p_hg_tr[pi],TRAIN_N);bsf(plane_hg_te[pi],p_hg_te[pi],TEST_N);
    bsf(plane_vg_tr[pi],p_vg_tr[pi],TRAIN_N);bsf(plane_vg_te[pi],p_vg_te[pi],TEST_N);
    /* Transitions */
    p_ht_tr[pi]=aligned_alloc(32,(size_t)TRAIN_N*TRANS_PAD);p_ht_te[pi]=aligned_alloc(32,(size_t)TEST_N*TRANS_PAD);
    p_vt_tr[pi]=aligned_alloc(32,(size_t)TRAIN_N*TRANS_PAD);p_vt_te[pi]=aligned_alloc(32,(size_t)TEST_N*TRANS_PAD);
    trf(p_px_tr[pi],p_ht_tr[pi],p_vt_tr[pi],TRAIN_N);trf(p_px_te[pi],p_ht_te[pi],p_vt_te[pi],TEST_N);
    /* Encoding D */
    p_joint_tr[pi]=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD);p_joint_te[pi]=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    jsf(p_joint_tr[pi],TRAIN_N,p_px_tr[pi],p_hg_tr[pi],p_vg_tr[pi],p_ht_tr[pi],p_vt_tr[pi]);
    jsf(p_joint_te[pi],TEST_N,p_px_te[pi],p_hg_te[pi],p_vg_te[pi],p_ht_te[pi],p_vt_te[pi]);
    /* Background */
    long vc[BYTE_VALS]={0};for(int i=0;i<TRAIN_N;i++){const uint8_t*sig=p_joint_tr[pi]+(size_t)i*SIG_PAD;
        for(int k=0;k<N_BLOCKS;k++)vc[sig[k]]++;}
    p_bg[pi]=0;long mc=0;for(int v=0;v<BYTE_VALS;v++)if(vc[v]>mc){mc=vc[v];p_bg[pi]=(uint8_t)v;}
    printf("    Plane %d BG=%d (%.1f%%)\n",pi,p_bg[pi],100.0*mc/((long)TRAIN_N*N_BLOCKS));
    /* Hot map + IG */
    p_hot[pi]=aligned_alloc(32,(size_t)N_BLOCKS*BYTE_VALS*CLS_PAD*4);
    memset(p_hot[pi],0,(size_t)N_BLOCKS*BYTE_VALS*CLS_PAD*4);
    for(int i=0;i<TRAIN_N;i++){int l=train_labels[i];const uint8_t*sig=p_joint_tr[pi]+(size_t)i*SIG_PAD;
        for(int k=0;k<N_BLOCKS;k++)p_hot[pi][(size_t)k*BYTE_VALS*CLS_PAD+(size_t)sig[k]*CLS_PAD+l]++;}
    {int cc[N_CLASSES]={0};for(int i=0;i<TRAIN_N;i++)cc[train_labels[i]]++;
     double hc=0;for(int c=0;c<N_CLASSES;c++){double p=(double)cc[c]/TRAIN_N;if(p>0)hc-=p*log2(p);}
     double raw2[N_BLOCKS],mx=0;
     for(int k=0;k<N_BLOCKS;k++){double hcond=0;
         for(int v=0;v<BYTE_VALS;v++){if((uint8_t)v==p_bg[pi])continue;
             const uint32_t*h=p_hot[pi]+(size_t)k*BYTE_VALS*CLS_PAD+(size_t)v*CLS_PAD;
             int vt=0;for(int c=0;c<N_CLASSES;c++)vt+=(int)h[c];if(!vt)continue;
             double pv=(double)vt/TRAIN_N,hv=0;for(int c=0;c<N_CLASSES;c++){double pc2=(double)h[c]/vt;if(pc2>0)hv-=pc2*log2(pc2);}
             hcond+=pv*hv;}raw2[k]=hc-hcond;if(raw2[k]>mx)mx=raw2[k];}
     for(int k=0;k<N_BLOCKS;k++){p_ig[pi][k]=mx>0?(uint16_t)(raw2[k]/mx*IG_SCALE+0.5):1;if(!p_ig[pi][k])p_ig[pi][k]=1;}}
    /* Index */
    memset(p_idx_sz[pi],0,sizeof(p_idx_sz[pi]));
    for(int i=0;i<TRAIN_N;i++){const uint8_t*sig=p_joint_tr[pi]+(size_t)i*SIG_PAD;
        for(int k=0;k<N_BLOCKS;k++)if(sig[k]!=p_bg[pi])p_idx_sz[pi][k][sig[k]]++;}
    uint32_t tot=0;for(int k=0;k<N_BLOCKS;k++)for(int v=0;v<BYTE_VALS;v++){p_idx_off[pi][k][v]=tot;tot+=p_idx_sz[pi][k][v];}
    p_idx_pool[pi]=malloc((size_t)tot*4);
    uint32_t(*wp)[BYTE_VALS]=malloc((size_t)N_BLOCKS*BYTE_VALS*4);memcpy(wp,p_idx_off[pi],(size_t)N_BLOCKS*BYTE_VALS*4);
    for(int i=0;i<TRAIN_N;i++){const uint8_t*sig=p_joint_tr[pi]+(size_t)i*SIG_PAD;
        for(int k=0;k<N_BLOCKS;k++)if(sig[k]!=p_bg[pi])p_idx_pool[pi][wp[k][sig[k]]++]=(uint32_t)i;}free(wp);
    printf("    Plane %d index: %u entries (%.1f MB)\n",pi,tot,(double)tot*4/(1024*1024));
}

int main(int argc,char**argv){
    double t0=now_sec();
    if(argc>1){data_dir=argv[1];size_t l=strlen(data_dir);
        if(l&&data_dir[l-1]!='/'){char*b2=malloc(l+2);memcpy(b2,data_dir,l);b2[l]='/';b2[l+1]=0;data_dir=b2;}}
    printf("=== SSTT CIFAR-10: MT7 Full-Pipeline Voting ===\n\n");
    load_data();

    /* MT7 quantization: 4 ternary planes */
    printf("Computing 4 MT7 planes + gradients...\n");
    for(int pi=0;pi<N_PLANES;pi++){
        plane_tr[pi]=aligned_alloc(32,(size_t)TRAIN_N*PADDED);plane_te[pi]=aligned_alloc(32,(size_t)TEST_N*PADDED);
        plane_hg_tr[pi]=aligned_alloc(32,(size_t)TRAIN_N*PADDED);plane_hg_te[pi]=aligned_alloc(32,(size_t)TEST_N*PADDED);
        plane_vg_tr[pi]=aligned_alloc(32,(size_t)TRAIN_N*PADDED);plane_vg_te[pi]=aligned_alloc(32,(size_t)TEST_N*PADDED);
    }
    for(int img=0;img<TRAIN_N;img++){const uint8_t*s=raw_train+(size_t)img*PIXELS;
        for(int i=0;i<PIXELS;i++){int bin=(int)s[i]*2187/256;
            plane_tr[0][(size_t)img*PADDED+i]=(int8_t)(bin/729-1);int r=bin%729;
            plane_tr[1][(size_t)img*PADDED+i]=(int8_t)(r/243-1);r%=243;
            plane_tr[2][(size_t)img*PADDED+i]=(int8_t)(r/81-1);r%=81;
            plane_tr[3][(size_t)img*PADDED+i]=(int8_t)(r/27-1);r%=27;
            plane_tr[4][(size_t)img*PADDED+i]=(int8_t)(r/9-1);r%=9;
            plane_tr[5][(size_t)img*PADDED+i]=(int8_t)(r/3-1);
            plane_tr[6][(size_t)img*PADDED+i]=(int8_t)(r%3-1);}}
    for(int img=0;img<TEST_N;img++){const uint8_t*s=raw_test+(size_t)img*PIXELS;
        for(int i=0;i<PIXELS;i++){int bin=(int)s[i]*2187/256;
            plane_te[0][(size_t)img*PADDED+i]=(int8_t)(bin/729-1);int r=bin%729;
            plane_te[1][(size_t)img*PADDED+i]=(int8_t)(r/243-1);r%=243;
            plane_te[2][(size_t)img*PADDED+i]=(int8_t)(r/81-1);r%=81;
            plane_te[3][(size_t)img*PADDED+i]=(int8_t)(r/27-1);r%=27;
            plane_te[4][(size_t)img*PADDED+i]=(int8_t)(r/9-1);r%=9;
            plane_te[5][(size_t)img*PADDED+i]=(int8_t)(r/3-1);
            plane_te[6][(size_t)img*PADDED+i]=(int8_t)(r%3-1);}}
    for(int pi=0;pi<N_PLANES;pi++){
        gr(plane_tr[pi],plane_hg_tr[pi],plane_vg_tr[pi],TRAIN_N,IMG_W,IMG_H,PADDED);
        gr(plane_te[pi],plane_hg_te[pi],plane_vg_te[pi],TEST_N,IMG_W,IMG_H,PADDED);}

    /* Topo features (grayscale) */
    {int8_t*gt_tr=aligned_alloc(32,(size_t)TRAIN_N*GRAY_PADDED),*gt_te=aligned_alloc(32,(size_t)TEST_N*GRAY_PADDED);
     ghg_tr=aligned_alloc(32,(size_t)TRAIN_N*GRAY_PADDED);ghg_te=aligned_alloc(32,(size_t)TEST_N*GRAY_PADDED);
     gvg_tr=aligned_alloc(32,(size_t)TRAIN_N*GRAY_PADDED);gvg_te=aligned_alloc(32,(size_t)TEST_N*GRAY_PADDED);
     const __m256i bi=_mm256_set1_epi8((char)0x80),th=_mm256_set1_epi8((char)(170^0x80)),
         tl=_mm256_set1_epi8((char)(85^0x80)),on=_mm256_set1_epi8(1);
     for(int i2=0;i2<TRAIN_N;i2++){const uint8_t*si=raw_gray_tr+(size_t)i2*GRAY_PIXELS;int8_t*di=gt_tr+(size_t)i2*GRAY_PADDED;
         for(int i=0;i<GRAY_PIXELS;i+=32){__m256i p=_mm256_loadu_si256((const __m256i*)(si+i));__m256i sp=_mm256_xor_si256(p,bi);
             _mm256_storeu_si256((__m256i*)(di+i),_mm256_sub_epi8(_mm256_and_si256(_mm256_cmpgt_epi8(sp,th),on),
                 _mm256_and_si256(_mm256_cmpgt_epi8(tl,sp),on)));}}
     for(int i2=0;i2<TEST_N;i2++){const uint8_t*si=raw_gray_te+(size_t)i2*GRAY_PIXELS;int8_t*di=gt_te+(size_t)i2*GRAY_PADDED;
         for(int i=0;i<GRAY_PIXELS;i+=32){__m256i p=_mm256_loadu_si256((const __m256i*)(si+i));__m256i sp=_mm256_xor_si256(p,bi);
             _mm256_storeu_si256((__m256i*)(di+i),_mm256_sub_epi8(_mm256_and_si256(_mm256_cmpgt_epi8(sp,th),on),
                 _mm256_and_si256(_mm256_cmpgt_epi8(tl,sp),on)));}}
     gr(gt_tr,ghg_tr,gvg_tr,TRAIN_N,SPATIAL_W,SPATIAL_H,GRAY_PADDED);gr(gt_te,ghg_te,gvg_te,TEST_N,SPATIAL_W,SPATIAL_H,GRAY_PADDED);
     divneg_tr=malloc((size_t)TRAIN_N*2);divneg_te=malloc((size_t)TEST_N*2);divcy_tr=malloc((size_t)TRAIN_N*2);divcy_te=malloc((size_t)TEST_N*2);
     for(int i=0;i<TRAIN_N;i++)divf(ghg_tr+(size_t)i*GRAY_PADDED,gvg_tr+(size_t)i*GRAY_PADDED,&divneg_tr[i],&divcy_tr[i]);
     for(int i=0;i<TEST_N;i++)divf(ghg_te+(size_t)i*GRAY_PADDED,gvg_te+(size_t)i*GRAY_PADDED,&divneg_te[i],&divcy_te[i]);
     gdiv_tr=malloc((size_t)TRAIN_N*MAX_REGIONS*2);gdiv_te=malloc((size_t)TEST_N*MAX_REGIONS*2);
     for(int i=0;i<TRAIN_N;i++)gdf(ghg_tr+(size_t)i*GRAY_PADDED,gvg_tr+(size_t)i*GRAY_PADDED,3,3,gdiv_tr+(size_t)i*MAX_REGIONS);
     for(int i=0;i<TEST_N;i++)gdf(ghg_te+(size_t)i*GRAY_PADDED,gvg_te+(size_t)i*GRAY_PADDED,3,3,gdiv_te+(size_t)i*MAX_REGIONS);
     free(gt_tr);free(gt_te);}

    /* Build per-plane voting infra */
    printf("\nBuilding 4-plane voting infrastructure...\n");
    for(int v=0;v<BYTE_VALS;v++)for(int b3=0;b3<8;b3++)p_nbr[v][b3]=(uint8_t)(v^(1<<b3));
    for(int pi=0;pi<N_PLANES;pi++)build_plane(pi);
    printf("  Built (%.1f sec)\n\n",now_sec()-t0);

    /* ================================================================
     * MULTI-PLANE VOTING + MT7 DOT + TOPO + BAYESIAN
     * ================================================================ */
    printf("Running multi-plane vote + full stack ranking...\n");
    uint32_t *vbuf=calloc(TRAIN_N,4);
    int nreg=9;
    int plane_weights[]={729,243,81,27,9,3,1}; /* weight each plane by its significance */

    int correct_1plane=0, correct_4plane=0;
    int correct_1plane_stack=0, correct_4plane_stack=0;

    for(int i=0;i<TEST_N;i++){
        /* 4-plane voting: accumulate votes across all planes */
        memset(vbuf,0,TRAIN_N*4);
        for(int pi=0;pi<N_PLANES;pi++){
            const uint8_t*sig=p_joint_te[pi]+(size_t)i*SIG_PAD;
            uint8_t bgv=p_bg[pi];
            int pw=plane_weights[pi];
            for(int k=0;k<N_BLOCKS;k++){
                uint8_t bv=sig[k];if(bv==bgv)continue;
                uint16_t w=(uint16_t)(p_ig[pi][k]*pw),wh=w>1?w/2:1;
                {uint32_t off=p_idx_off[pi][k][bv];uint16_t sz=p_idx_sz[pi][k][bv];
                 for(uint16_t j=0;j<sz;j++)vbuf[p_idx_pool[pi][off+j]]+=w;}
                for(int nb=0;nb<8;nb++){uint8_t nv=p_nbr[bv][nb];if(nv==bgv)continue;
                    uint32_t noff=p_idx_off[pi][k][nv];uint16_t nsz=p_idx_sz[pi][k][nv];
                    for(uint16_t j=0;j<nsz;j++)vbuf[p_idx_pool[pi][noff+j]]+=wh;}
            }
        }
        cand_t cands4[TOP_K];int nc4=topk(vbuf,TRAIN_N,cands4,TOP_K);

        /* Also: single-plane voting (plane 0 only, for comparison) */
        memset(vbuf,0,TRAIN_N*4);
        {const uint8_t*sig=p_joint_te[0]+(size_t)i*SIG_PAD;uint8_t bgv=p_bg[0];
         for(int k=0;k<N_BLOCKS;k++){uint8_t bv=sig[k];if(bv==bgv)continue;
             uint16_t w=p_ig[0][k],wh=w>1?w/2:1;
             {uint32_t off=p_idx_off[0][k][bv];uint16_t sz=p_idx_sz[0][k][bv];
              for(uint16_t j=0;j<sz;j++)vbuf[p_idx_pool[0][off+j]]+=w;}
             for(int nb=0;nb<8;nb++){uint8_t nv=p_nbr[bv][nb];if(nv==bgv)continue;
                 uint32_t noff=p_idx_off[0][k][nv];uint16_t nsz=p_idx_sz[0][k][nv];
                 for(uint16_t j=0;j<nsz;j++)vbuf[p_idx_pool[0][noff+j]]+=wh;}}}
        cand_t cands1[TOP_K];int nc1=topk(vbuf,TRAIN_N,cands1,TOP_K);

        /* Score candidates with MT7 dot + topo + Bayesian */
        /* Bayesian: use plane-0 hot map (coarsest) */
        double bp[N_CLASSES];
        {const uint8_t*sig=p_joint_te[0]+(size_t)i*SIG_PAD;
         for(int c=0;c<N_CLASSES;c++)bp[c]=0.0;
         for(int k=0;k<N_BLOCKS;k++){uint8_t bv=sig[k];if(bv==p_bg[0])continue;
             const uint32_t*h=p_hot[0]+(size_t)k*BYTE_VALS*CLS_PAD+(size_t)bv*CLS_PAD;
             for(int c=0;c<N_CLASSES;c++)bp[c]+=log(h[c]+0.5);}
         double mx=-1e30;for(int c=0;c<N_CLASSES;c++)if(bp[c]>mx)mx=bp[c];
         for(int c=0;c<N_CLASSES;c++)bp[c]-=mx;}

        int16_t q_dn=divneg_te[i],q_dc=divcy_te[i];
        const int16_t*q_gd=gdiv_te+(size_t)i*MAX_REGIONS;

        /* Score and classify: N-plane candidates — dot across all planes */
        for(int j=0;j<nc4;j++){uint32_t id=cands4[j].id;
            int64_t mtd=0;
            for(int pi=0;pi<N_PLANES;pi++){int pw=plane_weights[pi];
                mtd+=(int64_t)pw*(tdot(plane_te[pi]+(size_t)i*PADDED,plane_tr[pi]+(size_t)id*PADDED,PADDED)
                                 +tdot(plane_hg_te[pi]+(size_t)i*PADDED,plane_hg_tr[pi]+(size_t)id*PADDED,PADDED)
                                 +tdot(plane_vg_te[pi]+(size_t)i*PADDED,plane_vg_tr[pi]+(size_t)id*PADDED,PADDED));}
            int32_t mt4d=(int32_t)(mtd/100); /* scale down to avoid overflow */
            int16_t cdn=divneg_tr[id],cdc=divcy_tr[id];int32_t ds=-(int32_t)abs(q_dn-cdn);
            if(q_dc>=0&&cdc>=0)ds-=(int32_t)abs(q_dc-cdc)*2;else if((q_dc<0)!=(cdc<0))ds-=10;
            const int16_t*cg=gdiv_tr+(size_t)id*MAX_REGIONS;int32_t gl=0;for(int r=0;r<nreg;r++)gl+=abs(q_gd[r]-cg[r]);
            uint8_t lbl=train_labels[id];
            cands4[j].score=(int64_t)512*mt4d+(int64_t)200*ds+(int64_t)200*(-gl)+(int64_t)50*(int64_t)(bp[lbl]*1000);}
        qsort(cands4,(size_t)nc4,sizeof(cand_t),cmps);
        if(nc4>0&&train_labels[cands4[0].id]==test_labels[i])correct_4plane_stack++;

        /* k=1 vote-only for 4-plane */
        {uint32_t bid=0,bv2=0;memset(vbuf,0,TRAIN_N*4);
         for(int pi=0;pi<N_PLANES;pi++){const uint8_t*sig=p_joint_te[pi]+(size_t)i*SIG_PAD;uint8_t bgv=p_bg[pi];int pw=plane_weights[pi];
             for(int k=0;k<N_BLOCKS;k++){uint8_t bv=sig[k];if(bv==bgv)continue;uint16_t w=(uint16_t)(p_ig[pi][k]*pw);
                 uint32_t off=p_idx_off[pi][k][bv];uint16_t sz=p_idx_sz[pi][k][bv];
                 for(uint16_t j=0;j<sz;j++)vbuf[p_idx_pool[pi][off+j]]+=w;}}
         for(int j=0;j<TRAIN_N;j++)if(vbuf[j]>bv2){bv2=vbuf[j];bid=(uint32_t)j;}
         if(train_labels[bid]==test_labels[i])correct_4plane++;}

        /* 1-plane vote-only */
        {uint32_t bid=0,bv2=0;for(int j=0;j<TRAIN_N;j++){
             /* reuse cands1 — top-1 by vote */
             if(nc1>0&&cands1[0].votes>bv2){bv2=cands1[0].votes;bid=cands1[0].id;}}
         if(train_labels[cands1[0].id]==test_labels[i])correct_1plane++;}

        /* 1-plane full stack */
        for(int j=0;j<nc1;j++){uint32_t id=cands1[j].id;
            int64_t mtd=0;
            for(int pi=0;pi<N_PLANES;pi++){int pw=plane_weights[pi];
                mtd+=(int64_t)pw*(tdot(plane_te[pi]+(size_t)i*PADDED,plane_tr[pi]+(size_t)id*PADDED,PADDED)
                                 +tdot(plane_hg_te[pi]+(size_t)i*PADDED,plane_hg_tr[pi]+(size_t)id*PADDED,PADDED)
                                 +tdot(plane_vg_te[pi]+(size_t)i*PADDED,plane_vg_tr[pi]+(size_t)id*PADDED,PADDED));}
            int32_t mt4d=(int32_t)(mtd/100);
            int16_t cdn=divneg_tr[id],cdc=divcy_tr[id];int32_t ds=-(int32_t)abs(q_dn-cdn);
            if(q_dc>=0&&cdc>=0)ds-=(int32_t)abs(q_dc-cdc)*2;else if((q_dc<0)!=(cdc<0))ds-=10;
            const int16_t*cg=gdiv_tr+(size_t)id*MAX_REGIONS;int32_t gl=0;for(int r=0;r<nreg;r++)gl+=abs(q_gd[r]-cg[r]);
            uint8_t lbl=train_labels[id];
            cands1[j].score=(int64_t)512*mt4d+(int64_t)200*ds+(int64_t)200*(-gl)+(int64_t)50*(int64_t)(bp[lbl]*1000);}
        qsort(cands1,(size_t)nc1,sizeof(cand_t),cmps);
        if(nc1>0&&train_labels[cands1[0].id]==test_labels[i])correct_1plane_stack++;

        if((i+1)%1000==0)fprintf(stderr,"  %d/%d\r",i+1,TEST_N);
    }
    fprintf(stderr,"\n");free(vbuf);

    printf("\n=== RESULTS (all k=1) ===\n\n");
    printf("  %-45s  Accuracy\n","Method");
    printf("  %-45s  %5.2f%%\n","1-plane vote-only (coarsest ternary)",100.0*correct_1plane/TEST_N);
    printf("  %-45s  %5.2f%%\n","4-plane vote-only (all MT7 planes)",100.0*correct_4plane/TEST_N);
    printf("  %-45s  %5.2f%%\n","1-plane + full stack (current best config)",100.0*correct_1plane_stack/TEST_N);
    printf("  %-45s  %5.2f%%\n","4-plane + full stack (MT7 everywhere)",100.0*correct_4plane_stack/TEST_N);
    printf("\n  Delta (4-plane vs 1-plane, vote-only): %+.2fpp\n",100.0*(correct_4plane-correct_1plane)/TEST_N);
    printf("  Delta (4-plane vs 1-plane, full stack): %+.2fpp\n",100.0*(correct_4plane_stack-correct_1plane_stack)/TEST_N);

    printf("\nTotal: %.1f sec\n",now_sec()-t0);
    return 0;
}
