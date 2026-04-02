/*
 * sstt_topo9_val_5trit.c — 5-Trit (5-Eye) Variant of topo9_val
 *
 * This version uses "5 trits per pixel" by employing an ensemble of 5 
 * different ternary quantization perspectives ("eyes") for retrieval.
 *
 * Eyes:
 *   1. Fixed 85/170 (Standard)
 *   2. Adaptive P33/P67 (Per-image)
 *   3. Fixed 64/192 (Wide zero)
 *   4. Fixed 96/160 (Narrow zero)
 *   5. Adaptive P20/P80 (Wide adaptive)
 *
 * Retrieval: Votes are aggregated across all 5 eyes.
 * Ranking: Topological ranking (topo9) remains anchored to Eye 1.
 *
 * Build: make sstt_topo9_val_5trit
 */

#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define TRAIN_N    60000
#define TEST_N     10000
#define IMG_W      28
#define IMG_H      28
#define PIXELS     784
#define PADDED     800
#define N_CLASSES  10
#define BLKS_PER_ROW 9
#define N_BLOCKS   252
#define SIG_PAD    256
#define BYTE_VALS  256
#define BG_PIXEL 0
#define BG_GRAD 13
#define BG_TRANS 13
#define H_TRANS_PER_ROW 8
#define N_HTRANS (H_TRANS_PER_ROW*IMG_H)
#define V_TRANS_PER_COL 27
#define N_VTRANS (BLKS_PER_ROW*V_TRANS_PER_COL)
#define TRANS_PAD 256
#define IG_SCALE 16
#define TOP_K 200
#define HP_PAD 32
#define FG_W 30
#define FG_H 30
#define FG_SZ (FG_W*FG_H)
#define CENT_BOTH_OPEN 50
#define CENT_DISAGREE 80
#define MAX_REGIONS 16
#define VAL_N      5000
#define HOLDOUT_START 5000
#define MAX_EYES   6

static const char *data_dir="data/";
static double now_sec(void){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return ts.tv_sec+ts.tv_nsec*1e-9;}

static uint8_t *raw_train_img,*raw_test_img,*train_labels,*test_labels;
static int8_t *tern_train,*tern_test,*hgrad_train,*hgrad_test,*vgrad_train,*vgrad_test;
static int16_t *cent_train,*cent_test,*hprof_train,*hprof_test;
static int16_t *divneg_train,*divneg_test,*divneg_cy_train,*divneg_cy_test;
static int16_t *gdiv_2x4_train,*gdiv_2x4_test,*gdiv_3x3_train,*gdiv_3x3_test;

typedef struct {
    uint8_t *joint_tr, *joint_te;
    uint16_t ig_w[N_BLOCKS];
    uint8_t bg;
    uint8_t nbr[BYTE_VALS][8];
    uint32_t idx_off[N_BLOCKS][BYTE_VALS];
    uint16_t idx_sz[N_BLOCKS][BYTE_VALS];
    uint32_t *idx_pool;
} eye_t;

static eye_t eyes[MAX_EYES];
static int *g_hist=NULL;static size_t g_hist_cap=0;

/* === Quantization Eyes === */
static int cmp_u8(const void*a,const void*b){return *(const uint8_t*)a-*(const uint8_t*)b;}

static void quant_eye1(const uint8_t*s,int8_t*d,int n){for(int i=0;i<n;i++){const uint8_t*si=s+(size_t)i*PIXELS;int8_t*di=d+(size_t)i*PADDED;for(int j=0;j<PIXELS;j++)di[j]=si[j]>170?1:si[j]<85?-1:0;}}
static void quant_eye2(const uint8_t*s,int8_t*d,int n){uint8_t*sr=malloc(PIXELS);for(int i=0;i<n;i++){const uint8_t*si=s+(size_t)i*PIXELS;int8_t*di=d+(size_t)i*PADDED;memcpy(sr,si,PIXELS);qsort(sr,PIXELS,1,cmp_u8);uint8_t p33=sr[PIXELS/3],p67=sr[2*PIXELS/3];if(p33==p67){p33=p33>0?p33-1:0;p67=p67<255?p67+1:255;}for(int j=0;j<PIXELS;j++)di[j]=si[j]>p67?1:si[j]<p33?-1:0;}free(sr);}
static void quant_eye3(const uint8_t*s,int8_t*d,int n){for(int i=0;i<n;i++){const uint8_t*si=s+(size_t)i*PIXELS;int8_t*di=d+(size_t)i*PADDED;for(int j=0;j<PIXELS;j++)di[j]=si[j]>192?1:si[j]<64?-1:0;}}
static void quant_eye4(const uint8_t*s,int8_t*d,int n){for(int i=0;i<n;i++){const uint8_t*si=s+(size_t)i*PIXELS;int8_t*di=d+(size_t)i*PADDED;for(int j=0;j<PIXELS;j++)di[j]=si[j]>160?1:si[j]<96?-1:0;}}
static void quant_eye5(const uint8_t*s,int8_t*d,int n){uint8_t*sr=malloc(PIXELS);for(int i=0;i<n;i++){const uint8_t*si=s+(size_t)i*PIXELS;int8_t*di=d+(size_t)i*PADDED;memcpy(sr,si,PIXELS);qsort(sr,PIXELS,1,cmp_u8);uint8_t p20=sr[PIXELS/5],p80=sr[4*PIXELS/5];if(p20==p80){p20=p20>0?p20-1:0;p80=p80<255?p80+1:255;}for(int j=0;j<PIXELS;j++)di[j]=si[j]>p80?1:si[j]<p20?-1:0;}free(sr);}
static void quant_eye6(const uint8_t*s,int8_t*d,int n){uint8_t*sr=malloc(PIXELS);for(int i=0;i<n;i++){const uint8_t*si=s+(size_t)i*PIXELS;int8_t*di=d+(size_t)i*PADDED;memcpy(sr,si,PIXELS);qsort(sr,PIXELS,1,cmp_u8);uint8_t p10=sr[PIXELS/10],p90=sr[PIXELS*9/10];if(p10==p90){p10=p10>0?p10-1:0;p90=p90<255?p90+1:255;}for(int j=0;j<PIXELS;j++)di[j]=si[j]>p90?1:si[j]<p10?-1:0;}free(sr);}

typedef void (*quant_fn)(const uint8_t*,int8_t*,int);
static quant_fn qfns[MAX_EYES] = {quant_eye1, quant_eye2, quant_eye3, quant_eye4, quant_eye5, quant_eye6};

/* === Boilerplate from topo9_val === */
static uint8_t *load_idx(const char*path,uint32_t*cnt,uint32_t*ro,uint32_t*co){FILE*f=fopen(path,"rb");if(!f){fprintf(stderr,"ERR:%s\n",path);exit(1);}uint32_t m,n;if(fread(&m,4,1,f)!=1||fread(&n,4,1,f)!=1){fclose(f);exit(1);}m=__builtin_bswap32(m);n=__builtin_bswap32(n);*cnt=n;size_t s=1;if((m&0xFF)>=3){uint32_t r,c;if(fread(&r,4,1,f)!=1||fread(&c,4,1,f)!=1){fclose(f);exit(1);}r=__builtin_bswap32(r);c=__builtin_bswap32(c);if(ro)*ro=r;if(co)*co=c;s=(size_t)r*c;}else{if(ro)*ro=0;if(co)*co=0;}size_t total=(size_t)n*s;uint8_t*d=malloc(total);if(!d||fread(d,1,total,f)!=total){fclose(f);exit(1);}fclose(f);return d;}
static void load_data(void){uint32_t n,r,c;char p[256];snprintf(p,sizeof(p),"%strain-images-idx3-ubyte",data_dir);raw_train_img=load_idx(p,&n,&r,&c);snprintf(p,sizeof(p),"%strain-labels-idx1-ubyte",data_dir);train_labels=load_idx(p,&n,NULL,NULL);snprintf(p,sizeof(p),"%st10k-images-idx3-ubyte",data_dir);raw_test_img=load_idx(p,&n,&r,&c);snprintf(p,sizeof(p),"%st10k-labels-idx1-ubyte",data_dir);test_labels=load_idx(p,&n,NULL,NULL);}
static inline int8_t clamp_trit(int v){return v>0?1:v<0?-1:0;}
static void gradients(const int8_t*t,int8_t*h,int8_t*v,int n){for(int i=0;i<n;i++){const int8_t*ti=t+(size_t)i*PADDED;int8_t*hi=h+(size_t)i*PADDED;int8_t*vi=v+(size_t)i*PADDED;for(int y=0;y<IMG_H;y++){for(int x=0;x<IMG_W-1;x++)hi[y*IMG_W+x]=clamp_trit(ti[y*IMG_W+x+1]-ti[y*IMG_W+x]);hi[y*IMG_W+IMG_W-1]=0;}memset(hi+PIXELS,0,PADDED-PIXELS);for(int y=0;y<IMG_H-1;y++)for(int x=0;x<IMG_W;x++)vi[y*IMG_W+x]=clamp_trit(ti[(y+1)*IMG_W+x]-ti[y*IMG_W+x]);memset(vi+(IMG_H-1)*IMG_W,0,IMG_W);memset(vi+PIXELS,0,PADDED-PIXELS);}}
static inline uint8_t benc(int8_t a,int8_t b,int8_t c){return(uint8_t)((a+1)*9+(b+1)*3+(c+1));}
static void block_sigs(const int8_t*data,uint8_t*sigs,int n){for(int i=0;i<n;i++){const int8_t*img=data+(size_t)i*PADDED;uint8_t*sig=sigs+(size_t)i*SIG_PAD;for(int y=0;y<IMG_H;y++)for(int s=0;s<BLKS_PER_ROW;s++){int b=y*IMG_W+s*3;sig[y*BLKS_PER_ROW+s]=benc(img[b],img[b+1],img[b+2]);}memset(sig+N_BLOCKS,0xFF,SIG_PAD-N_BLOCKS);}}
static inline uint8_t tenc(uint8_t a,uint8_t b){int8_t a0=(a/9)-1,a1=((a/3)%3)-1,a2=(a%3)-1,b0=(b/9)-1,b1=((b/3)%3)-1,b2=(b%3)-1;return benc(clamp_trit(b0-a0),clamp_trit(b1-a1),clamp_trit(b2-a2));}
static void trans_fn(const uint8_t*bs,int str,uint8_t*ht,uint8_t*vt,int n){for(int i=0;i<n;i++){const uint8_t*s=bs+(size_t)i*str;uint8_t*h=ht+(size_t)i*TRANS_PAD;uint8_t*v=vt+(size_t)i*TRANS_PAD;for(int y=0;y<IMG_H;y++)for(int ss=0;ss<H_TRANS_PER_ROW;ss++)h[y*H_TRANS_PER_ROW+ss]=tenc(s[y*BLKS_PER_ROW+ss],s[y*BLKS_PER_ROW+ss+1]);memset(h+N_HTRANS,0xFF,TRANS_PAD-N_HTRANS);for(int y=0;y<V_TRANS_PER_COL;y++)for(int ss=0;ss<BLKS_PER_ROW;ss++)v[y*BLKS_PER_ROW+ss]=tenc(s[y*BLKS_PER_ROW+ss],s[(y+1)*BLKS_PER_ROW+ss]);memset(v+N_VTRANS,0xFF,TRANS_PAD-N_VTRANS);}}
static inline uint8_t enc_d(uint8_t px,uint8_t hg,uint8_t vg,uint8_t ht,uint8_t vt){int ps=((px/9)-1)+(((px/3)%3)-1)+((px%3)-1),hs=((hg/9)-1)+(((hg/3)%3)-1)+((hg%3)-1),vs=((vg/9)-1)+(((vg/3)%3)-1)+((vg%3)-1);uint8_t pc=ps<0?0:ps==0?1:ps<3?2:3,hc=hs<0?0:hs==0?1:hs<3?2:3,vc=vs<0?0:vs==0?1:vs<3?2:3;return pc|(hc<<2)|(vc<<4)|((ht!=BG_TRANS)?1<<6:0)|((vt!=BG_TRANS)?1<<7:0);}
static void joint_sigs_fn(uint8_t*out,int n,const uint8_t*px,const uint8_t*hg,const uint8_t*vg,const uint8_t*ht,const uint8_t*vt){for(int i=0;i<n;i++){const uint8_t*pi=px+(size_t)i*SIG_PAD,*hi=hg+(size_t)i*SIG_PAD,*vi=vg+(size_t)i*SIG_PAD;const uint8_t*hti=ht+(size_t)i*TRANS_PAD,*vti=vt+(size_t)i*TRANS_PAD;uint8_t*oi=out+(size_t)i*SIG_PAD;for(int y=0;y<IMG_H;y++)for(int s=0;s<BLKS_PER_ROW;s++){int k=y*BLKS_PER_ROW+s;uint8_t htb=s>0?hti[y*H_TRANS_PER_ROW+(s-1)]:BG_TRANS;uint8_t vtb=y>0?vti[(y-1)*BLKS_PER_ROW+s]:BG_TRANS;oi[k]=enc_d(pi[k],hi[k],vi[k],htb,vtb);}memset(oi+N_BLOCKS,0xFF,SIG_PAD-N_BLOCKS);}}

static void compute_ig(const uint8_t*sigs,int nv,uint8_t bgv,uint16_t*ig_out){int cc[N_CLASSES]={0};for(int i=0;i<TRAIN_N;i++)cc[train_labels[i]]++;double hc=0;for(int c=0;c<N_CLASSES;c++){double p=(double)cc[c]/TRAIN_N;if(p>0)hc-=p*log2(p);}double raw[N_BLOCKS],mx=0;for(int k=0;k<N_BLOCKS;k++){int*cnt=calloc((size_t)nv*N_CLASSES,sizeof(int));int*vt=calloc(nv,sizeof(int));for(int i=0;i<TRAIN_N;i++){int v=sigs[(size_t)i*SIG_PAD+k];cnt[v*N_CLASSES+train_labels[i]]++;vt[v]++;}double hcond=0;for(int v=0;v<nv;v++){if(!vt[v]||v==(int)bgv)continue;double pv=(double)vt[v]/TRAIN_N,hv=0;for(int c=0;c<N_CLASSES;c++){double pc=(double)cnt[v*N_CLASSES+c]/vt[v];if(pc>0)hv-=pc*log2(pc);}hcond+=pv*hv;}raw[k]=hc-hcond;if(raw[k]>mx)mx=raw[k];free(cnt);free(vt);}for(int k=0;k<N_BLOCKS;k++){ig_out[k]=mx>0?(uint16_t)(raw[k]/mx*IG_SCALE+0.5):1;if(!ig_out[k])ig_out[k]=1;}}

static void build_eye_infrastructure(eye_t*e, int8_t*t_tr, int8_t*t_te){
    int8_t *h_tr=aligned_alloc(32,(size_t)TRAIN_N*PADDED),*h_te=aligned_alloc(32,(size_t)TEST_N*PADDED);
    int8_t *v_tr=aligned_alloc(32,(size_t)TRAIN_N*PADDED),*v_te=aligned_alloc(32,(size_t)TEST_N*PADDED);
    gradients(t_tr,h_tr,v_tr,TRAIN_N);gradients(t_te,h_te,v_te,TEST_N);
    uint8_t *px_tr=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD),*px_te=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    uint8_t *hg_tr=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD),*hg_te=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    uint8_t *vg_tr=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD),*vg_te=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    block_sigs(t_tr,px_tr,TRAIN_N);block_sigs(t_te,px_te,TEST_N);
    block_sigs(h_tr,hg_tr,TRAIN_N);block_sigs(h_te,hg_te,TEST_N);
    block_sigs(v_tr,vg_tr,TRAIN_N);block_sigs(v_te,vg_te,TEST_N);
    uint8_t *ht_tr=aligned_alloc(32,(size_t)TRAIN_N*TRANS_PAD),*ht_te=aligned_alloc(32,(size_t)TEST_N*TRANS_PAD);
    uint8_t *vt_tr=aligned_alloc(32,(size_t)TRAIN_N*TRANS_PAD),*vt_te=aligned_alloc(32,(size_t)TEST_N*TRANS_PAD);
    trans_fn(px_tr,SIG_PAD,ht_tr,vt_tr,TRAIN_N);trans_fn(px_te,SIG_PAD,ht_te,vt_te,TEST_N);
    e->joint_tr=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD);e->joint_te=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    joint_sigs_fn(e->joint_tr,TRAIN_N,px_tr,hg_tr,vg_tr,ht_tr,vt_tr);
    joint_sigs_fn(e->joint_te,TEST_N,px_te,hg_te,vg_te,ht_te,vt_te);
    long vc[BYTE_VALS]={0};for(int i=0;i<TRAIN_N;i++){const uint8_t*s=e->joint_tr+(size_t)i*SIG_PAD;for(int k=0;k<N_BLOCKS;k++)vc[s[k]]++;}
    e->bg=0;long mc=0;for(int v=0;v<BYTE_VALS;v++)if(vc[v]>mc){mc=vc[v];e->bg=(uint8_t)v;}
    compute_ig(e->joint_tr,BYTE_VALS,e->bg,e->ig_w);
    for(int v=0;v<BYTE_VALS;v++)for(int b=0;b<8;b++)e->nbr[v][b]=(uint8_t)(v^(1<<b));
    memset(e->idx_sz,0,sizeof(e->idx_sz));
    for(int i=0;i<TRAIN_N;i++){const uint8_t*s=e->joint_tr+(size_t)i*SIG_PAD;for(int k=0;k<N_BLOCKS;k++)if(s[k]!=e->bg)e->idx_sz[k][s[k]]++;}
    uint32_t tot=0;for(int k=0;k<N_BLOCKS;k++)for(int v=0;v<BYTE_VALS;v++){e->idx_off[k][v]=tot;tot+=e->idx_sz[k][v];}
    e->idx_pool=malloc((size_t)tot*sizeof(uint32_t));uint32_t(*wp)[BYTE_VALS]=malloc((size_t)N_BLOCKS*BYTE_VALS*4);memcpy(wp,e->idx_off,(size_t)N_BLOCKS*BYTE_VALS*4);
    for(int i=0;i<TRAIN_N;i++){const uint8_t*s=e->joint_tr+(size_t)i*SIG_PAD;for(int k=0;k<N_BLOCKS;k++)if(s[k]!=e->bg)e->idx_pool[wp[k][s[k]]++]=(uint32_t)i;}
    free(wp);free(h_tr);free(h_te);free(v_tr);free(v_te);free(px_tr);free(px_te);free(hg_tr);free(hg_te);free(vg_tr);free(vg_te);free(ht_tr);free(ht_te);free(vt_tr);free(vt_te);
}

static void vote_all_eyes(uint32_t*votes, int img){
    memset(votes,0,TRAIN_N*sizeof(uint32_t));
    for(int ei=0; ei<MAX_EYES; ei++){
        eye_t *e = &eyes[ei];
        const uint8_t*sig=e->joint_te+(size_t)img*SIG_PAD;
        for(int k=0;k<N_BLOCKS;k++){
            uint8_t bv=sig[k];if(bv==e->bg)continue;
            uint16_t w=e->ig_w[k],wh=w>1?w/2:1;
            {uint32_t off=e->idx_off[k][bv];uint16_t sz=e->idx_sz[k][bv];const uint32_t*ids=e->idx_pool+off;for(uint16_t j=0;j<sz;j++)votes[ids[j]]+=w;}
            for(int nb=0;nb<8;nb++){
                uint8_t nv=e->nbr[bv][nb];if(nv==e->bg)continue;
                uint32_t noff=e->idx_off[k][nv];uint16_t nsz=e->idx_sz[k][nv];const uint32_t*nids=e->idx_pool+noff;for(uint16_t j=0;j<nsz;j++)votes[nids[j]]+=wh;
            }
        }
    }
}

/* === Topological Features (Anchored to Eye 1) === */
static int16_t enclosed_centroid(const int8_t*tern){uint8_t grid[FG_SZ];memset(grid,0,sizeof(grid));for(int y=0;y<IMG_H;y++)for(int x=0;x<IMG_W;x++)grid[(y+1)*FG_W+(x+1)]=(tern[y*IMG_W+x]>0)?1:0;uint8_t visited[FG_SZ];memset(visited,0,sizeof(visited));int stack[FG_SZ];int sp=0;for(int y=0;y<FG_H;y++)for(int x=0;x<FG_W;x++){if(y==0||y==FG_H-1||x==0||x==FG_W-1){int pos=y*FG_W+x;if(!grid[pos]&&!visited[pos]){visited[pos]=1;stack[sp++]=pos;}}}while(sp>0){int pos=stack[--sp];int py=pos/FG_W,px2=pos%FG_W;const int dx[]={0,0,1,-1},dy[]={1,-1,0,0};for(int d=0;d<4;d++){int ny=py+dy[d],nx=px2+dx[d];if(ny<0||ny>=FG_H||nx<0||nx>=FG_W)continue;int npos=ny*FG_W+nx;if(!visited[npos]&&!grid[npos]){visited[npos]=1;stack[sp++]=npos;}}}int sum_y=0,count=0;for(int y=1;y<FG_H-1;y++)for(int x=1;x<FG_W-1;x++){int pos=y*FG_W+x;if(!grid[pos]&&!visited[pos]){sum_y+=(y-1);count++;}}return count>0?(int16_t)(sum_y/count):-1;}
static void compute_centroid_all(const int8_t*tern,int16_t*out,int n){for(int i=0;i<n;i++)out[i]=enclosed_centroid(tern+(size_t)i*PADDED);}
static void hprofile(const int8_t*tern,int16_t*prof){for(int y=0;y<IMG_H;y++){int c=0;for(int x=0;x<IMG_W;x++)c+=(tern[y*IMG_W+x]>0);prof[y]=(int16_t)c;}for(int y=IMG_H;y<HP_PAD;y++)prof[y]=0;}
static void compute_hprof_all(const int8_t*tern,int16_t*out,int n){for(int i=0;i<n;i++)hprofile(tern+(size_t)i*PADDED,out+(size_t)i*HP_PAD);}
static void div_features(const int8_t*hg,const int8_t*vg,int16_t*ns,int16_t*cy){int neg_sum=0,neg_ysum=0,neg_cnt=0;for(int y=0;y<IMG_H;y++)for(int x=0;x<IMG_W;x++){int dh=(int)hg[y*IMG_W+x]-(x>0?(int)hg[y*IMG_W+x-1]:0);int dv=(int)vg[y*IMG_W+x]-(y>0?(int)vg[(y-1)*IMG_W+x]:0);int d=dh+dv;if(d<0){neg_sum+=d;neg_ysum+=y;neg_cnt++;}}*ns=(int16_t)(neg_sum<-32767?-32767:neg_sum);*cy=neg_cnt>0?(int16_t)(neg_ysum/neg_cnt):-1;}
static void compute_div_all(const int8_t*hg,const int8_t*vg,int16_t*ns,int16_t*cy,int n){for(int i=0;i<n;i++)div_features(hg+(size_t)i*PADDED,vg+(size_t)i*PADDED,&ns[i],&cy[i]);}
static inline int32_t tdot(const int8_t*a,const int8_t*b){__m256i acc=_mm256_setzero_si256();for(int i=0;i<PADDED;i+=32)acc=_mm256_add_epi8(acc,_mm256_sign_epi8(_mm256_load_si256((const __m256i*)(a+i)),_mm256_load_si256((const __m256i*)(b+i))));__m256i lo=_mm256_cvtepi8_epi16(_mm256_castsi256_si128(acc)),hi=_mm256_cvtepi8_epi16(_mm256_extracti128_si256(acc,1));__m256i s32=_mm256_madd_epi16(_mm256_add_epi16(lo,hi),_mm256_set1_epi16(1));__m128i s=_mm_add_epi32(_mm256_castsi256_si128(s32),_mm256_extracti128_si256(s32,1));s=_mm_add_epi32(s,_mm_shuffle_epi32(s,_MM_SHUFFLE(1,0,3,2)));s=_mm_add_epi32(s,_mm_shuffle_epi32(s,_MM_SHUFFLE(2,3,0,1)));return _mm_cvtsi128_si32(s);}

/* Grid divergence */
static void grid_div(const int8_t*hg,const int8_t*vg,int grow,int gcol,int16_t*out){
    int nr=grow*gcol;int regions[MAX_REGIONS];memset(regions,0,sizeof(int)*nr);
    for(int y=0;y<IMG_H;y++)for(int x=0;x<IMG_W;x++){int dh=(int)hg[y*IMG_W+x]-(x>0?(int)hg[y*IMG_W+x-1]:0);int dv=(int)vg[y*IMG_W+x]-(y>0?(int)vg[(y-1)*IMG_W+x]:0);int d=dh+dv;if(d<0){int ry=y*grow/IMG_H;int rx=x*gcol/IMG_W;if(ry>=grow)ry=grow-1;if(rx>=gcol)rx=gcol-1;regions[ry*gcol+rx]+=d;}}
    for(int i=0;i<nr;i++)out[i]=(int16_t)(regions[i]<-32767?-32767:regions[i]);
}
static void compute_grid_div(const int8_t*hg,const int8_t*vg,int grow,int gcol,int16_t*out,int n){for(int i=0;i<n;i++)grid_div(hg+(size_t)i*PADDED,vg+(size_t)i*PADDED,grow,gcol,out+(size_t)i*MAX_REGIONS);}

/* Candidate Struct */
typedef struct {
    uint32_t id, votes;
    int32_t dot_px, dot_vg;
    int32_t div_sim, cent_sim, prof_sim;
    int32_t gdiv_sim;
    int64_t combined;
} cand_t;

static int cmp_votes_d(const void*a,const void*b){return(int)((const cand_t*)b)->votes-(int)((const cand_t*)a)->votes;}
static int cmp_comb_d(const void*a,const void*b){int64_t da=((const cand_t*)a)->combined,db=((const cand_t*)b)->combined;return(db>da)-(db<da);}
static int select_top_k(const uint32_t*votes,int n,cand_t*out,int k){uint32_t mx=0;for(int i=0;i<n;i++)if(votes[i]>mx)mx=votes[i];if(!mx)return 0;if((size_t)(mx+1)>g_hist_cap){g_hist_cap=(size_t)(mx+1)+4096;free(g_hist);g_hist=malloc(g_hist_cap*sizeof(int));}memset(g_hist,0,(mx+1)*sizeof(int));for(int i=0;i<n;i++)if(votes[i])g_hist[votes[i]]++;int cum=0,thr;for(thr=(int)mx;thr>=1;thr--){cum+=g_hist[thr];if(cum>=k)break;}if(thr<1)thr=1;int nc=0;for(int i=0;i<n&&nc<k;i++)if(votes[i]>=(uint32_t)thr){out[nc]=(cand_t){0};out[nc].id=(uint32_t)i;out[nc].votes=votes[i];nc++;}qsort(out,(size_t)nc,sizeof(cand_t),cmp_votes_d);return nc;}

static void compute_base(cand_t*cands,int nc,int ti,int16_t q_cent,const int16_t*q_prof,int16_t q_dn,int16_t q_dc){
    const int8_t*tp=tern_test+(size_t)ti*PADDED,*tv=vgrad_test+(size_t)ti*PADDED;
    for(int j=0;j<nc;j++){uint32_t id=cands[j].id;
        cands[j].dot_px=tdot(tp,tern_train+(size_t)id*PADDED);
        cands[j].dot_vg=tdot(tv,vgrad_train+(size_t)id*PADDED);
        int16_t cc=cent_train[id];if(q_cent<0&&cc<0)cands[j].cent_sim=CENT_BOTH_OPEN;else if(q_cent<0||cc<0)cands[j].cent_sim=-CENT_DISAGREE;else cands[j].cent_sim=-(int32_t)abs(q_cent-cc);
        const int16_t*cp=hprof_train+(size_t)id*HP_PAD;int32_t pd=0;for(int y=0;y<IMG_H;y++)pd+=(int32_t)q_prof[y]*cp[y];cands[j].prof_sim=pd;
        int16_t cdn=divneg_train[id],cdc=divneg_cy_train[id];int32_t ds=-(int32_t)abs(q_dn-cdn);if(q_dc>=0&&cdc>=0)ds-=(int32_t)abs(q_dc-cdc)*2;else if((q_dc<0)!=(cdc<0))ds-=10;cands[j].div_sim=ds;
    }
}

static int knn_vote(const cand_t*c,int nc,int k){int v[N_CLASSES]={0};if(k>nc)k=nc;for(int i=0;i<k;i++)v[train_labels[c[i].id]]++;int best=0;for(int c2=1;c2<N_CLASSES;c2++)if(v[c2]>v[best])best=c2;return best;}
static int cmp_i16(const void*a,const void*b){return *(const int16_t*)a-*(const int16_t*)b;}
static int compute_mad(const cand_t*cands,int nc){if(nc<3)return 1000;int16_t vals[TOP_K];for(int j=0;j<nc;j++)vals[j]=divneg_train[cands[j].id];qsort(vals,(size_t)nc,sizeof(int16_t),cmp_i16);int16_t med=vals[nc/2];int16_t devs[TOP_K];for(int j=0;j<nc;j++)devs[j]=(int16_t)abs(vals[j]-med);qsort(devs,(size_t)nc,sizeof(int16_t),cmp_i16);return devs[nc/2]>0?devs[nc/2]:1;}

/* === Ranking Pipelines (Same as topo9_val) === */
static int run_static(const cand_t*pre,const int*nc_arr,const int16_t*qg,const int16_t*tg,int nreg,int w_c,int w_p,int w_d,int w_g,int sc,int start,int end,uint8_t*preds){
    int correct=0;for(int i=start;i<end;i++){cand_t cands[TOP_K];int nc=nc_arr[i];memcpy(cands,pre+(size_t)i*TOP_K,(size_t)nc*sizeof(cand_t));const int16_t*qi=qg+(size_t)i*MAX_REGIONS;for(int j=0;j<nc;j++){const int16_t*ci=tg+(size_t)cands[j].id*MAX_REGIONS;int32_t l=0;for(int r=0;r<nreg;r++)l+=abs(qi[r]-ci[r]);cands[j].gdiv_sim=-l;}int mad=compute_mad(cands,nc);int64_t s=(int64_t)sc;int wc=(int)(s*w_c/(s+mad)),wp=(int)(s*w_p/(s+mad)),wd=(int)(s*w_d/(s+mad)),wg=(int)(s*w_g/(s+mad));for(int j=0;j<nc;j++)cands[j].combined=(int64_t)256*cands[j].dot_px+(int64_t)192*cands[j].dot_vg+(int64_t)wc*cands[j].cent_sim+(int64_t)wp*cands[j].prof_sim+(int64_t)wd*cands[j].div_sim+(int64_t)wg*cands[j].gdiv_sim;qsort(cands,(size_t)nc,sizeof(cand_t),cmp_comb_d);int pred=knn_vote(cands,nc,3);if(preds)preds[i]=(uint8_t)pred;if(pred==test_labels[i])correct++;}return correct;
}

static void report(const uint8_t*preds,const char*label,int correct,int start,int end){
    int n=end-start;printf("--- %s ---\n",label);printf("  Accuracy: %.2f%%  (%d/%d correct, %d errors)\n",100.0*correct/n,correct,n,n-correct);
    int conf[N_CLASSES][N_CLASSES];memset(conf,0,sizeof(conf));for(int i=start;i<end;i++)conf[test_labels[i]][preds[i]]++;
    typedef struct{int a,b,count;}pair_t;pair_t pairs[45];int np=0;for(int a=0;a<N_CLASSES;a++)for(int b=a+1;b<N_CLASSES;b++)pairs[np++]=(pair_t){a,b,conf[a][b]+conf[b][a]};
    for(int i=0;i<np-1;i++)for(int j=i+1;j<np;j++)if(pairs[j].count>pairs[i].count){pair_t t2=pairs[i];pairs[i]=pairs[j];pairs[j]=t2;}
    printf("  Top-5:");for(int i=0;i<5&&i<np;i++)printf("  %d<->%d:%d",pairs[i].a,pairs[i].b,pairs[i].count);printf("\n\n");
}

int main(int argc,char**argv){
    double t0=now_sec();
    if(argc>1){data_dir=argv[1];size_t l=strlen(data_dir);if(l&&data_dir[l-1]!='/'){char*buf=malloc(l+2);memcpy(buf,data_dir,l);buf[l]='/';buf[l+1]='\0';data_dir=buf;}}
    int is_fashion = strstr(data_dir,"fashion") != NULL;
    printf("=== SSTT Topo9-Val-5Trit: 5-Eye Retrieval Variant (%s) ===\n\n", is_fashion?"Fashion-MNIST":"MNIST");

    load_data();
    tern_train=aligned_alloc(32,(size_t)TRAIN_N*PADDED);tern_test=aligned_alloc(32,(size_t)TEST_N*PADDED);
    hgrad_train=aligned_alloc(32,(size_t)TRAIN_N*PADDED);hgrad_test=aligned_alloc(32,(size_t)TEST_N*PADDED);
    vgrad_train=aligned_alloc(32,(size_t)TRAIN_N*PADDED);vgrad_test=aligned_alloc(32,(size_t)TEST_N*PADDED);

    /* Eye 1 is the reference for topological features */
    qfns[0](raw_train_img,tern_train,TRAIN_N);qfns[0](raw_test_img,tern_test,TEST_N);
    gradients(tern_train,hgrad_train,vgrad_train,TRAIN_N);gradients(tern_test,hgrad_test,vgrad_test,TEST_N);

    printf("Building 5-Eye index...\n");
    for(int ei=0; ei<MAX_EYES; ei++){
        printf("  Eye %d: ", ei+1); fflush(stdout);
        int8_t *t_tr=aligned_alloc(32,(size_t)TRAIN_N*PADDED), *t_te=aligned_alloc(32,(size_t)TEST_N*PADDED);
        qfns[ei](raw_train_img, t_tr, TRAIN_N); qfns[ei](raw_test_img, t_te, TEST_N);
        build_eye_infrastructure(&eyes[ei], t_tr, t_te);
        free(t_tr); free(t_te);
        printf("done\n");
    }

    printf("Computing features (Eye 1 anchored)...\n");
    cent_train=malloc((size_t)TRAIN_N*2);cent_test=malloc((size_t)TEST_N*2);
    compute_centroid_all(tern_train,cent_train,TRAIN_N);compute_centroid_all(tern_test,cent_test,TEST_N);
    hprof_train=aligned_alloc(32,(size_t)TRAIN_N*HP_PAD*2);hprof_test=aligned_alloc(32,(size_t)TEST_N*HP_PAD*2);
    compute_hprof_all(tern_train,hprof_train,TRAIN_N);compute_hprof_all(tern_test,hprof_test,TEST_N);
    divneg_train=malloc((size_t)TRAIN_N*2);divneg_test=malloc((size_t)TEST_N*2);
    divneg_cy_train=malloc((size_t)TRAIN_N*2);divneg_cy_test=malloc((size_t)TEST_N*2);
    compute_div_all(hgrad_train,vgrad_train,divneg_train,divneg_cy_train,TRAIN_N);
    compute_div_all(hgrad_test,vgrad_test,divneg_test,divneg_cy_test,TEST_N);
    gdiv_2x4_train=malloc((size_t)TRAIN_N*MAX_REGIONS*2);gdiv_2x4_test=malloc((size_t)TEST_N*MAX_REGIONS*2);
    compute_grid_div(hgrad_train,vgrad_train,2,4,gdiv_2x4_train,TRAIN_N);
    compute_grid_div(hgrad_test,vgrad_test,2,4,gdiv_2x4_test,TEST_N);
    gdiv_3x3_train=malloc((size_t)TRAIN_N*MAX_REGIONS*2);gdiv_3x3_test=malloc((size_t)TEST_N*MAX_REGIONS*2);
    compute_grid_div(hgrad_train,vgrad_train,3,3,gdiv_3x3_train,TRAIN_N);
    compute_grid_div(hgrad_test,vgrad_test,3,3,gdiv_3x3_test,TEST_N);

    printf("Precomputing candidates (aggregated votes)...\n");
    cand_t*pre=malloc((size_t)TEST_N*TOP_K*sizeof(cand_t));
    int*nc_arr=malloc((size_t)TEST_N*sizeof(int));
    uint32_t*votes=calloc(TRAIN_N,sizeof(uint32_t));
    for(int i=0;i<TEST_N;i++){
        vote_all_eyes(votes,i);
        cand_t*ci=pre+(size_t)i*TOP_K;
        int nc=select_top_k(votes,TRAIN_N,ci,TOP_K);
        compute_base(ci,nc,i,cent_test[i],hprof_test+(size_t)i*HP_PAD,divneg_test[i],divneg_cy_test[i]);
        nc_arr[i]=nc;if((i+1)%2000==0)fprintf(stderr,"  %d/%d\r",i+1,TEST_N);
    }
    fprintf(stderr,"\n");free(votes);

    printf("=== EVALUATION (Holdout Split) ===\n");
    uint8_t*preds=calloc(TEST_N,1);
    int16_t *qg,*tg; int nreg,wc,wp,wd,wg,sc;
    int best_grid = is_fashion;
    if(best_grid){ qg=gdiv_3x3_test; tg=gdiv_3x3_train; nreg=9; }
    else { qg=gdiv_2x4_test; tg=gdiv_2x4_train; nreg=8; }

    printf("Grid searching weights on Val...\n");
    int wc_v[]={0,25,50,100}, wp_v[]={0,8,16}, wd_v[]={50,100,200,400}, wg_v[]={50,100,200,400}, sc_v[]={20,50,100};
    int best_c=0,b_wc=25,b_wp=0,b_wd=200,b_wg=100,b_sc=50;
    for(int ci=0;ci<4;ci++)for(int pi=0;pi<3;pi++)for(int di=0;di<4;di++)for(int gvi=0;gvi<4;gvi++)for(int si=0;si<3;si++){
        int c=run_static(pre,nc_arr,qg,tg,nreg,wc_v[ci],wp_v[pi],wd_v[di],wg_v[gvi],sc_v[si],0,VAL_N,NULL);
        if(c>best_c){best_c=c;b_wc=wc_v[ci];b_wp=wp_v[pi];b_wd=wd_v[di];b_wg=wg_v[gvi];b_sc=sc_v[si];}}
    
    printf("Best (Val): wc=%d wp=%d wd=%d wg=%d sc=%d -> %.2f%%\n",b_wc,b_wp,b_wd,b_wg,b_sc,100.0*best_c/VAL_N);

    int c_hold = run_static(pre,nc_arr,qg,tg,nreg,b_wc,b_wp,b_wd,b_wg,b_sc,HOLDOUT_START,TEST_N,preds);
    report(preds,"5-Trit Topo9 Static (Holdout)",c_hold,HOLDOUT_START,TEST_N);

    printf("Total: %.2f sec\n",now_sec()-t0);
    return 0;
}
