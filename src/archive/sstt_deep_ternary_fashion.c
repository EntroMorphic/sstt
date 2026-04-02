/*
 * sstt_deep_ternary_fashion.c — Deep 5-Trit Pipeline for Fashion-MNIST
 *
 * 1. RETRIEVAL: 5-Eye Ensemble (optimized for clothing intensity range).
 *    Eyes: 85/170, P33/67, P20/P80, P40/P60, 64/192.
 * 2. RANKING: Deep Multi-Channel Dot Product (sum of all 5 eye dots).
 * 3. TOPO: Divergence + Centroids (Anchored to Eye 1).
 * 4. VALIDATION: Full Grid Search for weights on VAL split.
 *
 * Build: make build/sstt_deep_ternary_fashion
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
#define PIXELS     784
#define PADDED     800
#define N_CLASSES  10
#define BLKS_PER_ROW 9
#define N_BLOCKS   252
#define SIG_PAD    256
#define BYTE_VALS  256
#define IG_SCALE   16
#define TOP_K      200
#define VAL_N      5000
#define HOLDOUT_START 5000
#define MAX_EYES   5
#define MAX_REGIONS 16
#define TRANS_PAD 256
#define BG_TRANS 13
#define FG_SZ 900
#define FG_W 30
#define FG_H 30
#define HP_PAD 32
#define CENT_BOTH_OPEN 50
#define CENT_DISAGREE 80

static const char *data_dir="data-fashion/";
static double now_sec(void){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return ts.tv_sec+ts.tv_nsec*1e-9;}

static uint8_t *raw_train_img,*raw_test_img,*train_labels,*test_labels;

/* 5-Channel Ternary Data */
static int8_t *tern_train[MAX_EYES], *tern_test[MAX_EYES];
static int8_t *vgrad_train[MAX_EYES], *vgrad_test[MAX_EYES];
static int8_t *hgrad_train[MAX_EYES], *hgrad_test[MAX_EYES];

/* Topo features (Eye 1 anchored) */
static int16_t *cent_train,*cent_test,*hprof_train,*hprof_test;
static int16_t *divneg_train,*divneg_test,*divneg_cy_train,*divneg_cy_test;
static int16_t *gdiv_3x3_train,*gdiv_3x3_test;

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
static void quant_eye3(const uint8_t*s,int8_t*d,int n){uint8_t*sr=malloc(PIXELS);for(int i=0;i<n;i++){const uint8_t*si=s+(size_t)i*PIXELS;int8_t*di=d+(size_t)i*PADDED;memcpy(sr,si,PIXELS);qsort(sr,PIXELS,1,cmp_u8);uint8_t p20=sr[PIXELS/5],p80=sr[4*PIXELS/5];if(p20==p80){p20=p20>0?p20-1:0;p80=p80<255?p80+1:255;}for(int j=0;j<PIXELS;j++)di[j]=si[j]>p80?1:si[j]<p20?-1:0;}free(sr);}
static void quant_eye4(const uint8_t*s,int8_t*d,int n){uint8_t*sr=malloc(PIXELS);for(int i=0;i<n;i++){const uint8_t*si=s+(size_t)i*PIXELS;int8_t*di=d+(size_t)i*PADDED;memcpy(sr,si,PIXELS);qsort(sr,PIXELS,1,cmp_u8);uint8_t p40=sr[PIXELS*40/100],p60=sr[PIXELS*60/100];if(p40==p60){p40=p40>0?p40-1:0;p60=p60<255?p60+1:255;}for(int j=0;j<PIXELS;j++)di[j]=si[j]>p60?1:si[j]<p40?-1:0;}free(sr);}
static void quant_eye5(const uint8_t*s,int8_t*d,int n){for(int i=0;i<n;i++){const uint8_t*si=s+(size_t)i*PIXELS;int8_t*di=d+(size_t)i*PADDED;for(int j=0;j<PIXELS;j++)di[j]=si[j]>192?1:si[j]<64?-1:0;}}

typedef void (*quant_fn)(const uint8_t*,int8_t*,int);
static quant_fn qfns[MAX_EYES] = {quant_eye1, quant_eye2, quant_eye3, quant_eye4, quant_eye5};

/* === Boilerplate === */
static uint8_t *load_idx(const char*path,uint32_t*cnt,uint32_t*ro,uint32_t*co){FILE*f=fopen(path,"rb");if(!f){fprintf(stderr,"ERR:%s\n",path);exit(1);}uint32_t m,n;if(fread(&m,4,1,f)!=1||fread(&n,4,1,f)!=1){fclose(f);exit(1);}m=__builtin_bswap32(m);n=__builtin_bswap32(n);*cnt=n;size_t s=1;if((m&0xFF)>=3){uint32_t r,c;if(fread(&r,4,1,f)!=1||fread(&c,4,1,f)!=1){fclose(f);exit(1);}r=__builtin_bswap32(r);c=__builtin_bswap32(c);if(ro)*ro=r;if(co)*co=c;s=(size_t)r*c;}else{if(ro)*ro=0;if(co)*co=0;}size_t total=(size_t)n*s;uint8_t*d=malloc(total);if(!d||fread(d,1,total,f)!=total){fclose(f);exit(1);}fclose(f);return d;}
static void load_data(void){uint32_t n,r,c;char p[256];snprintf(p,sizeof(p),"%strain-images-idx3-ubyte",data_dir);raw_train_img=load_idx(p,&n,&r,&c);snprintf(p,sizeof(p),"%strain-labels-idx1-ubyte",data_dir);train_labels=load_idx(p,&n,NULL,NULL);snprintf(p,sizeof(p),"%st10k-images-idx3-ubyte",data_dir);raw_test_img=load_idx(p,&n,&r,&c);snprintf(p,sizeof(p),"%st10k-labels-idx1-ubyte",data_dir);test_labels=load_idx(p,&n,NULL,NULL);}
static inline int8_t clamp_trit(int v){return v>0?1:v<0?-1:0;}
static void gradients(const int8_t*t,int8_t*h,int8_t*v,int n){for(int i=0;i<n;i++){const int8_t*ti=t+(size_t)i*PADDED;int8_t*hi=h+(size_t)i*PADDED;int8_t*vi=v+(size_t)i*PADDED;for(int y=0;y<28;y++){for(int x=0;x<27;x++)hi[y*28+x]=clamp_trit(ti[y*28+x+1]-ti[y*28+x]);hi[y*28+27]=0;}for(int y=0;y<27;y++)for(int x=0;x<28;x++)vi[y*28+x]=clamp_trit(ti[(y+1)*28+x]-ti[y*28+x]);memset(vi+27*28,0,28);}}
static inline uint8_t benc(int8_t a,int8_t b,int8_t c){return(uint8_t)((a+1)*9+(b+1)*3+(c+1));}
static void block_sigs(const int8_t*data,uint8_t*sigs,int n){for(int i=0;i<n;i++){const int8_t*img=data+(size_t)i*PADDED;uint8_t*sig=sigs+(size_t)i*SIG_PAD;for(int y=0;y<28;y++)for(int s=0;s<9;s++){int b=y*28+s*3;sig[y*9+s]=benc(img[b],img[b+1],img[b+2]);}}}
static inline uint8_t tenc(uint8_t a,uint8_t b){int8_t a0=(a/9)-1,a1=((a/3)%3)-1,a2=(a%3)-1,b0=(b/9)-1,b1=((b/3)%3)-1,b2=(b%3)-1;return benc(clamp_trit(b0-a0),clamp_trit(b1-a1),clamp_trit(b2-a2));}
static void trans_fn(const uint8_t*bs,uint8_t*ht,uint8_t*vt,int n){for(int i=0;i<n;i++){const uint8_t*s=bs+(size_t)i*SIG_PAD;uint8_t*h=ht+(size_t)i*TRANS_PAD,*v=vt+(size_t)i*TRANS_PAD;for(int y=0;y<28;y++)for(int ss=0;ss<8;ss++)h[y*8+ss]=tenc(s[y*9+ss],s[y*9+ss+1]);for(int y=0;y<27;y++)for(int ss=0;ss<9;ss++)v[y*9+ss]=tenc(s[y*9+ss],s[(y+1)*9+ss]);}}
static inline uint8_t enc_d(uint8_t px,uint8_t hg,uint8_t vg,uint8_t ht,uint8_t vt){int ps=((px/9)-1)+(((px/3)%3)-1)+((px%3)-1),hs=((hg/9)-1)+(((hg/3)%3)-1)+((hg%3)-1),vs=((vg/9)-1)+(((vg/3)%3)-1)+((vg%3)-1);uint8_t pc=ps<0?0:ps==0?1:ps<3?2:3,hc=hs<0?0:hs==0?1:hs<3?2:3,vc=vs<0?0:vs==0?1:vs<3?2:3;return pc|(hc<<2)|(vc<<4)|((ht!=BG_TRANS)?1<<6:0)|((vt!=BG_TRANS)?1<<7:0);}
static void joint_sigs_fn(uint8_t*out,int n,const uint8_t*px,const uint8_t*hg,const uint8_t*vg,const uint8_t*ht,const uint8_t*vt){for(int i=0;i<n;i++){const uint8_t*pi=px+(size_t)i*SIG_PAD,*hi=hg+(size_t)i*SIG_PAD,*vi=vg+(size_t)i*SIG_PAD,*hti=ht+(size_t)i*TRANS_PAD,*vti=vt+(size_t)i*TRANS_PAD;uint8_t*oi=out+(size_t)i*SIG_PAD;for(int y=0;y<28;y++)for(int s=0;s<9;s++){int k=y*9+s;uint8_t htb=s>0?hti[y*8+s-1]:BG_TRANS,vtb=y>0?vti[(y-1)*9+s]:BG_TRANS;oi[k]=enc_d(pi[k],hi[k],vi[k],htb,vtb);}}}
static void compute_ig(const uint8_t*sigs,uint16_t*ig_out){int cc[10]={0};for(int i=0;i<TRAIN_N;i++)cc[train_labels[i]]++;double hc=0;for(int c=0;c<10;c++){double p=(double)cc[c]/TRAIN_N;if(p>0)hc-=p*log2(p);}for(int k=0;k<N_BLOCKS;k++){int*cnt=calloc(256*10,sizeof(int)),vt[256]={0};for(int i=0;i<TRAIN_N;i++){int v=sigs[i*SIG_PAD+k];cnt[v*10+train_labels[i]]++;vt[v]++;}double hcond=0;for(int v=0;v<256;v++){if(!vt[v])continue;double pv=(double)vt[v]/TRAIN_N,hv=0;for(int c=0;c<10;c++){double pc=(double)cnt[v*10+c]/vt[v];if(pc>0)hv-=pc*log2(pc);}hcond+=pv*hv;}ig_out[k]=(uint16_t)((hc-hcond)*IG_SCALE*10+0.5);if(!ig_out[k])ig_out[k]=1;free(cnt);}}
static void build_eye_infrastructure(eye_t*e, int8_t*t_tr, int8_t*t_te){
    int8_t *h_tr=malloc(60000*800),*h_te=malloc(10000*800),*v_tr=malloc(60000*800),*v_te=malloc(10000*800);
    gradients(t_tr,h_tr,v_tr,TRAIN_N);gradients(t_te,h_te,v_te,TEST_N);
    uint8_t *px_tr=malloc(60000*SIG_PAD),*px_te=malloc(10000*SIG_PAD),*hg_tr=malloc(60000*SIG_PAD),*hg_te=malloc(10000*SIG_PAD),*vg_tr=malloc(60000*SIG_PAD),*vg_te=malloc(10000*SIG_PAD);
    block_sigs(t_tr,px_tr,TRAIN_N);block_sigs(t_te,px_te,TEST_N);block_sigs(h_tr,hg_tr,TRAIN_N);block_sigs(h_te,hg_te,TEST_N);block_sigs(v_tr,vg_tr,TRAIN_N);block_sigs(v_te,vg_te,TEST_N);
    uint8_t *ht_tr=malloc(60000*TRANS_PAD),*ht_te=malloc(10000*TRANS_PAD),*vt_tr=malloc(60000*TRANS_PAD),*vt_te=malloc(10000*TRANS_PAD);
    trans_fn(px_tr,ht_tr,vt_tr,TRAIN_N);trans_fn(px_te,ht_te,vt_te,TEST_N);
    e->joint_tr=malloc(60000*SIG_PAD);e->joint_te=malloc(10000*SIG_PAD);
    joint_sigs_fn(e->joint_tr,TRAIN_N,px_tr,hg_tr,vg_tr,ht_tr,vt_tr);joint_sigs_fn(e->joint_te,TEST_N,px_te,hg_te,vg_te,ht_te,vt_te);
    long vc[256]={0};for(int i=0;i<TRAIN_N;i++)for(int k=0;k<252;k++)vc[e->joint_tr[i*256+k]]++;
    e->bg=0;long mc=0;for(int v=0;v<256;v++)if(vc[v]>mc){mc=vc[v];e->bg=(uint8_t)v;}
    compute_ig(e->joint_tr,e->ig_w);
    for(int v=0;v<256;v++)for(int b=0;b<8;b++)e->nbr[v][b]=(uint8_t)(v^(1<<b));
    memset(e->idx_sz,0,sizeof(e->idx_sz));for(int i=0;i<60000;i++)for(int k=0;k<252;k++)if(e->joint_tr[i*256+k]!=e->bg)e->idx_sz[k][e->joint_tr[i*256+k]]++;
    uint32_t tot=0;for(int k=0;k<252;k++)for(int v=0;v<256;v++){e->idx_off[k][v]=tot;tot+=e->idx_sz[k][v];}
    e->idx_pool=malloc(tot*4);uint32_t*wp=malloc(252*256*4);memcpy(wp,e->idx_off,252*256*4);
    for(int i=0;i<60000;i++)for(int k=0;k<252;k++)if(e->joint_tr[i*256+k]!=e->bg)e->idx_pool[wp[k*256+e->joint_tr[i*256+k]]++]=(uint32_t)i;
    free(wp);free(h_tr);free(h_te);free(v_tr);free(v_te);free(px_tr);free(px_te);free(hg_tr);free(hg_te);free(vg_tr);free(vg_te);free(ht_tr);free(ht_te);free(vt_tr);free(vt_te);
}

static void vote_all_eyes(uint32_t*votes, int img){
    memset(votes,0,60000*4);
    for(int ei=0; ei<MAX_EYES; ei++){
        eye_t *e = &eyes[ei];uint8_t*sig=e->joint_te+img*SIG_PAD;
        for(int k=0;k<252;k++){
            uint8_t bv=sig[k];if(bv==e->bg)continue;
            uint16_t w=e->ig_w[k],wh=w>1?w/2:1;
            uint32_t off=e->idx_off[k][bv];for(uint16_t j=0;j<e->idx_sz[k][bv];j++)votes[e->idx_pool[off+j]]+=w;
            for(int nb=0;nb<8;nb++){uint8_t nv=e->nbr[bv][nb];if(nv==e->bg)continue;uint32_t noff=e->idx_off[k][nv];for(uint16_t j=0;j<e->idx_sz[k][nv];j++)votes[e->idx_pool[noff+j]]+=wh;}
        }
    }
}

/* === Topo Features === */
static int16_t enclosed_centroid(const int8_t*tern){uint8_t grid[FG_SZ];memset(grid,0,sizeof(grid));for(int y=0;y<28;y++)for(int x=0;x<28;x++)grid[(y+1)*30+(x+1)]=(tern[y*28+x]>0)?1:0;uint8_t visited[900]={0};int stack[900],sp=0;for(int y=0;y<30;y++)for(int x=0;x<30;x++){if(y==0||y==29||x==0||x==29){if(!grid[y*30+x]&&!visited[y*30+x]){visited[y*30+x]=1;stack[sp++]=y*30+x;}}}while(sp>0){int p=stack[--sp],py=p/30,px=p%30;const int dx[]={0,0,1,-1},dy[]={1,-1,0,0};for(int d=0;d<4;d++){int ny=py+dy[d],nx=px+dx[d];if(ny<0||ny>=30||nx<0||nx>=30)continue;if(!visited[ny*30+nx]&&!grid[ny*30+nx]){visited[ny*30+nx]=1;stack[sp++]=ny*30+nx;}}}int sum_y=0,count=0;for(int y=1;y<29;y++)for(int x=1;x<29;x++)if(!grid[y*30+x]&&!visited[y*30+x]){sum_y+=(y-1);count++;}return count>0?(int16_t)(sum_y/count):-1;}
static inline int32_t tdot(const int8_t*a,const int8_t*b){__m256i acc=_mm256_setzero_si256();for(int i=0;i<800;i+=32)acc=_mm256_add_epi8(acc,_mm256_sign_epi8(_mm256_loadu_si256((const __m256i*)(a+i)),_mm256_loadu_si256((const __m256i*)(b+i))));__m256i lo=_mm256_cvtepi8_epi16(_mm256_castsi256_si128(acc)),hi=_mm256_cvtepi8_epi16(_mm256_extracti128_si256(acc,1));__m256i s32=_mm256_madd_epi16(_mm256_add_epi16(lo,hi),_mm256_set1_epi16(1));__m128i s=_mm_add_epi32(_mm256_castsi256_si128(s32),_mm256_extracti128_si256(s32,1));s=_mm_add_epi32(s,_mm_shuffle_epi32(s,_MM_SHUFFLE(1,0,3,2)));s=_mm_add_epi32(s,_mm_shuffle_epi32(s,_MM_SHUFFLE(2,3,0,1)));return _mm_cvtsi128_si32(s);}
static void grid_div(const int8_t*hg,const int8_t*vg,int16_t*out){int regions[9]={0};for(int y=0;y<28;y++)for(int x=0;x<28;x++){int dh=(int)hg[y*28+x]-(x>0?hg[y*28+x-1]:0),dv=(int)vg[y*28+x]-(y>0?vg[(y-1)*28+x]:0),d=dh+dv;if(d<0){int ry=y*3/28,rx=x*3/28;regions[(ry>=3?2:ry)*3+(rx>=3?2:rx)]+=d;}}for(int i=0;i<9;i++)out[i]=(int16_t)(regions[i]<-32767?-32767:regions[i]);}

typedef struct { uint32_t id, votes; int32_t deep_dot, div_sim, cent_sim; int64_t combined; } cand_t;
static int cmp_votes(const void*a,const void*b){return(int)((const cand_t*)b)->votes-(int)((const cand_t*)a)->votes;}
static int cmp_comb(const void*a,const void*b){int64_t da=((const cand_t*)a)->combined,db=((const cand_t*)b)->combined;return(db>da)-(db<da);}
static int cmp_i16(const void*a,const void*b){return *(const int16_t*)a-*(const int16_t*)b;}

static int run_static(cand_t*pre,const int*nc_arr,const int16_t*qg,const int16_t*tg,int w_c,int w_d,int sc,int start,int end){
    int correct=0;for(int i=start;i<end;i++){cand_t cands[TOP_K];int nc=nc_arr[i];memcpy(cands,pre+(size_t)i*TOP_K,nc*sizeof(cand_t));
    for(int j=0;j<nc;j++){int32_t l=0;for(int r=0;r<9;r++)l+=abs(qg[i*16+r]-tg[cands[j].id*16+r]);cands[j].div_sim=-l;}
    int16_t vals[TOP_K];for(int j=0;j<nc;j++)vals[j]=divneg_train[cands[j].id];qsort(vals,nc,2,cmp_i16);int16_t med=vals[nc/2];for(int j=0;j<nc;j++)vals[j]=(int16_t)abs(vals[j]-med);qsort(vals,nc,2,cmp_i16);int mad=vals[nc/2]>0?vals[nc/2]:1;
    int wc=(int)(sc*w_c/(sc+mad)),wd=(int)(sc*w_d/(sc+mad));
    for(int j=0;j<nc;j++)cands[j].combined=(int64_t)cands[j].deep_dot+(int64_t)wc*cands[j].cent_sim+(int64_t)wd*cands[j].div_sim;
    qsort(cands,nc,sizeof(cand_t),cmp_comb);
    int v[10]={0};for(int j=0;j<3&&j<nc;j++)v[train_labels[cands[j].id]]++;int best=0;for(int c=1;c<10;c++)if(v[c]>v[best])best=c;if(best==test_labels[i])correct++;}return correct;
}

int main(int argc,char**argv){
    double t0=now_sec();if(argc>1)data_dir=argv[1];
    printf("=== SSTT Deep Fashion: 5-Eye + Deep Dot + Grid Search ===\n");
    load_data();
    for(int ei=0;ei<MAX_EYES;ei++){
        tern_train[ei]=aligned_alloc(32,60000*800);tern_test[ei]=aligned_alloc(32,10000*800);
        vgrad_train[ei]=aligned_alloc(32,60000*800);vgrad_test[ei]=aligned_alloc(32,10000*800);
        hgrad_train[ei]=aligned_alloc(32,60000*800);hgrad_test[ei]=aligned_alloc(32,10000*800);
        qfns[ei](raw_train_img,tern_train[ei],60000);qfns[ei](raw_test_img,tern_test[ei],10000);
        gradients(tern_train[ei],hgrad_train[ei],vgrad_train[ei],60000);gradients(tern_test[ei],hgrad_test[ei],vgrad_test[ei],10000);
        build_eye_infrastructure(&eyes[ei],tern_train[ei],tern_test[ei]);
        printf("  Eye %d done\n",ei+1);
    }
    cent_train=malloc(60000*2);cent_test=malloc(10000*2);for(int i=0;i<60000;i++)cent_train[i]=enclosed_centroid(tern_train[0]+i*800);for(int i=0;i<10000;i++)cent_test[i]=enclosed_centroid(tern_test[0]+i*800);
    divneg_train=malloc(60000*2);divneg_test=malloc(10000*2);
    for(int i=0;i<60000;i++){int8_t*h=hgrad_train[0]+i*800,*v=vgrad_train[0]+i*800;int s=0;for(int j=0;j<784;j++){int dh=h[j]-(j%28>0?h[j-1]:0),dv=v[j]-(j>=28?v[j-28]:0);if(dh+dv<0)s+=dh+dv;}divneg_train[i]=(int16_t)s;}
    for(int i=0;i<10000;i++){int8_t*h=hgrad_test[0]+i*800,*v=vgrad_test[0]+i*800;int s=0;for(int j=0;j<784;j++){int dh=h[j]-(j%28>0?h[j-1]:0),dv=v[j]-(j>=28?v[j-28]:0);if(dh+dv<0)s+=dh+dv;}divneg_test[i]=(int16_t)s;}
    gdiv_3x3_train=malloc(60000*16*2);gdiv_3x3_test=malloc(10000*16*2);for(int i=0;i<60000;i++)grid_div(hgrad_train[0]+i*800,vgrad_train[0]+i*800,gdiv_3x3_train+i*16);for(int i=0;i<10000;i++)grid_div(hgrad_test[0]+i*800,vgrad_test[0]+i*800,gdiv_3x3_test+i*16);

    cand_t*pre=malloc(10000*TOP_K*sizeof(cand_t));int*nc_arr=malloc(10000*4);uint32_t*votes=malloc(60000*4);
    for(int i=0;i<10000;i++){
        vote_all_eyes(votes,i);uint32_t mx=0;for(int j=0;j<60000;j++)if(votes[j]>mx)mx=votes[j];
        int nc=0;for(int j=0;j<60000&&nc<TOP_K;j++)if(votes[j]>mx*0.5){pre[i*TOP_K+nc].id=j;pre[i*TOP_K+nc].votes=votes[j];
            /* Use Eye 1 Dot only (stable ranking) */
            pre[i*TOP_K+nc].deep_dot = tdot(tern_test[0]+i*800,tern_train[0]+j*800)*256 + tdot(vgrad_test[0]+i*800,vgrad_train[0]+j*800)*192;
            int16_t cc=cent_train[j];if(cent_test[i]<0&&cc<0)pre[i*TOP_K+nc].cent_sim=50;else if(cent_test[i]<0||cc<0)pre[i*TOP_K+nc].cent_sim=-80;else pre[i*TOP_K+nc].cent_sim=-(int)abs(cent_test[i]-cc);
            nc++;}nc_arr[i]=nc;}
    
    printf("Grid searching weights...\n");
    int wc_v[]={0,25,50},wd_v[]={50,100,200,400},sc_v[]={20,50,100};
    int best_c=0,b_wc=25,b_wd=200,b_sc=50;
    for(int ci=0;ci<3;ci++)for(int di=0;di<4;di++)for(int si=0;si<3;si++){
        int c=run_static(pre,nc_arr,gdiv_3x3_test,gdiv_3x3_train,wc_v[ci],wd_v[di],sc_v[si],0,VAL_N);
        if(c>best_c){best_c=c;b_wc=wc_v[ci];b_wd=wd_v[di];b_sc=sc_v[si];}}
    printf("Best (Val): wc=%d wd=%d sc=%d -> %.2f%%\n",b_wc,b_wd,b_sc,100.0*best_c/VAL_N);
    int c_hold=run_static(pre,nc_arr,gdiv_3x3_test,gdiv_3x3_train,b_wc,b_wd,b_sc,HOLDOUT_START,TEST_N);
    printf("Holdout Accuracy: %.2f%% (%d/5000)\n",100.0*c_hold/5000,c_hold);
    printf("Total: %.2f sec\n",now_sec()-t0);
    return 0;
}
