/*
 * sstt_bag_topo9.c — Bagged Retrieval + Topological Ranking
 *
 * Experiment: Are position bagging (from sstt_bag_positions.c) and
 * topological ranking (from sstt_topo9.c) additive or redundant?
 *
 * Test A: Baseline topo9 static ranking (no bagging).
 * Test B: M=10, R=0.6 bagged retrieval -> topo9 static ranking.
 * Test C: Sweep M={5,7,10} x R={0.5,0.6,0.7} with topo9 ranking.
 *
 * Build: gcc -O3 -mavx2 -march=native -Wall -Wextra -o build/sstt_bag_topo9 src/sstt_bag_topo9.c -lm
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

/* Bagging parameters */
#define MAX_BAGS 15
#define BAG_SEED 42
#define BAG_TOP_K 50
#define MERGE_TOP_K TOP_K

static const char *data_dir="data/";
static double now_sec(void){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return ts.tv_sec+ts.tv_nsec*1e-9;}

static uint8_t *raw_train_img,*raw_test_img,*train_labels,*test_labels;
static int8_t *tern_train,*tern_test,*hgrad_train,*hgrad_test,*vgrad_train,*vgrad_test;
static uint8_t *px_tr,*px_te,*hg_tr,*hg_te,*vg_tr,*vg_te,*ht_tr,*ht_te,*vt_tr,*vt_te,*joint_tr,*joint_te;
static uint16_t ig_w[N_BLOCKS];static uint8_t nbr[BYTE_VALS][8];
static uint32_t idx_off[N_BLOCKS][BYTE_VALS];static uint16_t idx_sz[N_BLOCKS][BYTE_VALS];
static uint32_t *idx_pool;static uint8_t bg;static int *g_hist=NULL;static size_t g_hist_cap=0;
static int16_t *cent_train,*cent_test,*hprof_train,*hprof_test;
static int16_t *divneg_train,*divneg_test,*divneg_cy_train,*divneg_cy_test;
/* gdiv_2x4 / gdiv_3x3 selected per-dataset via qg/tg pointers */
static int16_t *gdiv_2x4_train,*gdiv_2x4_test;
static int16_t *gdiv_3x3_train,*gdiv_3x3_test;

/* Bag position masks */
static uint8_t bag_mask[MAX_BAGS][N_BLOCKS];

/* === LCG PRNG === */
static uint32_t lcg_state;
static void lcg_seed(uint32_t s){lcg_state=s;}
static uint32_t lcg_next(void){lcg_state=lcg_state*1664525u+1013904223u;return lcg_state;}

/* === Bag mask generation (Fisher-Yates) === */
static void generate_bag_masks(int n_bags,double bag_frac,uint32_t seed){
    int bag_pos=(int)(bag_frac*N_BLOCKS);if(bag_pos<1)bag_pos=1;if(bag_pos>N_BLOCKS)bag_pos=N_BLOCKS;
    lcg_seed(seed);int perm[N_BLOCKS];
    for(int m=0;m<n_bags;m++){
        for(int k=0;k<N_BLOCKS;k++)perm[k]=k;
        for(int k=N_BLOCKS-1;k>0;k--){int j=(int)(lcg_next()%(uint32_t)(k+1));int tmp=perm[k];perm[k]=perm[j];perm[j]=tmp;}
        memset(bag_mask[m],0,N_BLOCKS);
        for(int p=0;p<bag_pos;p++)bag_mask[m][perm[p]]=1;
    }
}

/* === All boilerplate (same as topo9, maximally compressed) === */
static uint8_t *load_idx(const char*path,uint32_t*cnt,uint32_t*ro,uint32_t*co){FILE*f=fopen(path,"rb");if(!f){fprintf(stderr,"ERR:%s\n",path);exit(1);}uint32_t m,n;if(fread(&m,4,1,f)!=1||fread(&n,4,1,f)!=1){fclose(f);exit(1);}m=__builtin_bswap32(m);n=__builtin_bswap32(n);*cnt=n;size_t s=1;if((m&0xFF)>=3){uint32_t r,c;if(fread(&r,4,1,f)!=1||fread(&c,4,1,f)!=1){fclose(f);exit(1);}r=__builtin_bswap32(r);c=__builtin_bswap32(c);if(ro)*ro=r;if(co)*co=c;s=(size_t)r*c;}else{if(ro)*ro=0;if(co)*co=0;}size_t total=(size_t)n*s;uint8_t*d=malloc(total);if(!d||fread(d,1,total,f)!=total){fclose(f);exit(1);}fclose(f);return d;}
static void load_data(void){uint32_t n,r,c;char p[256];snprintf(p,sizeof(p),"%strain-images-idx3-ubyte",data_dir);raw_train_img=load_idx(p,&n,&r,&c);snprintf(p,sizeof(p),"%strain-labels-idx1-ubyte",data_dir);train_labels=load_idx(p,&n,NULL,NULL);snprintf(p,sizeof(p),"%st10k-images-idx3-ubyte",data_dir);raw_test_img=load_idx(p,&n,&r,&c);snprintf(p,sizeof(p),"%st10k-labels-idx1-ubyte",data_dir);test_labels=load_idx(p,&n,NULL,NULL);}
static inline int8_t clamp_trit(int v){return v>0?1:v<0?-1:0;}
static void quant_tern(const uint8_t*src,int8_t*dst,int n){const __m256i bias=_mm256_set1_epi8((char)0x80),thi=_mm256_set1_epi8((char)(170^0x80)),tlo=_mm256_set1_epi8((char)(85^0x80)),one=_mm256_set1_epi8(1);for(int i=0;i<n;i++){const uint8_t*s=src+(size_t)i*PIXELS;int8_t*d=dst+(size_t)i*PADDED;int k;for(k=0;k+32<=PIXELS;k+=32){__m256i px=_mm256_loadu_si256((const __m256i*)(s+k));__m256i sp=_mm256_xor_si256(px,bias);_mm256_storeu_si256((__m256i*)(d+k),_mm256_sub_epi8(_mm256_and_si256(_mm256_cmpgt_epi8(sp,thi),one),_mm256_and_si256(_mm256_cmpgt_epi8(tlo,sp),one)));}for(;k<PIXELS;k++)d[k]=s[k]>170?1:s[k]<85?-1:0;memset(d+PIXELS,0,PADDED-PIXELS);}}
static void gradients(const int8_t*t,int8_t*h,int8_t*v,int n){for(int i=0;i<n;i++){const int8_t*ti=t+(size_t)i*PADDED;int8_t*hi=h+(size_t)i*PADDED;int8_t*vi=v+(size_t)i*PADDED;for(int y=0;y<IMG_H;y++){for(int x=0;x<IMG_W-1;x++)hi[y*IMG_W+x]=clamp_trit(ti[y*IMG_W+x+1]-ti[y*IMG_W+x]);hi[y*IMG_W+IMG_W-1]=0;}memset(hi+PIXELS,0,PADDED-PIXELS);for(int y=0;y<IMG_H-1;y++)for(int x=0;x<IMG_W;x++)vi[y*IMG_W+x]=clamp_trit(ti[(y+1)*IMG_W+x]-ti[y*IMG_W+x]);memset(vi+(IMG_H-1)*IMG_W,0,IMG_W);memset(vi+PIXELS,0,PADDED-PIXELS);}}
static inline uint8_t benc(int8_t a,int8_t b,int8_t c){return(uint8_t)((a+1)*9+(b+1)*3+(c+1));}
static void block_sigs(const int8_t*data,uint8_t*sigs,int n){for(int i=0;i<n;i++){const int8_t*img=data+(size_t)i*PADDED;uint8_t*sig=sigs+(size_t)i*SIG_PAD;for(int y=0;y<IMG_H;y++)for(int s=0;s<BLKS_PER_ROW;s++){int b=y*IMG_W+s*3;sig[y*BLKS_PER_ROW+s]=benc(img[b],img[b+1],img[b+2]);}memset(sig+N_BLOCKS,0xFF,SIG_PAD-N_BLOCKS);}}
static inline uint8_t tenc(uint8_t a,uint8_t b){int8_t a0=(a/9)-1,a1=((a/3)%3)-1,a2=(a%3)-1,b0=(b/9)-1,b1=((b/3)%3)-1,b2=(b%3)-1;return benc(clamp_trit(b0-a0),clamp_trit(b1-a1),clamp_trit(b2-a2));}
static void trans_fn(const uint8_t*bs,int str,uint8_t*ht,uint8_t*vt,int n){for(int i=0;i<n;i++){const uint8_t*s=bs+(size_t)i*str;uint8_t*h=ht+(size_t)i*TRANS_PAD;uint8_t*v=vt+(size_t)i*TRANS_PAD;for(int y=0;y<IMG_H;y++)for(int ss=0;ss<H_TRANS_PER_ROW;ss++)h[y*H_TRANS_PER_ROW+ss]=tenc(s[y*BLKS_PER_ROW+ss],s[y*BLKS_PER_ROW+ss+1]);memset(h+N_HTRANS,0xFF,TRANS_PAD-N_HTRANS);for(int y=0;y<V_TRANS_PER_COL;y++)for(int ss=0;ss<BLKS_PER_ROW;ss++)v[y*BLKS_PER_ROW+ss]=tenc(s[y*BLKS_PER_ROW+ss],s[(y+1)*BLKS_PER_ROW+ss]);memset(v+N_VTRANS,0xFF,TRANS_PAD-N_VTRANS);}}
static uint8_t enc_d(uint8_t px,uint8_t hg,uint8_t vg,uint8_t ht,uint8_t vt){int ps=((px/9)-1)+(((px/3)%3)-1)+((px%3)-1),hs=((hg/9)-1)+(((hg/3)%3)-1)+((hg%3)-1),vs=((vg/9)-1)+(((vg/3)%3)-1)+((vg%3)-1);uint8_t pc=ps<0?0:ps==0?1:ps<3?2:3,hc=hs<0?0:hs==0?1:hs<3?2:3,vc=vs<0?0:vs==0?1:vs<3?2:3;return pc|(hc<<2)|(vc<<4)|((ht!=BG_TRANS)?1<<6:0)|((vt!=BG_TRANS)?1<<7:0);}
static void joint_sigs_fn(uint8_t*out,int n,const uint8_t*px,const uint8_t*hg,const uint8_t*vg,const uint8_t*ht,const uint8_t*vt){for(int i=0;i<n;i++){const uint8_t*pi=px+(size_t)i*SIG_PAD,*hi=hg+(size_t)i*SIG_PAD,*vi=vg+(size_t)i*SIG_PAD;const uint8_t*hti=ht+(size_t)i*TRANS_PAD,*vti=vt+(size_t)i*TRANS_PAD;uint8_t*oi=out+(size_t)i*SIG_PAD;for(int y=0;y<IMG_H;y++)for(int s=0;s<BLKS_PER_ROW;s++){int k=y*BLKS_PER_ROW+s;uint8_t htb=s>0?hti[y*H_TRANS_PER_ROW+(s-1)]:BG_TRANS;uint8_t vtb=y>0?vti[(y-1)*BLKS_PER_ROW+s]:BG_TRANS;oi[k]=enc_d(pi[k],hi[k],vi[k],htb,vtb);}memset(oi+N_BLOCKS,0xFF,SIG_PAD-N_BLOCKS);}}
static void compute_ig(const uint8_t*sigs,int nv,uint8_t bgv,uint16_t*ig_out){int cc[N_CLASSES]={0};for(int i=0;i<TRAIN_N;i++)cc[train_labels[i]]++;double hc=0;for(int c=0;c<N_CLASSES;c++){double p=(double)cc[c]/TRAIN_N;if(p>0)hc-=p*log2(p);}double raw[N_BLOCKS],mx=0;for(int k=0;k<N_BLOCKS;k++){int*cnt=calloc((size_t)nv*N_CLASSES,sizeof(int));int*vt=calloc(nv,sizeof(int));for(int i=0;i<TRAIN_N;i++){int v=sigs[(size_t)i*SIG_PAD+k];cnt[v*N_CLASSES+train_labels[i]]++;vt[v]++;}double hcond=0;for(int v=0;v<nv;v++){if(!vt[v]||v==(int)bgv)continue;double pv=(double)vt[v]/TRAIN_N,hv=0;for(int c=0;c<N_CLASSES;c++){double pc=(double)cnt[v*N_CLASSES+c]/vt[v];if(pc>0)hv-=pc*log2(pc);}hcond+=pv*hv;}raw[k]=hc-hcond;if(raw[k]>mx)mx=raw[k];free(cnt);free(vt);}for(int k=0;k<N_BLOCKS;k++){ig_out[k]=mx>0?(uint16_t)(raw[k]/mx*IG_SCALE+0.5):1;if(!ig_out[k])ig_out[k]=1;}}
static void build_index(void){long vc[BYTE_VALS]={0};for(int i=0;i<TRAIN_N;i++){const uint8_t*s=joint_tr+(size_t)i*SIG_PAD;for(int k=0;k<N_BLOCKS;k++)vc[s[k]]++;}bg=0;long mc=0;for(int v=0;v<BYTE_VALS;v++)if(vc[v]>mc){mc=vc[v];bg=(uint8_t)v;}compute_ig(joint_tr,BYTE_VALS,bg,ig_w);for(int v=0;v<BYTE_VALS;v++)for(int b=0;b<8;b++)nbr[v][b]=(uint8_t)(v^(1<<b));memset(idx_sz,0,sizeof(idx_sz));for(int i=0;i<TRAIN_N;i++){const uint8_t*s=joint_tr+(size_t)i*SIG_PAD;for(int k=0;k<N_BLOCKS;k++)if(s[k]!=bg)idx_sz[k][s[k]]++;}uint32_t tot=0;for(int k=0;k<N_BLOCKS;k++)for(int v=0;v<BYTE_VALS;v++){idx_off[k][v]=tot;tot+=idx_sz[k][v];}idx_pool=malloc((size_t)tot*sizeof(uint32_t));uint32_t(*wp)[BYTE_VALS]=malloc((size_t)N_BLOCKS*BYTE_VALS*4);memcpy(wp,idx_off,(size_t)N_BLOCKS*BYTE_VALS*4);for(int i=0;i<TRAIN_N;i++){const uint8_t*s=joint_tr+(size_t)i*SIG_PAD;for(int k=0;k<N_BLOCKS;k++)if(s[k]!=bg)idx_pool[wp[k][s[k]]++]=(uint32_t)i;}free(wp);printf("  Index: %u entries (%.1f MB)\n",tot,(double)tot*4/1048576);}

/* === Structural features (centroid, hprofile, divergence, grid divergence) === */
static int16_t enclosed_centroid(const int8_t*tern){uint8_t grid[FG_SZ];memset(grid,0,sizeof(grid));for(int y=0;y<IMG_H;y++)for(int x=0;x<IMG_W;x++)grid[(y+1)*FG_W+(x+1)]=(tern[y*IMG_W+x]>0)?1:0;uint8_t visited[FG_SZ];memset(visited,0,sizeof(visited));int stack[FG_SZ];int sp=0;for(int y=0;y<FG_H;y++)for(int x=0;x<FG_W;x++){if(y==0||y==FG_H-1||x==0||x==FG_W-1){int pos=y*FG_W+x;if(!grid[pos]&&!visited[pos]){visited[pos]=1;stack[sp++]=pos;}}}while(sp>0){int pos=stack[--sp];int py=pos/FG_W,px2=pos%FG_W;const int dx[]={0,0,1,-1},dy[]={1,-1,0,0};for(int d=0;d<4;d++){int ny=py+dy[d],nx=px2+dx[d];if(ny<0||ny>=FG_H||nx<0||nx>=FG_W)continue;int npos=ny*FG_W+nx;if(!visited[npos]&&!grid[npos]){visited[npos]=1;stack[sp++]=npos;}}}int sum_y=0,count=0;for(int y=1;y<FG_H-1;y++)for(int x=1;x<FG_W-1;x++){int pos=y*FG_W+x;if(!grid[pos]&&!visited[pos]){sum_y+=(y-1);count++;}}return count>0?(int16_t)(sum_y/count):-1;}
static void compute_centroid_all(const int8_t*tern,int16_t*out,int n){for(int i=0;i<n;i++)out[i]=enclosed_centroid(tern+(size_t)i*PADDED);}
static void hprofile(const int8_t*tern,int16_t*prof){for(int y=0;y<IMG_H;y++){int c=0;for(int x=0;x<IMG_W;x++)c+=(tern[y*IMG_W+x]>0);prof[y]=(int16_t)c;}for(int y=IMG_H;y<HP_PAD;y++)prof[y]=0;}
static void compute_hprof_all(const int8_t*tern,int16_t*out,int n){for(int i=0;i<n;i++)hprofile(tern+(size_t)i*PADDED,out+(size_t)i*HP_PAD);}
static void div_features(const int8_t*hg,const int8_t*vg,int16_t*ns,int16_t*cy){int neg_sum=0,neg_ysum=0,neg_cnt=0;for(int y=0;y<IMG_H;y++)for(int x=0;x<IMG_W;x++){int dh=(int)hg[y*IMG_W+x]-(x>0?(int)hg[y*IMG_W+x-1]:0);int dv=(int)vg[y*IMG_W+x]-(y>0?(int)vg[(y-1)*IMG_W+x]:0);int d=dh+dv;if(d<0){neg_sum+=d;neg_ysum+=y;neg_cnt++;}}*ns=(int16_t)(neg_sum<-32767?-32767:neg_sum);*cy=neg_cnt>0?(int16_t)(neg_ysum/neg_cnt):-1;}
static void compute_div_all(const int8_t*hg,const int8_t*vg,int16_t*ns,int16_t*cy,int n){for(int i=0;i<n;i++)div_features(hg+(size_t)i*PADDED,vg+(size_t)i*PADDED,&ns[i],&cy[i]);}
static inline int32_t tdot(const int8_t*a,const int8_t*b){__m256i acc=_mm256_setzero_si256();for(int i=0;i<PADDED;i+=32)acc=_mm256_add_epi8(acc,_mm256_sign_epi8(_mm256_load_si256((const __m256i*)(a+i)),_mm256_load_si256((const __m256i*)(b+i))));__m256i lo=_mm256_cvtepi8_epi16(_mm256_castsi256_si128(acc)),hi=_mm256_cvtepi8_epi16(_mm256_extracti128_si256(acc,1));__m256i s32=_mm256_madd_epi16(_mm256_add_epi16(lo,hi),_mm256_set1_epi16(1));__m128i s=_mm_add_epi32(_mm256_castsi256_si128(s32),_mm256_extracti128_si256(s32,1));s=_mm_hadd_epi32(s,s);s=_mm_hadd_epi32(s,s);return _mm_cvtsi128_si32(s);}

/* === Vote: full (all positions) === */
static void vote(uint32_t*votes,int img){memset(votes,0,TRAIN_N*sizeof(uint32_t));const uint8_t*sig=joint_te+(size_t)img*SIG_PAD;for(int k=0;k<N_BLOCKS;k++){uint8_t bv=sig[k];if(bv==bg)continue;uint16_t w=ig_w[k],wh=w>1?w/2:1;{uint32_t off=idx_off[k][bv];uint16_t sz=idx_sz[k][bv];const uint32_t*ids=idx_pool+off;for(uint16_t j=0;j<sz;j++)votes[ids[j]]+=w;}for(int nb=0;nb<8;nb++){uint8_t nv=nbr[bv][nb];if(nv==bg)continue;uint32_t noff=idx_off[k][nv];uint16_t nsz=idx_sz[k][nv];const uint32_t*nids=idx_pool+noff;for(uint16_t j=0;j<nsz;j++)votes[nids[j]]+=wh;}}}

/* === Vote: masked (for bagging, skips positions where mask[k]==0) === */
static void vote_masked(uint32_t*votes,int img,const uint8_t*mask){memset(votes,0,TRAIN_N*sizeof(uint32_t));const uint8_t*sig=joint_te+(size_t)img*SIG_PAD;for(int k=0;k<N_BLOCKS;k++){if(!mask[k])continue;uint8_t bv=sig[k];if(bv==bg)continue;uint16_t w=ig_w[k],wh=w>1?w/2:1;{uint32_t off=idx_off[k][bv];uint16_t sz=idx_sz[k][bv];const uint32_t*ids=idx_pool+off;for(uint16_t j=0;j<sz;j++)votes[ids[j]]+=w;}for(int nb=0;nb<8;nb++){uint8_t nv=nbr[bv][nb];if(nv==bg)continue;uint32_t noff=idx_off[k][nv];uint16_t nsz=idx_sz[k][nv];const uint32_t*nids=idx_pool+noff;for(uint16_t j=0;j<nsz;j++)votes[nids[j]]+=wh;}}}

/* === Grid divergence === */
static void grid_div(const int8_t*hg,const int8_t*vg,int grow,int gcol,int16_t*out){
    int nr=grow*gcol;int regions[MAX_REGIONS];memset(regions,0,sizeof(int)*nr);
    for(int y=0;y<IMG_H;y++)for(int x=0;x<IMG_W;x++){int dh=(int)hg[y*IMG_W+x]-(x>0?(int)hg[y*IMG_W+x-1]:0);int dv=(int)vg[y*IMG_W+x]-(y>0?(int)vg[(y-1)*IMG_W+x]:0);int d=dh+dv;if(d<0){int ry=y*grow/IMG_H;int rx=x*gcol/IMG_W;if(ry>=grow)ry=grow-1;if(rx>=gcol)rx=gcol-1;regions[ry*gcol+rx]+=d;}}
    for(int i=0;i<nr;i++)out[i]=(int16_t)(regions[i]<-32767?-32767:regions[i]);
}
static void compute_grid_div(const int8_t*hg,const int8_t*vg,int grow,int gcol,int16_t*out,int n){for(int i=0;i<n;i++)grid_div(hg+(size_t)i*PADDED,vg+(size_t)i*PADDED,grow,gcol,out+(size_t)i*MAX_REGIONS);}

/* ================================================================
 *  Candidate with all features precomputed (same as topo9)
 * ================================================================ */
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

/* Per-bag top-K selection into lightweight array (id+votes only) */
typedef struct { uint32_t id; uint32_t votes; } bag_cand_t;
static int cmp_bagcand_d(const void*a,const void*b){return(int)((const bag_cand_t*)b)->votes-(int)((const bag_cand_t*)a)->votes;}
static int select_bag_top_k(const uint32_t*votes,int n,bag_cand_t*out,int k){
    uint32_t mx=0;for(int i=0;i<n;i++)if(votes[i]>mx)mx=votes[i];if(!mx)return 0;
    if((size_t)(mx+1)>g_hist_cap){g_hist_cap=(size_t)(mx+1)+4096;free(g_hist);g_hist=malloc(g_hist_cap*sizeof(int));}
    memset(g_hist,0,(mx+1)*sizeof(int));for(int i=0;i<n;i++)if(votes[i])g_hist[votes[i]]++;
    int cum=0,thr;for(thr=(int)mx;thr>=1;thr--){cum+=g_hist[thr];if(cum>=k)break;}if(thr<1)thr=1;
    int nc=0;for(int i=0;i<n&&nc<k;i++)if(votes[i]>=(uint32_t)thr){out[nc].id=(uint32_t)i;out[nc].votes=votes[i];nc++;}
    qsort(out,(size_t)nc,sizeof(bag_cand_t),cmp_bagcand_d);return nc;
}

/* === Merge bag candidates into cand_t array (appearance-weighted) === */
static int merge_bag_cands(bag_cand_t bag_cands[][BAG_TOP_K],int*bag_nc,int n_bags,cand_t*out,int merge_k){
    /* Use per-ID arrays (TRAIN_N is only 60K) */
    static uint16_t appear[TRAIN_N];
    static uint32_t tvotes[TRAIN_N];
    static uint8_t seen[TRAIN_N];
    memset(appear,0,sizeof(appear));memset(tvotes,0,sizeof(tvotes));memset(seen,0,sizeof(seen));
    uint32_t*uids=malloc((size_t)n_bags*BAG_TOP_K*sizeof(uint32_t));int nu=0;
    for(int m=0;m<n_bags;m++)for(int j=0;j<bag_nc[m];j++){
        uint32_t id=bag_cands[m][j].id;appear[id]++;tvotes[id]+=bag_cands[m][j].votes;
        if(!seen[id]){seen[id]=1;uids[nu++]=id;}
    }
    /* Sort by (appearance DESC, total_votes DESC) using cand_t temporarily */
    int nm=nu<merge_k?nu:merge_k;
    /* Build sortable array */
    typedef struct{uint32_t id;uint16_t app;uint32_t tv;} msort_t;
    msort_t*ms=malloc((size_t)nu*sizeof(msort_t));
    for(int j=0;j<nu;j++){ms[j].id=uids[j];ms[j].app=appear[uids[j]];ms[j].tv=tvotes[uids[j]];}
    /* Sort: appearance DESC, then total_votes DESC */
    int cmp_ms(const void*a,const void*b){
        const msort_t*ma=(const msort_t*)a,*mb=(const msort_t*)b;
        if(mb->app!=ma->app)return(int)mb->app-(int)ma->app;
        if(mb->tv!=ma->tv)return(mb->tv>ma->tv)?1:-1;
        return 0;
    }
    qsort(ms,(size_t)nu,sizeof(msort_t),cmp_ms);
    for(int j=0;j<nm;j++){out[j]=(cand_t){0};out[j].id=ms[j].id;out[j].votes=ms[j].tv;}
    free(ms);free(uids);return nm;
}

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

/* MAD */
static int cmp_i16(const void*a,const void*b){return *(const int16_t*)a-*(const int16_t*)b;}
static int compute_mad(const cand_t*cands,int nc){if(nc<3)return 1000;int16_t vals[TOP_K];for(int j=0;j<nc;j++)vals[j]=divneg_train[cands[j].id];qsort(vals,(size_t)nc,sizeof(int16_t),cmp_i16);int16_t med=vals[nc/2];int16_t devs[TOP_K];for(int j=0;j<nc;j++)devs[j]=(int16_t)abs(vals[j]-med);qsort(devs,(size_t)nc,sizeof(int16_t),cmp_i16);return devs[nc/2]>0?devs[nc/2]:1;}

/* ================================================================
 *  Static sort-then-vote with topo9 Kalman-weighted ranking
 * ================================================================ */
static int run_static_tuned(const cand_t*pre,const int*nc_arr,
                       const int16_t*qg,const int16_t*tg,int nreg,
                       int w_c,int w_p,int w_d,int w_g,int sc,
                       uint8_t*preds){
    int correct=0;
    for(int i=0;i<TEST_N;i++){
        cand_t cands[TOP_K];int nc=nc_arr[i];
        memcpy(cands,pre+(size_t)i*TOP_K,(size_t)nc*sizeof(cand_t));
        const int16_t*qi=qg+(size_t)i*MAX_REGIONS;
        for(int j=0;j<nc;j++){const int16_t*ci=tg+(size_t)cands[j].id*MAX_REGIONS;int32_t l=0;for(int r=0;r<nreg;r++)l+=abs(qi[r]-ci[r]);cands[j].gdiv_sim=-l;}
        int mad=compute_mad(cands,nc);int64_t s=(int64_t)sc;
        int wc=(int)(s*w_c/(s+mad)),wp=(int)(s*w_p/(s+mad)),wd=(int)(s*w_d/(s+mad)),wg=(int)(s*w_g/(s+mad));
        for(int j=0;j<nc;j++)cands[j].combined=(int64_t)256*cands[j].dot_px+(int64_t)192*cands[j].dot_vg+(int64_t)wc*cands[j].cent_sim+(int64_t)wp*cands[j].prof_sim+(int64_t)wd*cands[j].div_sim+(int64_t)wg*cands[j].gdiv_sim;
        qsort(cands,(size_t)nc,sizeof(cand_t),cmp_comb_d);
        int pred=knn_vote(cands,nc,3);
        if(preds)preds[i]=(uint8_t)pred;
        if(pred==test_labels[i])correct++;
    }
    return correct;
}

/* ================================================================
 *  Error mode analysis (works on pre-computed candidate arrays)
 * ================================================================ */
static void error_mode_analysis(const cand_t*pre,const int*nc_arr,
                                const int16_t*qg,const int16_t*tg,int nreg,
                                int w_c,int w_p,int w_d,int w_g,int sc,
                                const char*label){
    int mode_a=0,mode_b=0,mode_c=0,n_errors=0,n_correct=0;
    for(int i=0;i<TEST_N;i++){
        cand_t cands[TOP_K];int nc=nc_arr[i];
        memcpy(cands,pre+(size_t)i*TOP_K,(size_t)nc*sizeof(cand_t));
        const int16_t*qi=qg+(size_t)i*MAX_REGIONS;
        for(int j=0;j<nc;j++){const int16_t*ci=tg+(size_t)cands[j].id*MAX_REGIONS;int32_t l=0;for(int r=0;r<nreg;r++)l+=abs(qi[r]-ci[r]);cands[j].gdiv_sim=-l;}
        int mad=compute_mad(cands,nc);int64_t s=(int64_t)sc;
        int wc=(int)(s*w_c/(s+mad)),wp=(int)(s*w_p/(s+mad)),wd=(int)(s*w_d/(s+mad)),wg=(int)(s*w_g/(s+mad));
        for(int j=0;j<nc;j++)cands[j].combined=(int64_t)256*cands[j].dot_px+(int64_t)192*cands[j].dot_vg+(int64_t)wc*cands[j].cent_sim+(int64_t)wp*cands[j].prof_sim+(int64_t)wd*cands[j].div_sim+(int64_t)wg*cands[j].gdiv_sim;
        qsort(cands,(size_t)nc,sizeof(cand_t),cmp_comb_d);
        int pred=knn_vote(cands,nc,3);
        int tl=test_labels[i];
        if(pred==tl){n_correct++;continue;}
        n_errors++;
        /* Mode A: true label not in top-K candidates */
        int found=0;for(int j=0;j<nc;j++)if(train_labels[cands[j].id]==tl){found=1;break;}
        if(!found){mode_a++;continue;}
        /* Mode B: true label in top-K but wrong class wins combined ranking */
        int r1_lbl=train_labels[cands[0].id];
        if(r1_lbl!=tl){mode_b++;continue;}
        /* Mode C: true label wins rank-1 combined but loses k=3 vote */
        mode_c++;
    }
    printf("\n  Error Mode Analysis: %s\n",label);
    printf("    Total errors: %d / %d (%.2f%%)\n",n_errors,TEST_N,100.0*n_correct/TEST_N);
    printf("    Mode A (true not in top-K):           %d (%.1f%% of errors)\n",mode_a,n_errors>0?100.0*mode_a/n_errors:0.0);
    printf("    Mode B (true in top-K, wrong ranks):  %d (%.1f%%)\n",mode_b,n_errors>0?100.0*mode_b/n_errors:0.0);
    printf("    Mode C (true rank-1, loses k=3 vote): %d (%.1f%%)\n",mode_c,n_errors>0?100.0*mode_c/n_errors:0.0);
}

static void report(const uint8_t*preds,const char*label,int correct){
    printf("--- %s ---\n",label);printf("  Accuracy: %.2f%%  (%d errors)\n",100.0*correct/TEST_N,TEST_N-correct);
    int conf[N_CLASSES][N_CLASSES];memset(conf,0,sizeof(conf));for(int i=0;i<TEST_N;i++)conf[test_labels[i]][preds[i]]++;
    typedef struct{int a,b,count;}pair_t;pair_t pairs[45];int np=0;
    for(int a=0;a<N_CLASSES;a++)for(int b=a+1;b<N_CLASSES;b++)pairs[np++]=(pair_t){a,b,conf[a][b]+conf[b][a]};
    for(int i=0;i<np-1;i++)for(int j=i+1;j<np;j++)if(pairs[j].count>pairs[i].count){pair_t t2=pairs[i];pairs[i]=pairs[j];pairs[j]=t2;}
    printf("  Top-5:");for(int i=0;i<5&&i<np;i++)printf("  %d<->%d:%d",pairs[i].a,pairs[i].b,pairs[i].count);printf("\n\n");
}

/* ================================================================
 *  Main
 * ================================================================ */
int main(int argc,char**argv){
    double t0=now_sec();
    if(argc>1){data_dir=argv[1];size_t l=strlen(data_dir);if(l&&data_dir[l-1]!='/'){char*buf=malloc(l+2);memcpy(buf,data_dir,l);buf[l]='/';buf[l+1]='\0';data_dir=buf;}}
    int is_fashion=strstr(data_dir,"fashion")!=NULL;
    const char*ds=is_fashion?"Fashion-MNIST":"MNIST";
    printf("=== SSTT Bag+Topo9: Bagged Retrieval + Topological Ranking (%s) ===\n\n",ds);

    load_data();
    tern_train=aligned_alloc(32,(size_t)TRAIN_N*PADDED);tern_test=aligned_alloc(32,(size_t)TEST_N*PADDED);
    hgrad_train=aligned_alloc(32,(size_t)TRAIN_N*PADDED);hgrad_test=aligned_alloc(32,(size_t)TEST_N*PADDED);
    vgrad_train=aligned_alloc(32,(size_t)TRAIN_N*PADDED);vgrad_test=aligned_alloc(32,(size_t)TEST_N*PADDED);
    quant_tern(raw_train_img,tern_train,TRAIN_N);quant_tern(raw_test_img,tern_test,TEST_N);
    gradients(tern_train,hgrad_train,vgrad_train,TRAIN_N);gradients(tern_test,hgrad_test,vgrad_test,TEST_N);
    px_tr=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD);px_te=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    hg_tr=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD);hg_te=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    vg_tr=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD);vg_te=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    block_sigs(tern_train,px_tr,TRAIN_N);block_sigs(tern_test,px_te,TEST_N);
    block_sigs(hgrad_train,hg_tr,TRAIN_N);block_sigs(hgrad_test,hg_te,TEST_N);
    block_sigs(vgrad_train,vg_tr,TRAIN_N);block_sigs(vgrad_test,vg_te,TEST_N);
    ht_tr=aligned_alloc(32,(size_t)TRAIN_N*TRANS_PAD);ht_te=aligned_alloc(32,(size_t)TEST_N*TRANS_PAD);
    vt_tr=aligned_alloc(32,(size_t)TRAIN_N*TRANS_PAD);vt_te=aligned_alloc(32,(size_t)TEST_N*TRANS_PAD);
    trans_fn(px_tr,SIG_PAD,ht_tr,vt_tr,TRAIN_N);trans_fn(px_te,SIG_PAD,ht_te,vt_te,TEST_N);
    joint_tr=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD);joint_te=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    joint_sigs_fn(joint_tr,TRAIN_N,px_tr,hg_tr,vg_tr,ht_tr,vt_tr);
    joint_sigs_fn(joint_te,TEST_N,px_te,hg_te,vg_te,ht_te,vt_te);
    printf("Building index...\n");build_index();

    printf("Computing features...\n");double tf=now_sec();
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
    printf("  Features: %.2f sec\n\n",now_sec()-tf);

    /* Dataset-adaptive parameters */
    int16_t *qg,*tg;int nreg;
    int w_c,w_p,w_d,w_g,sc_val;
    if(is_fashion){qg=gdiv_3x3_test;tg=gdiv_3x3_train;nreg=9;w_c=25;w_p=0;w_d=200;w_g=200;sc_val=50;}
    else{qg=gdiv_2x4_test;tg=gdiv_2x4_train;nreg=8;w_c=50;w_p=16;w_d=200;w_g=100;sc_val=50;}

    /* ================================================================
     *  Test A: Baseline (no bagging) — standard topo9 static ranking
     * ================================================================ */
    printf("=== Test A: Baseline (no bagging, topo9 static ranking) ===\n");
    double tA=now_sec();
    cand_t*pre_a=malloc((size_t)TEST_N*TOP_K*sizeof(cand_t));
    int*nc_a=malloc((size_t)TEST_N*sizeof(int));
    uint32_t*votes=calloc(TRAIN_N,sizeof(uint32_t));
    for(int i=0;i<TEST_N;i++){
        vote(votes,i);cand_t*ci=pre_a+(size_t)i*TOP_K;
        int nc=select_top_k(votes,TRAIN_N,ci,TOP_K);
        compute_base(ci,nc,i,cent_test[i],hprof_test+(size_t)i*HP_PAD,divneg_test[i],divneg_cy_test[i]);
        nc_a[i]=nc;if((i+1)%2000==0)fprintf(stderr,"  A: %d/%d\r",i+1,TEST_N);
    }
    fprintf(stderr,"\n");
    printf("  Precompute: %.2f sec\n",now_sec()-tA);

    uint8_t*preds_a=calloc(TEST_N,1);
    int cA=run_static_tuned(pre_a,nc_a,qg,tg,nreg,w_c,w_p,w_d,w_g,sc_val,preds_a);
    report(preds_a,"A: Baseline topo9 (no bagging)",cA);
    error_mode_analysis(pre_a,nc_a,qg,tg,nreg,w_c,w_p,w_d,w_g,sc_val,"A: Baseline");

    /* ================================================================
     *  Test B: Bagged retrieval (M=10, R=0.6) + topo9 ranking
     * ================================================================ */
    printf("\n=== Test B: Bagged retrieval (M=10, R=0.6) + topo9 ranking ===\n");
    int B_M=10;double B_R=0.6;
    generate_bag_masks(B_M,B_R,BAG_SEED);
    int bag_pos=(int)(B_R*N_BLOCKS);
    printf("  Config: M=%d R=%.1f (%d pos/bag)\n",B_M,B_R,bag_pos);

    double tB=now_sec();
    cand_t*pre_b=malloc((size_t)TEST_N*TOP_K*sizeof(cand_t));
    int*nc_b=malloc((size_t)TEST_N*sizeof(int));
    bag_cand_t(*bcands)[BAG_TOP_K]=malloc((size_t)B_M*BAG_TOP_K*sizeof(bag_cand_t));
    int*bnc=malloc((size_t)B_M*sizeof(int));
    long total_merged=0;

    for(int i=0;i<TEST_N;i++){
        /* Run M bagged votes */
        for(int m=0;m<B_M;m++){
            vote_masked(votes,i,bag_mask[m]);
            bnc[m]=select_bag_top_k(votes,TRAIN_N,bcands[m],BAG_TOP_K);
        }
        /* Merge candidates from all bags */
        cand_t*ci=pre_b+(size_t)i*TOP_K;
        int nc=merge_bag_cands(bcands,bnc,B_M,ci,TOP_K);
        total_merged+=nc;
        /* Compute topo9 base features on merged candidates */
        compute_base(ci,nc,i,cent_test[i],hprof_test+(size_t)i*HP_PAD,divneg_test[i],divneg_cy_test[i]);
        nc_b[i]=nc;
        if((i+1)%2000==0)fprintf(stderr,"  B: %d/%d\r",i+1,TEST_N);
    }
    fprintf(stderr,"\n");
    printf("  Precompute: %.2f sec  Mean candidates: %.1f\n",now_sec()-tB,(double)total_merged/TEST_N);

    uint8_t*preds_b=calloc(TEST_N,1);
    int cB=run_static_tuned(pre_b,nc_b,qg,tg,nreg,w_c,w_p,w_d,w_g,sc_val,preds_b);
    report(preds_b,"B: Bagged (M=10,R=0.6) + topo9",cB);
    printf("  Delta vs baseline: %+d (%+.2f pp)\n\n",cB-cA,100.0*(cB-cA)/TEST_N);
    error_mode_analysis(pre_b,nc_b,qg,tg,nreg,w_c,w_p,w_d,w_g,sc_val,"B: Bagged M=10 R=0.6");

    /* Per-pair delta */
    printf("\n  Per-pair delta (B vs A):\n  %-8s %8s %8s %8s\n","Pair","A_err","B_err","Delta");
    {int tpairs[][2]={{3,5},{4,9},{1,7},{3,8},{7,9},{0,6},{2,4},{2,6}};
     for(int pi=0;pi<8;pi++){int a2=tpairs[pi][0],b2=tpairs[pi][1];int ea=0,eb=0;
         for(int i=0;i<TEST_N;i++){int tl=test_labels[i];if(tl!=a2&&tl!=b2)continue;
             if(preds_a[i]!=tl&&(preds_a[i]==a2||preds_a[i]==b2))ea++;
             if(preds_b[i]!=tl&&(preds_b[i]==a2||preds_b[i]==b2))eb++;}
         printf("  %d<->%d: %8d %8d %+8d\n",a2,b2,ea,eb,eb-ea);}}

    /* ================================================================
     *  Test C: Sweep M={5,7,10} x R={0.5,0.6,0.7} + topo9 ranking
     * ================================================================ */
    printf("\n=== Test C: Bagged+Topo9 Sweep ===\n\n");
    int m_vals[]={5,7,10};double r_vals[]={0.5,0.6,0.7};
    int n_m=3,n_r=3;
    typedef struct{int m;double r;int correct;double elapsed;double mean_cands;}sweep_t;
    sweep_t results[9];int nr=0;
    double best_acc=0;int best_mi=0,best_ri=0;

    for(int mi=0;mi<n_m;mi++){
        for(int ri=0;ri<n_r;ri++){
            int M=m_vals[mi];double R=r_vals[ri];
            generate_bag_masks(M,R,BAG_SEED);
            printf("  Sweep M=%d R=%.1f ...\n",M,R);
            double ts=now_sec();
            cand_t*pre_s=malloc((size_t)TEST_N*TOP_K*sizeof(cand_t));
            int*nc_s=malloc((size_t)TEST_N*sizeof(int));
            bag_cand_t(*bc)[BAG_TOP_K]=malloc((size_t)M*BAG_TOP_K*sizeof(bag_cand_t));
            int*bn=malloc((size_t)M*sizeof(int));
            long tm=0;
            for(int i=0;i<TEST_N;i++){
                for(int m=0;m<M;m++){vote_masked(votes,i,bag_mask[m]);bn[m]=select_bag_top_k(votes,TRAIN_N,bc[m],BAG_TOP_K);}
                cand_t*ci=pre_s+(size_t)i*TOP_K;
                int nc=merge_bag_cands(bc,bn,M,ci,TOP_K);tm+=nc;
                compute_base(ci,nc,i,cent_test[i],hprof_test+(size_t)i*HP_PAD,divneg_test[i],divneg_cy_test[i]);
                nc_s[i]=nc;
            }
            int cs=run_static_tuned(pre_s,nc_s,qg,tg,nreg,w_c,w_p,w_d,w_g,sc_val,NULL);
            double el=now_sec()-ts;
            double acc=100.0*cs/TEST_N;
            printf("    -> %.2f%% (%d errors, %+d vs A)  %.1f cands  %.2f sec\n",acc,TEST_N-cs,cs-cA,(double)tm/TEST_N,el);
            results[nr]=(sweep_t){M,R,cs,el,(double)tm/TEST_N};
            if(acc>best_acc){best_acc=acc;best_mi=mi;best_ri=ri;}
            nr++;
            free(pre_s);free(nc_s);free(bc);free(bn);
        }
    }

    /* Print sweep table */
    printf("\n  Sweep Summary (accuracy %%):\n\n");
    printf("  %-6s","M\\R");for(int ri=0;ri<n_r;ri++)printf("  R=%-5.1f  ",r_vals[ri]);
    printf("  | best\n  ------");for(int ri=0;ri<n_r;ri++)(void)ri,printf("-----------");printf("--+------\n");
    for(int mi=0;mi<n_m;mi++){
        printf("  M=%-4d",m_vals[mi]);double rb=0;
        for(int ri=0;ri<n_r;ri++){int idx=mi*n_r+ri;double a=100.0*results[idx].correct/TEST_N;printf("  %5.2f%%   ",a);if(a>rb)rb=a;}
        printf("  | %5.2f%%\n",rb);
    }
    printf("\n  Baseline (A):  %.2f%%  (%d errors)\n",100.0*cA/TEST_N,TEST_N-cA);
    printf("  Best sweep:    M=%d R=%.1f -> %.2f%%  (%+.2f pp vs A)\n",
           m_vals[best_mi],r_vals[best_ri],best_acc,best_acc-100.0*cA/TEST_N);

    /* Delta table vs baseline */
    printf("\n  Delta vs baseline (pp):\n\n");
    printf("  %-6s","M\\R");for(int ri=0;ri<n_r;ri++)printf("  R=%-5.1f  ",r_vals[ri]);printf("\n  ------");
    for(int ri=0;ri<n_r;ri++){(void)ri;printf("-----------");}printf("\n");
    for(int mi=0;mi<n_m;mi++){
        printf("  M=%-4d",m_vals[mi]);
        for(int ri=0;ri<n_r;ri++){int idx=mi*n_r+ri;printf("  %+5.2f    ",100.0*(results[idx].correct-cA)/TEST_N);}
        printf("\n");
    }

    printf("\n=== SUMMARY ===\n");
    printf("  Test A (baseline topo9):     %.2f%%  (%d errors)\n",100.0*cA/TEST_N,TEST_N-cA);
    printf("  Test B (M=10 R=0.6 + topo9): %.2f%%  (%d errors, %+d)\n",100.0*cB/TEST_N,TEST_N-cB,cB-cA);
    printf("  Test C best:                 %.2f%%\n",best_acc);
    printf("  Conclusion: bagging + topo9 are %s\n",(cB>cA)?"ADDITIVE":"REDUNDANT (or marginal)");
    printf("\nTotal: %.2f sec\n",now_sec()-t0);

    free(pre_a);free(nc_a);free(preds_a);
    free(pre_b);free(nc_b);free(preds_b);
    free(bcands);free(bnc);free(votes);
    free(gdiv_2x4_train);free(gdiv_2x4_test);free(gdiv_3x3_train);free(gdiv_3x3_test);
    free(cent_train);free(cent_test);free(hprof_train);free(hprof_test);
    free(divneg_train);free(divneg_test);free(divneg_cy_train);free(divneg_cy_test);
    free(idx_pool);free(joint_tr);free(joint_te);
    free(px_tr);free(px_te);free(hg_tr);free(hg_te);free(vg_tr);free(vg_te);
    free(ht_tr);free(ht_te);free(vt_tr);free(vt_te);
    free(tern_train);free(tern_test);free(hgrad_train);free(hgrad_test);free(vgrad_train);free(vgrad_test);
    free(raw_train_img);free(raw_test_img);free(train_labels);free(test_labels);
    return 0;
}
