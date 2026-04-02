/*
 * sstt_confidence_map.c — Quantized Confidence Map: Maps All The Way Down
 *
 * Tests whether a precomputed lookup table on quantized ranking-level
 * features can replace the Kalman/Bayesian/CfC computation.
 *
 * Mathematical tests:
 *   1. Mutual information: I(quantized_features; correct/incorrect)
 *   2. Precision-coverage tradeoff: route by map confidence
 *   3. Cross-validated: build map on half, evaluate on other half
 *
 * Build: make sstt_confidence_map
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

/* Map dimensions */
#define AGREE_BINS  11  /* 0..10 */
#define MAD_BINS    6   /* 0-4, 5-9, 10-14, 15-19, 20-29, 30+ */
#define MARGIN_BINS 4   /* quartiles */
#define MAP_SIZE    (AGREE_BINS * MAD_BINS * MARGIN_BINS)

static const char *data_dir="data/";
static double now_sec(void){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return ts.tv_sec+ts.tv_nsec*1e-9;}

/* All boilerplate declarations */
static uint8_t *raw_train_img,*raw_test_img,*train_labels,*test_labels;
static int8_t *tern_train,*tern_test,*hgrad_train,*hgrad_test,*vgrad_train,*vgrad_test;
static uint8_t *px_tr,*px_te,*hg_tr,*hg_te,*vg_tr,*vg_te,*ht_tr,*ht_te,*vt_tr,*vt_te,*joint_tr,*joint_te;
static uint16_t ig_w[N_BLOCKS];static uint8_t nbr[BYTE_VALS][8];
static uint32_t idx_off[N_BLOCKS][BYTE_VALS];static uint16_t idx_sz[N_BLOCKS][BYTE_VALS];
static uint32_t *idx_pool;static uint8_t bg;static int *g_hist=NULL;static size_t g_hist_cap=0;
static int16_t *cent_train,*cent_test;
static int16_t *divneg_train,*divneg_test,*divneg_cy_train,*divneg_cy_test;
static int16_t *gdiv_train,*gdiv_test;

/* All boilerplate functions (same as other topo experiments) */
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
static void build_index(void){long vc[BYTE_VALS]={0};for(int i=0;i<TRAIN_N;i++){const uint8_t*s=joint_tr+(size_t)i*SIG_PAD;for(int k=0;k<N_BLOCKS;k++)vc[s[k]]++;}bg=0;long mc=0;for(int v=0;v<BYTE_VALS;v++)if(vc[v]>mc){mc=vc[v];bg=(uint8_t)v;}compute_ig(joint_tr,BYTE_VALS,bg,ig_w);for(int v=0;v<BYTE_VALS;v++)for(int b=0;b<8;b++)nbr[v][b]=(uint8_t)(v^(1<<b));memset(idx_sz,0,sizeof(idx_sz));for(int i=0;i<TRAIN_N;i++){const uint8_t*s=joint_tr+(size_t)i*SIG_PAD;for(int k=0;k<N_BLOCKS;k++)if(s[k]!=bg)idx_sz[k][s[k]]++;}uint32_t tot=0;for(int k=0;k<N_BLOCKS;k++)for(int v=0;v<BYTE_VALS;v++){idx_off[k][v]=tot;tot+=idx_sz[k][v];}idx_pool=malloc((size_t)tot*sizeof(uint32_t));uint32_t(*wp)[BYTE_VALS]=malloc((size_t)N_BLOCKS*BYTE_VALS*4);memcpy(wp,idx_off,(size_t)N_BLOCKS*BYTE_VALS*4);for(int i=0;i<TRAIN_N;i++){const uint8_t*s=joint_tr+(size_t)i*SIG_PAD;for(int k=0;k<N_BLOCKS;k++)if(s[k]!=bg)idx_pool[wp[k][s[k]]++]=(uint32_t)i;}free(wp);printf("  Index: %u entries\n",tot);}
static int16_t enclosed_centroid(const int8_t*tern){uint8_t grid[FG_SZ];memset(grid,0,sizeof(grid));for(int y=0;y<IMG_H;y++)for(int x=0;x<IMG_W;x++)grid[(y+1)*FG_W+(x+1)]=(tern[y*IMG_W+x]>0)?1:0;uint8_t visited[FG_SZ];memset(visited,0,sizeof(visited));int stack[FG_SZ];int sp=0;for(int y=0;y<FG_H;y++)for(int x=0;x<FG_W;x++){if(y==0||y==FG_H-1||x==0||x==FG_W-1){int pos=y*FG_W+x;if(!grid[pos]&&!visited[pos]){visited[pos]=1;stack[sp++]=pos;}}}while(sp>0){int pos=stack[--sp];int py=pos/FG_W,px2=pos%FG_W;const int dx[]={0,0,1,-1},dy[]={1,-1,0,0};for(int d=0;d<4;d++){int ny=py+dy[d],nx=px2+dx[d];if(ny<0||ny>=FG_H||nx<0||nx>=FG_W)continue;int npos=ny*FG_W+nx;if(!visited[npos]&&!grid[npos]){visited[npos]=1;stack[sp++]=npos;}}}int sum_y=0,count=0;for(int y=1;y<FG_H-1;y++)for(int x=1;x<FG_W-1;x++){int pos=y*FG_W+x;if(!grid[pos]&&!visited[pos]){sum_y+=(y-1);count++;}}return count>0?(int16_t)(sum_y/count):-1;}
static void compute_centroid_all(const int8_t*tern,int16_t*out,int n){for(int i=0;i<n;i++)out[i]=enclosed_centroid(tern+(size_t)i*PADDED);}
static void div_features(const int8_t*hg,const int8_t*vg,int16_t*ns,int16_t*cy){int neg_sum=0,neg_ysum=0,neg_cnt=0;for(int y=0;y<IMG_H;y++)for(int x=0;x<IMG_W;x++){int dh=(int)hg[y*IMG_W+x]-(x>0?(int)hg[y*IMG_W+x-1]:0);int dv=(int)vg[y*IMG_W+x]-(y>0?(int)vg[(y-1)*IMG_W+x]:0);int d=dh+dv;if(d<0){neg_sum+=d;neg_ysum+=y;neg_cnt++;}}*ns=(int16_t)(neg_sum<-32767?-32767:neg_sum);*cy=neg_cnt>0?(int16_t)(neg_ysum/neg_cnt):-1;}
static void compute_div_all(const int8_t*hg,const int8_t*vg,int16_t*ns,int16_t*cy,int n){for(int i=0;i<n;i++)div_features(hg+(size_t)i*PADDED,vg+(size_t)i*PADDED,&ns[i],&cy[i]);}
static inline int32_t tdot(const int8_t*a,const int8_t*b){__m256i acc=_mm256_setzero_si256();for(int i=0;i<PADDED;i+=32)acc=_mm256_add_epi8(acc,_mm256_sign_epi8(_mm256_load_si256((const __m256i*)(a+i)),_mm256_load_si256((const __m256i*)(b+i))));__m256i lo=_mm256_cvtepi8_epi16(_mm256_castsi256_si128(acc)),hi=_mm256_cvtepi8_epi16(_mm256_extracti128_si256(acc,1));__m256i s32=_mm256_madd_epi16(_mm256_add_epi16(lo,hi),_mm256_set1_epi16(1));__m128i s=_mm_add_epi32(_mm256_castsi256_si128(s32),_mm256_extracti128_si256(s32,1));s=_mm_hadd_epi32(s,s);s=_mm_hadd_epi32(s,s);return _mm_cvtsi128_si32(s);}
static void vote(uint32_t*votes,int img){memset(votes,0,TRAIN_N*sizeof(uint32_t));const uint8_t*sig=joint_te+(size_t)img*SIG_PAD;for(int k=0;k<N_BLOCKS;k++){uint8_t bv=sig[k];if(bv==bg)continue;uint16_t w=ig_w[k],wh=w>1?w/2:1;{uint32_t off=idx_off[k][bv];uint16_t sz=idx_sz[k][bv];const uint32_t*ids=idx_pool+off;for(uint16_t j=0;j<sz;j++)votes[ids[j]]+=w;}for(int nb=0;nb<8;nb++){uint8_t nv=nbr[bv][nb];if(nv==bg)continue;uint32_t noff=idx_off[k][nv];uint16_t nsz=idx_sz[k][nv];const uint32_t*nids=idx_pool+noff;for(uint16_t j=0;j<nsz;j++)votes[nids[j]]+=wh;}}}
static void grid_div(const int8_t*hg,const int8_t*vg,int grow,int gcol,int16_t*out){int nr=grow*gcol;int regions[MAX_REGIONS];memset(regions,0,sizeof(int)*nr);for(int y=0;y<IMG_H;y++)for(int x=0;x<IMG_W;x++){int dh=(int)hg[y*IMG_W+x]-(x>0?(int)hg[y*IMG_W+x-1]:0);int dv=(int)vg[y*IMG_W+x]-(y>0?(int)vg[(y-1)*IMG_W+x]:0);int d=dh+dv;if(d<0){int ry=y*grow/IMG_H;int rx=x*gcol/IMG_W;if(ry>=grow)ry=grow-1;if(rx>=gcol)rx=gcol-1;regions[ry*gcol+rx]+=d;}}for(int i=0;i<nr;i++)out[i]=(int16_t)(regions[i]<-32767?-32767:regions[i]);}
static void compute_grid_div(const int8_t*hg,const int8_t*vg,int grow,int gcol,int16_t*out,int n){for(int i=0;i<n;i++)grid_div(hg+(size_t)i*PADDED,vg+(size_t)i*PADDED,grow,gcol,out+(size_t)i*MAX_REGIONS);}

/* ================================================================
 *  Candidate + classification (simplified from topo experiments)
 * ================================================================ */
typedef struct {
    uint32_t id, votes;
    int32_t dot_px, dot_vg, div_sim, cent_sim, gdiv_sim;
    int64_t combined;
} cand_t;

static int cmp_votes_d(const void*a,const void*b){return(int)((const cand_t*)b)->votes-(int)((const cand_t*)a)->votes;}
static int cmp_comb_d(const void*a,const void*b){int64_t da=((const cand_t*)a)->combined,db=((const cand_t*)b)->combined;return(db>da)-(db<da);}
static int select_top_k(const uint32_t*votes,int n,cand_t*out,int k){uint32_t mx=0;for(int i=0;i<n;i++)if(votes[i]>mx)mx=votes[i];if(!mx)return 0;if((size_t)(mx+1)>g_hist_cap){g_hist_cap=(size_t)(mx+1)+4096;free(g_hist);g_hist=malloc(g_hist_cap*sizeof(int));}memset(g_hist,0,(mx+1)*sizeof(int));for(int i=0;i<n;i++)if(votes[i])g_hist[votes[i]]++;int cum=0,thr;for(thr=(int)mx;thr>=1;thr--){cum+=g_hist[thr];if(cum>=k)break;}if(thr<1)thr=1;int nc=0;for(int i=0;i<n&&nc<k;i++)if(votes[i]>=(uint32_t)thr){out[nc]=(cand_t){0};out[nc].id=(uint32_t)i;out[nc].votes=votes[i];nc++;}qsort(out,(size_t)nc,sizeof(cand_t),cmp_votes_d);return nc;}

static int cmp_i16(const void*a,const void*b){return *(const int16_t*)a-*(const int16_t*)b;}

/* Quantization functions */
static int q_agree(int agree) { return agree > 10 ? 10 : agree; }
static int q_mad(int mad) {
    if(mad < 5) return 0;
    if(mad < 10) return 1;
    if(mad < 15) return 2;
    if(mad < 20) return 3;
    if(mad < 30) return 4;
    return 5;
}
static int q_margin(int32_t margin, int32_t *bounds) {
    for(int i=0; i<MARGIN_BINS-1; i++)
        if(margin <= bounds[i]) return i;
    return MARGIN_BINS-1;
}
static int map_idx(int qa, int qm, int qg) {
    return qa * MAD_BINS * MARGIN_BINS + qm * MARGIN_BINS + qg;
}

/* ================================================================
 *  Main
 * ================================================================ */
int main(int argc, char**argv) {
    double t0 = now_sec();
    if(argc>1){data_dir=argv[1];size_t l=strlen(data_dir);if(l&&data_dir[l-1]!='/'){char*buf=malloc(l+2);memcpy(buf,data_dir,l);buf[l]='/';buf[l+1]='\0';data_dir=buf;}}
    int is_fashion=strstr(data_dir,"fashion")!=NULL;
    const char*ds=is_fashion?"Fashion-MNIST":"MNIST";
    printf("=== SSTT Confidence Map (%s) ===\n\n",ds);

    /* Setup */
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
    cent_train=malloc((size_t)TRAIN_N*2);cent_test=malloc((size_t)TEST_N*2);
    compute_centroid_all(tern_train,cent_train,TRAIN_N);compute_centroid_all(tern_test,cent_test,TEST_N);
    divneg_train=malloc((size_t)TRAIN_N*2);divneg_test=malloc((size_t)TEST_N*2);
    divneg_cy_train=malloc((size_t)TRAIN_N*2);divneg_cy_test=malloc((size_t)TEST_N*2);
    compute_div_all(hgrad_train,vgrad_train,divneg_train,divneg_cy_train,TRAIN_N);
    compute_div_all(hgrad_test,vgrad_test,divneg_test,divneg_cy_test,TEST_N);
    int gr=is_fashion?3:2,gc=is_fashion?3:4;int nreg=gr*gc;
    gdiv_train=malloc((size_t)TRAIN_N*MAX_REGIONS*2);gdiv_test=malloc((size_t)TEST_N*MAX_REGIONS*2);
    compute_grid_div(hgrad_train,vgrad_train,gr,gc,gdiv_train,TRAIN_N);
    compute_grid_div(hgrad_test,vgrad_test,gr,gc,gdiv_test,TEST_N);

    int w_c=is_fashion?25:50, w_d=200, w_g=is_fashion?200:100, sc=50;

    /* ================================================================
     *  Phase 1: Classify all test images, collect profiles
     * ================================================================ */
    printf("\nClassifying...\n");
    typedef struct { int pred, correct, agree; int mad; int32_t margin; } profile_t;
    profile_t *prof = calloc(TEST_N, sizeof(profile_t));
    uint32_t *vbuf = calloc(TRAIN_N, sizeof(uint32_t));
    cand_t cands[TOP_K];
    int total_correct = 0;

    for(int i=0; i<TEST_N; i++) {
        vote(vbuf, i);
        int nc = select_top_k(vbuf, TRAIN_N, cands, TOP_K);
        const int8_t*tp=tern_test+(size_t)i*PADDED,*tv=vgrad_test+(size_t)i*PADDED;
        int16_t q_dn=divneg_test[i],q_dc=divneg_cy_test[i],q_c=cent_test[i];
        const int16_t*q_gd=gdiv_test+(size_t)i*MAX_REGIONS;
        for(int j=0;j<nc;j++){
            uint32_t id=cands[j].id;
            cands[j].dot_px=tdot(tp,tern_train+(size_t)id*PADDED);
            cands[j].dot_vg=tdot(tv,vgrad_train+(size_t)id*PADDED);
            int16_t cdn=divneg_train[id],cdc=divneg_cy_train[id];
            int32_t ds=-(int32_t)abs(q_dn-cdn);if(q_dc>=0&&cdc>=0)ds-=(int32_t)abs(q_dc-cdc)*2;else if((q_dc<0)!=(cdc<0))ds-=10;
            cands[j].div_sim=ds;
            int16_t cc=cent_train[id];if(q_c<0&&cc<0)cands[j].cent_sim=CENT_BOTH_OPEN;else if(q_c<0||cc<0)cands[j].cent_sim=-CENT_DISAGREE;else cands[j].cent_sim=-(int32_t)abs(q_c-cc);
            const int16_t*cg=gdiv_train+(size_t)id*MAX_REGIONS;int32_t gl=0;for(int r=0;r<nreg;r++)gl+=abs(q_gd[r]-cg[r]);cands[j].gdiv_sim=-gl;
        }
        /* MAD */
        int mad=1;
        if(nc>=3){int16_t vals[TOP_K];for(int j=0;j<nc;j++)vals[j]=divneg_train[cands[j].id];
            qsort(vals,(size_t)nc,sizeof(int16_t),cmp_i16);int16_t med=vals[nc/2];
            int16_t devs[TOP_K];for(int j=0;j<nc;j++)devs[j]=(int16_t)abs(vals[j]-med);
            qsort(devs,(size_t)nc,sizeof(int16_t),cmp_i16);mad=devs[nc/2]>0?devs[nc/2]:1;}
        /* Score + sort */
        int64_t s64=(int64_t)sc;
        int wd2=(int)(s64*w_d/(s64+mad)),wc2=(int)(s64*w_c/(s64+mad)),wg2=(int)(s64*w_g/(s64+mad));
        for(int j=0;j<nc;j++)cands[j].combined=(int64_t)256*cands[j].dot_px+(int64_t)192*cands[j].dot_vg+(int64_t)wd2*cands[j].div_sim+(int64_t)wc2*cands[j].cent_sim+(int64_t)wg2*cands[j].gdiv_sim;
        qsort(cands,(size_t)nc,sizeof(cand_t),cmp_comb_d);
        /* Bayesian sequential */
        int64_t state[N_CLASSES];memset(state,0,sizeof(state));
        int ks=nc<15?nc:15;
        for(int j=0;j<ks;j++){uint8_t lbl=train_labels[cands[j].id];state[lbl]+=cands[j].combined*2/(2+j);}
        int pred=0;for(int c=1;c<N_CLASSES;c++)if(state[c]>state[pred])pred=c;
        int correct=(pred==test_labels[i]);
        if(correct) total_correct++;
        /* Agreement */
        int k10=nc<10?nc:10;int top1_cls=train_labels[cands[0].id];int nsame=0;
        for(int j=0;j<k10;j++)if(train_labels[cands[j].id]==top1_cls)nsame++;
        /* Margin */
        int32_t margin=nc>=2?(int32_t)((cands[0].combined-cands[1].combined)/1000):0;

        prof[i].pred=pred; prof[i].correct=correct;
        prof[i].agree=nsame; prof[i].mad=mad; prof[i].margin=margin;

        if((i+1)%2000==0)fprintf(stderr,"  %d/%d\r",i+1,TEST_N);
    }
    fprintf(stderr,"\n");
    free(vbuf);
    int n_err=TEST_N-total_correct;
    printf("  Accuracy: %.2f%% (%d errors)\n\n",100.0*total_correct/TEST_N,n_err);

    /* Compute margin quartile bounds */
    int32_t *margins = malloc(TEST_N*sizeof(int32_t));
    for(int i=0;i<TEST_N;i++) margins[i]=prof[i].margin;
    /* Sort for quartiles */
    for(int i=0;i<TEST_N-1;i++)for(int j=i+1;j<TEST_N;j++)if(margins[j]<margins[i]){int32_t t=margins[i];margins[i]=margins[j];margins[j]=t;}
    int32_t margin_bounds[MARGIN_BINS-1];
    for(int q=0;q<MARGIN_BINS-1;q++) margin_bounds[q]=margins[(q+1)*TEST_N/MARGIN_BINS];
    free(margins);
    printf("  Margin quartile bounds:");
    for(int q=0;q<MARGIN_BINS-1;q++) printf(" %d",margin_bounds[q]);
    printf("\n\n");

    /* ================================================================
     *  Phase 2: Build confidence map (cross-validated)
     *  Split test set in half: odd indices build map, even evaluate.
     *  Then swap. Average the two.
     * ================================================================ */
    printf("=== Phase 2: Cross-Validated Confidence Map ===\n\n");

    for(int fold=0; fold<2; fold++) {
        /* Build map from one half */
        int map_correct[MAP_SIZE], map_total[MAP_SIZE];
        memset(map_correct,0,sizeof(map_correct));
        memset(map_total,0,sizeof(map_total));

        for(int i=0; i<TEST_N; i++) {
            if((i%2) != fold) continue; /* this half builds the map */
            int qa=q_agree(prof[i].agree);
            int qm=q_mad(prof[i].mad);
            int qg=q_margin(prof[i].margin, margin_bounds);
            int idx=map_idx(qa,qm,qg);
            map_total[idx]++;
            if(prof[i].correct) map_correct[idx]++;
        }

        /* Evaluate on the other half */
        int eval_accept=0, eval_accept_correct=0;
        int eval_reject=0, eval_reject_correct=0;
        int eval_total=0;

        /* Sweep confidence thresholds */
        printf("  Fold %d (build=%s, eval=%s):\n", fold,
               fold==0?"odd":"even", fold==0?"even":"odd");
        printf("  %-12s %8s %8s %10s %8s %10s\n",
               "Threshold","Accept","AccAcc","Coverage","Reject","RejAcc");

        double thresholds[] = {0.99, 0.98, 0.97, 0.95, 0.90, 0.80, 0.50, 0.0};
        for(int ti=0; ti<8; ti++) {
            double thr = thresholds[ti];
            int acc=0,acc_c=0,rej=0,rej_c=0;
            for(int i=0; i<TEST_N; i++) {
                if((i%2) == fold) continue; /* this half evaluates */
                int qa=q_agree(prof[i].agree);
                int qm=q_mad(prof[i].mad);
                int qg=q_margin(prof[i].margin, margin_bounds);
                int idx=map_idx(qa,qm,qg);
                double cell_acc = map_total[idx] > 0 ?
                    (double)map_correct[idx]/map_total[idx] : 0.5;
                if(cell_acc >= thr) {
                    acc++; if(prof[i].correct) acc_c++;
                } else {
                    rej++; if(prof[i].correct) rej_c++;
                }
            }
            int tot=acc+rej;
            printf("  >= %.0f%%       %8d %7.2f%% %9.1f%% %8d %7.2f%%\n",
                   thr*100, acc, acc>0?100.0*acc_c/acc:0.0,
                   100.0*acc/tot, rej, rej>0?100.0*rej_c/rej:0.0);
        }
        printf("\n");
    }

    /* ================================================================
     *  Phase 3: Mutual Information
     * ================================================================ */
    printf("=== Phase 3: Mutual Information ===\n\n");
    {
        /* I(Q; Y) where Q = quantized features, Y = correct/incorrect */
        /* H(Y) */
        double p_correct = (double)total_correct/TEST_N;
        double p_error = (double)n_err/TEST_N;
        double H_Y = 0;
        if(p_correct>0) H_Y -= p_correct*log2(p_correct);
        if(p_error>0) H_Y -= p_error*log2(p_error);

        /* H(Y|Q) — conditional entropy */
        int cell_total[MAP_SIZE], cell_correct[MAP_SIZE];
        memset(cell_total,0,sizeof(cell_total));
        memset(cell_correct,0,sizeof(cell_correct));
        for(int i=0;i<TEST_N;i++){
            int qa=q_agree(prof[i].agree);
            int qm=q_mad(prof[i].mad);
            int qg=q_margin(prof[i].margin, margin_bounds);
            int idx=map_idx(qa,qm,qg);
            cell_total[idx]++;
            if(prof[i].correct) cell_correct[idx]++;
        }
        double H_Y_Q = 0;
        int n_occupied = 0;
        for(int idx=0; idx<MAP_SIZE; idx++) {
            if(cell_total[idx] == 0) continue;
            n_occupied++;
            double p_q = (double)cell_total[idx]/TEST_N;
            double p_c = (double)cell_correct[idx]/cell_total[idx];
            double p_e = 1.0 - p_c;
            double h = 0;
            if(p_c>0) h -= p_c*log2(p_c);
            if(p_e>0) h -= p_e*log2(p_e);
            H_Y_Q += p_q * h;
        }
        double MI = H_Y - H_Y_Q;
        printf("  H(Y) = %.4f bits  (baseline entropy of correct/incorrect)\n", H_Y);
        printf("  H(Y|Q) = %.4f bits  (residual after knowing quantized features)\n", H_Y_Q);
        printf("  I(Q;Y) = %.4f bits  (mutual information)\n", MI);
        printf("  Reduction: %.1f%% of uncertainty eliminated\n", 100.0*MI/H_Y);
        printf("  Map cells: %d occupied out of %d possible\n\n", n_occupied, MAP_SIZE);

        /* Per-axis MI */
        printf("  Per-axis mutual information:\n");
        /* Agreement alone */
        { double h_cond=0;
          int at[AGREE_BINS]={0},ac[AGREE_BINS]={0};
          for(int i=0;i<TEST_N;i++){int qa=q_agree(prof[i].agree);at[qa]++;if(prof[i].correct)ac[qa]++;}
          for(int b=0;b<AGREE_BINS;b++){if(!at[b])continue;double pq=(double)at[b]/TEST_N;double pc=(double)ac[b]/at[b];double pe=1-pc;double h=0;if(pc>0)h-=pc*log2(pc);if(pe>0)h-=pe*log2(pe);h_cond+=pq*h;}
          printf("    Agreement alone: I = %.4f bits (%.1f%%)\n",H_Y-h_cond,100*(H_Y-h_cond)/H_Y);
        }
        /* MAD alone */
        { double h_cond=0;
          int at[MAD_BINS]={0},ac[MAD_BINS]={0};
          for(int i=0;i<TEST_N;i++){int qm=q_mad(prof[i].mad);at[qm]++;if(prof[i].correct)ac[qm]++;}
          for(int b=0;b<MAD_BINS;b++){if(!at[b])continue;double pq=(double)at[b]/TEST_N;double pc=(double)ac[b]/at[b];double pe=1-pc;double h=0;if(pc>0)h-=pc*log2(pc);if(pe>0)h-=pe*log2(pe);h_cond+=pq*h;}
          printf("    MAD alone:       I = %.4f bits (%.1f%%)\n",H_Y-h_cond,100*(H_Y-h_cond)/H_Y);
        }
        /* Margin alone */
        { double h_cond=0;
          int at[MARGIN_BINS]={0},ac[MARGIN_BINS]={0};
          for(int i=0;i<TEST_N;i++){int qg=q_margin(prof[i].margin,margin_bounds);at[qg]++;if(prof[i].correct)ac[qg]++;}
          for(int b=0;b<MARGIN_BINS;b++){if(!at[b])continue;double pq=(double)at[b]/TEST_N;double pc=(double)ac[b]/at[b];double pe=1-pc;double h=0;if(pc>0)h-=pc*log2(pc);if(pe>0)h-=pe*log2(pe);h_cond+=pq*h;}
          printf("    Margin alone:    I = %.4f bits (%.1f%%)\n",H_Y-h_cond,100*(H_Y-h_cond)/H_Y);
        }
        printf("    Combined:        I = %.4f bits (%.1f%%)\n\n",MI,100*MI/H_Y);
    }

    /* ================================================================
     *  Phase 4: Populated map cells (the actual lookup table)
     * ================================================================ */
    printf("=== Phase 4: The Map (top cells by volume) ===\n\n");
    {
        int cell_total[MAP_SIZE], cell_correct[MAP_SIZE];
        memset(cell_total,0,sizeof(cell_total));
        memset(cell_correct,0,sizeof(cell_correct));
        for(int i=0;i<TEST_N;i++){
            int qa=q_agree(prof[i].agree);
            int qm=q_mad(prof[i].mad);
            int qg=q_margin(prof[i].margin, margin_bounds);
            int idx=map_idx(qa,qm,qg);
            cell_total[idx]++;
            if(prof[i].correct) cell_correct[idx]++;
        }
        typedef struct{int idx,total,correct;}cell_t;
        cell_t cells[MAP_SIZE]; int nc2=0;
        for(int idx=0;idx<MAP_SIZE;idx++)
            if(cell_total[idx]>0) cells[nc2++]=(cell_t){idx,cell_total[idx],cell_correct[idx]};
        /* Sort by volume */
        for(int i=0;i<nc2-1;i++)for(int j=i+1;j<nc2;j++)if(cells[j].total>cells[i].total){cell_t t=cells[i];cells[i]=cells[j];cells[j]=t;}

        printf("  %-8s %-6s %-8s %8s %8s %8s\n","Agree","MAD","Margin","Count","Correct","ErrRate");
        int shown=0,cum_total=0,cum_correct=0;
        for(int ci=0;ci<nc2&&shown<25;ci++){
            int idx=cells[ci].idx;
            int qa=idx/(MAD_BINS*MARGIN_BINS);
            int qm=(idx/MARGIN_BINS)%MAD_BINS;
            int qg=idx%MARGIN_BINS;
            const char*mad_labels[]={"0-4","5-9","10-14","15-19","20-29","30+"};
            printf("  %-8d %-6s Q%-7d %8d %8d %7.1f%%\n",
                   qa,mad_labels[qm],qg+1,cells[ci].total,cells[ci].correct,
                   100.0*(cells[ci].total-cells[ci].correct)/cells[ci].total);
            cum_total+=cells[ci].total; cum_correct+=cells[ci].correct;
            shown++;
        }
        printf("  ...\n  Top %d cells: %d images (%.1f%%), %.2f%% accuracy\n\n",
               shown,cum_total,100.0*cum_total/TEST_N,100.0*cum_correct/cum_total);
    }

    printf("Total: %.2f sec\n",now_sec()-t0);
    free(prof);
    free(cent_train);free(cent_test);free(divneg_train);free(divneg_test);
    free(divneg_cy_train);free(divneg_cy_test);free(gdiv_train);free(gdiv_test);
    free(idx_pool);free(joint_tr);free(joint_te);
    free(px_tr);free(px_te);free(hg_tr);free(hg_te);free(vg_tr);free(vg_te);
    free(ht_tr);free(ht_te);free(vt_tr);free(vt_te);
    free(tern_train);free(tern_test);free(hgrad_train);free(hgrad_test);
    free(vgrad_train);free(vgrad_test);
    free(raw_train_img);free(raw_test_img);free(train_labels);free(test_labels);
    return 0;
}
