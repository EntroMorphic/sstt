/*
 * sstt_router_v1.c — Three-Tier Router: Production Edition
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
#define N_CLASSES  10
#define CLS_PAD    16
#define IMG_W      28
#define IMG_H      28
#define PIXELS     784
#define PADDED     800

#define BLKS_PER_ROW 9
#define N_BLOCKS   252
#define SIG_PAD    256
#define BYTE_VALS  256
#define BG_TRANS   13
#define H_TRANS_PER_ROW 8
#define N_HTRANS (H_TRANS_PER_ROW*IMG_H)
#define V_TRANS_PER_COL 27
#define N_VTRANS (BLKS_PER_ROW*V_TRANS_PER_COL)
#define TRANS_PAD 256
#define IG_SCALE 16
#define TOP_K 200

#define GM_BINS 45
#define G_DIM 4
#define TOTAL_BINS (G_DIM*G_DIM*GM_BINS)
#define MAX_REGIONS 16
#define FG_W 30
#define FG_H 30
#define FG_SZ (FG_W*FG_H)
#define CENT_BOTH_OPEN 50
#define CENT_DISAGREE 80
#define HP_PAD 32

static const char *data_dir = "data/";
static double now_sec(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + ts.tv_nsec * 1e-9; }

static uint8_t *raw_train_img, *raw_test_img, *train_labels, *test_labels;
static int8_t *tern_train, *tern_test, *hgrad_train, *hgrad_test, *vgrad_train, *vgrad_test;
static uint8_t *px_tr, *px_te, *hg_tr, *hg_te, *vg_tr, *vg_te, *ht_tr, *ht_te, *vt_tr, *vt_te, *joint_tr, *joint_te;
static uint16_t ig_w[N_BLOCKS]; static uint8_t nbr[BYTE_VALS][8];
static uint32_t idx_off[N_BLOCKS][BYTE_VALS]; static uint16_t idx_sz[N_BLOCKS][BYTE_VALS];
static uint32_t *idx_pool; static uint8_t bg;

static int16_t *cent_train, *cent_test, *hprof_train, *hprof_test;
static int16_t *divneg_train, *divneg_test, *divneg_cy_train, *divneg_cy_test;
static int16_t *gdiv_train, *gdiv_test;

static uint32_t px_map[N_BLOCKS][27][CLS_PAD] __attribute__((aligned(32)));
static uint32_t gm_map[TOTAL_BINS][8][CLS_PAD] __attribute__((aligned(32)));

typedef struct {
    uint32_t id, votes;
    int32_t dot_px, dot_vg;
    int32_t div_sim, cent_sim, prof_sim, gdiv_sim;
    int64_t combined;
} cand_t;

static int *g_hist=NULL; static size_t g_hist_cap=0;

/* === Boilerplate === */
static uint8_t *load_idx(const char*path,uint32_t*cnt,uint32_t*ro,uint32_t*co){FILE*f=fopen(path,"rb");if(!f){fprintf(stderr,"ERR:%s\n",path);exit(1);}uint32_t m,n;if(fread(&m,4,1,f)!=1||fread(&n,4,1,f)!=1){fclose(f);exit(1);}m=__builtin_bswap32(m);n=__builtin_bswap32(n);*cnt=n;size_t s=1;if((m&0xFF)>=3){uint32_t r,c;if(fread(&r,4,1,f)!=1||fread(&c,4,1,f)!=1){fclose(f);exit(1);}r=__builtin_bswap32(r);c=__builtin_bswap32(c);if(ro)*ro=r;if(co)*co=c;s=(size_t)r*c;}else{if(ro)*ro=0;if(co)*co=0;}size_t total=(size_t)n*s;uint8_t*d=malloc(total);if(!d||fread(d,1,total,f)!=total){fclose(f);exit(1);}fclose(f);return d;}
static void load_data(void){uint32_t n,r,c;char p[256];snprintf(p,sizeof(p),"%strain-images-idx3-ubyte",data_dir);raw_train_img=load_idx(p,&n,&r,&c);snprintf(p,sizeof(p),"%strain-labels-idx1-ubyte",data_dir);train_labels=load_idx(p,&n,NULL,NULL);snprintf(p,sizeof(p),"%st10k-images-idx3-ubyte",data_dir);raw_test_img=load_idx(p,&n,&r,&c);snprintf(p,sizeof(p),"%st10k-labels-idx1-ubyte",data_dir);test_labels=load_idx(p,&n,NULL,NULL);}
static inline int8_t ct(int v){return v>0?1:v<0?-1:0;}
static void quant_tern(const uint8_t*src,int8_t*dst,int n){const __m256i bias=_mm256_set1_epi8((char)0x80),thi=_mm256_set1_epi8((char)(170^0x80)),tlo=_mm256_set1_epi8((char)(85^0x80)),one=_mm256_set1_epi8(1);for(int i=0;i<n;i++){const uint8_t*s=src+(size_t)i*PIXELS;int8_t*d=dst+(size_t)i*PADDED;int k;for(k=0;k+32<=PIXELS;k+=32){__m256i px=_mm256_loadu_si256((const __m256i*)(s+k));__m256i sp=_mm256_xor_si256(px,bias);_mm256_storeu_si256((__m256i*)(d+k),_mm256_sub_epi8(_mm256_and_si256(_mm256_cmpgt_epi8(sp,thi),one),_mm256_and_si256(_mm256_cmpgt_epi8(tlo,sp),one)));}for(;k<PIXELS;k++)d[k]=s[k]>170?1:s[k]<85?-1:0;memset(d+PIXELS,0,PADDED-PIXELS);}}
static void gradients(const int8_t*t,int8_t*h,int8_t*v,int n){for(int i=0;i<n;i++){const int8_t*ti=t+(size_t)i*PADDED;int8_t*hi=h+(size_t)i*PADDED;int8_t*vi=v+(size_t)i*PADDED;for(int y=0;y<IMG_H;y++){for(int x=0;x<IMG_W-1;x++)hi[y*IMG_W+x]=ct(ti[y*IMG_W+x+1]-ti[y*IMG_W+x]);hi[y*IMG_W+IMG_W-1]=0;}memset(hi+PIXELS,0,PADDED-PIXELS);for(int y=0;y<IMG_H-1;y++)for(int x=0;x<IMG_W;x++)vi[y*IMG_W+x]=ct(ti[(y+1)*IMG_W+x]-ti[y*IMG_W+x]);memset(vi+(IMG_H-1)*IMG_W,0,IMG_W);memset(vi+PIXELS,0,PADDED-PIXELS);}}
static inline uint8_t benc(int8_t a,int8_t b,int8_t c){return(uint8_t)((a+1)*9+(b+1)*3+(c+1));}
static void block_sigs(const int8_t*data,uint8_t*sigs,int n){for(int i=0;i<n;i++){const int8_t*img=data+(size_t)i*PADDED;uint8_t*sig=sigs+(size_t)i*SIG_PAD;for(int y=0;y<IMG_H;y++)for(int s=0;s<BLKS_PER_ROW;s++){int b=y*IMG_W+s*3;sig[y*BLKS_PER_ROW+s]=benc(img[b],img[b+1],img[b+2]);}memset(sig+N_BLOCKS,0xFF,SIG_PAD-N_BLOCKS);}}
static inline uint8_t tenc(uint8_t a,uint8_t b){int8_t a0=(a/9)-1,a1=((a/3)%3)-1,a2=(a%3)-1,b0=(b/9)-1,b1=((b/3)%3)-1,b2=(b%3)-1;return benc(ct(b0-a0),ct(b1-a1),ct(b2-a2));}
static void trans_fn(const uint8_t*bs,int str,uint8_t*ht,uint8_t*vt,int n){for(int i=0;i<n;i++){const uint8_t*s=bs+(size_t)i*str;uint8_t*h=ht+(size_t)i*TRANS_PAD;uint8_t*v=vt+(size_t)i*TRANS_PAD;for(int y=0;y<IMG_H;y++)for(int ss=0;ss<H_TRANS_PER_ROW;ss++)h[y*H_TRANS_PER_ROW+ss]=tenc(s[y*BLKS_PER_ROW+ss],s[y*BLKS_PER_ROW+ss+1]);memset(h+N_HTRANS,0xFF,TRANS_PAD-N_HTRANS);for(int y=0;y<V_TRANS_PER_COL;y++)for(int ss=0;ss<BLKS_PER_ROW;ss++)v[y*BLKS_PER_ROW+ss]=tenc(s[y*BLKS_PER_ROW+ss],s[(y+1)*BLKS_PER_ROW+ss]);memset(v+N_VTRANS,0xFF,TRANS_PAD-N_VTRANS);}}
static uint8_t enc_d(uint8_t px,uint8_t hg,uint8_t vg,uint8_t ht,uint8_t vt){int ps=((px/9)-1)+(((px/3)%3)-1)+((px%3)-1),hs=((hg/9)-1)+(((hg/3)%3)-1)+((hg%3)-1),vs=((vg/9)-1)+(((vg/3)%3)-1)+((vg%3)-1);uint8_t pc=ps<0?0:ps==0?1:ps<3?2:3,hc=hs<0?0:hs==0?1:hs<3?2:3,vc=vs<0?0:vs==0?1:vs<3?2:3;return pc|(hc<<2)|(vc<<4)|((ht!=BG_TRANS)?1<<6:0)|((vt!=BG_TRANS)?1<<7:0);}
static void joint_sigs_fn(uint8_t*out,int n,const uint8_t*px,const uint8_t*hg,const uint8_t*vg,const uint8_t*ht,const uint8_t*vt){for(int i=0;i<n;i++){const uint8_t*pi=px+(size_t)i*SIG_PAD,*hi=hg+(size_t)i*SIG_PAD,*vi=vg+(size_t)i*SIG_PAD;const uint8_t*hti=ht+(size_t)i*TRANS_PAD,*vti=vt+(size_t)i*TRANS_PAD;uint8_t*oi=out+(size_t)i*SIG_PAD;for(int y=0;y<IMG_H;y++)for(int s=0;s<BLKS_PER_ROW;s++){int k=y*BLKS_PER_ROW+s;uint8_t htb=s>0?hti[y*H_TRANS_PER_ROW+(s-1)]:BG_TRANS;uint8_t vtb=y>0?vti[(y-1)*BLKS_PER_ROW+s]:BG_TRANS;oi[k]=enc_d(pi[k],hi[k],vi[k],htb,vtb);}memset(oi+N_BLOCKS,0xFF,SIG_PAD-N_BLOCKS);}}
static void compute_ig(const uint8_t*sigs,int nv,uint8_t bgv,uint16_t*ig_out){int cc[N_CLASSES]={0};for(int i=0;i<TRAIN_N;i++)cc[train_labels[i]]++;double hc=0;for(int c=0;c<N_CLASSES;c++){double p=(double)cc[c]/TRAIN_N;if(p>0)hc-=p*log2(p);}double raw[N_BLOCKS],mx=0;for(int k=0;k<N_BLOCKS;k++){int*cnt=calloc((size_t)nv*N_CLASSES,sizeof(int));int*v_t=calloc(nv,sizeof(int));for(int i=0;i<TRAIN_N;i++){int v=sigs[(size_t)i*SIG_PAD+k];cnt[v*N_CLASSES+train_labels[i]]++;v_t[v]++;}double hcond=0;for(int v=0;v<nv;v++){if(!v_t[v]||v==(int)bgv)continue;double pv=(double)v_t[v]/TRAIN_N,hv=0;for(int c=0;c<N_CLASSES;c++){double pc=(double)cnt[v*N_CLASSES+c]/v_t[v];if(pc>0)hv-=pc*log2(pc);}hcond+=pv*hv;}raw[k]=hc-hcond;if(raw[k]>mx)mx=raw[k];free(cnt);free(v_t);}for(int k=0;k<N_BLOCKS;k++){ig_out[k]=mx>0?(uint16_t)(raw[k]/mx*IG_SCALE+0.5):1;if(!ig_out[k])ig_out[k]=1;}}
static void build_index(void){long vc[BYTE_VALS]={0};for(int i=0;i<TRAIN_N;i++){const uint8_t*s=joint_tr+(size_t)i*SIG_PAD;for(int k=0;k<N_BLOCKS;k++)vc[s[k]]++;}bg=0;long mc=0;for(int v=0;v<BYTE_VALS;v++)if(vc[v]>mc){mc=vc[v];bg=(uint8_t)v;}compute_ig(joint_tr,BYTE_VALS,bg,ig_w);for(int v=0;v<BYTE_VALS;v++)for(int b=0;b<8;b++)nbr[v][b]=(uint8_t)(v^(1<<b));memset(idx_sz,0,sizeof(idx_sz));for(int i=0;i<TRAIN_N;i++){const uint8_t*s=joint_tr+(size_t)i*SIG_PAD;for(int k=0;k<N_BLOCKS;k++)if(s[k]!=bg)idx_sz[k][s[k]]++;}uint32_t tot=0;for(int k=0;k<N_BLOCKS;k++)for(int v=0;v<BYTE_VALS;v++){idx_off[k][v]=tot;tot+=idx_sz[k][v];}idx_pool=malloc((size_t)tot*sizeof(uint32_t));uint32_t(*wp)[BYTE_VALS]=malloc((size_t)N_BLOCKS*BYTE_VALS*4);memcpy(wp,idx_off,(size_t)N_BLOCKS*BYTE_VALS*4);for(int i=0;i<TRAIN_N;i++){const uint8_t*s=joint_tr+(size_t)i*SIG_PAD;for(int k=0;k<N_BLOCKS;k++)if(s[k]!=bg)idx_pool[wp[k][s[k]]++]=(uint32_t)i;}free(wp);}
static int cmp_votes_d(const void*a,const void*b){return(int)((const cand_t*)b)->votes-(int)((const cand_t*)a)->votes;}
static int cmp_comb_d(const void*a,const void*b){int64_t da=((const cand_t*)a)->combined,db=((const cand_t*)b)->combined;return(db>da)-(db<da);}
static int select_top_k(const uint32_t*votes,int n,cand_t*out,int k){uint32_t mx=0;for(int i=0;i<n;i++)if(votes[i]>mx)mx=votes[i];if(!mx)return 0;if((size_t)(mx+1)>g_hist_cap){g_hist_cap=(size_t)(mx+1)+4096;free(g_hist);g_hist=malloc(g_hist_cap*sizeof(int));}memset(g_hist,0,(mx+1)*sizeof(int));for(int i=0;i<n;i++)if(votes[i])g_hist[votes[i]]++;int cum=0,thr;for(thr=(int)mx;thr>=1;thr--){cum+=g_hist[thr];if(cum>=k)break;}if(thr<1)thr=1;int nc=0;for(int i=0;i<n&&nc<k;i++)if(votes[i]>=(uint32_t)thr){out[nc]=(cand_t){0};out[nc].id=(uint32_t)i;out[nc].votes=votes[i];nc++;}qsort(out,(size_t)nc,sizeof(cand_t),cmp_votes_d);return nc;}
static inline int32_t tdot(const int8_t*a,const int8_t*b){__m256i acc=_mm256_setzero_si256();for(int i=0;i<PADDED;i+=32)acc=_mm256_add_epi8(acc,_mm256_sign_epi8(_mm256_load_si256((const __m256i*)(a+i)),_mm256_load_si256((const __m256i*)(b+i))));__m256i lo=_mm256_cvtepi8_epi16(_mm256_castsi256_si128(acc)),hi=_mm256_cvtepi8_epi16(_mm256_extracti128_si256(acc,1));__m256i s32=_mm256_madd_epi16(_mm256_add_epi16(lo,hi),_mm256_set1_epi16(1));__m128i s=_mm_add_epi32(_mm256_castsi256_si128(s32),_mm256_extracti128_si256(s32,1));s=_mm_hadd_epi32(s,s);s=_mm_hadd_epi32(s,s);return _mm_cvtsi128_si32(s);}
static inline int dc(int d){return d<-2?-2:d>2?2:d;}
static inline uint8_t qc(int c){return c==0?0:c==1?1:c==2?2:c<=4?3:c<=8?4:c<=16?5:6;}
static void get_px_sig(const int8_t *img, uint8_t *sig){for(int y=0;y<IMG_H;y++)for(int s=0;s<BLKS_PER_ROW;s++)sig[y*BLKS_PER_ROW+s]=benc(img[y*IMG_W+s*3],img[y*IMG_W+s*3+1],img[y*IMG_W+s*3+2]);}
static void get_gm_sig(const int8_t *h,const int8_t *v,uint8_t *sig){memset(sig,0,TOTAL_BINS);for(int y=0;y<IMG_H;y++)for(int x=0;x<IMG_W;x++){int bin=(h[y*IMG_W+x]+1)*15+(v[y*IMG_W+x]+1)*5+(dc((h[y*IMG_W+x]-(x>0?h[y*IMG_W+x-1]:0))+(v[y*IMG_W+x]-(y>0?v[(y-1)*IMG_W+x]:0)))+2);sig[(y*G_DIM/IMG_H)*G_DIM*GM_BINS+(x*G_DIM/IMG_W)*GM_BINS+bin]++;}}
static int16_t enclosed_centroid(const int8_t*tern){uint8_t grid[FG_SZ];memset(grid,0,sizeof(grid));for(int y=0;y<IMG_H;y++)for(int x=0;x<IMG_W;x++)grid[(y+1)*FG_W+(x+1)]=(tern[y*IMG_W+x]>0)?1:0;uint8_t visited[FG_SZ];memset(visited,0,sizeof(visited));int stack[FG_SZ];int sp=0;for(int y=0;y<FG_H;y++)for(int x=0;x<FG_W;x++){if(y==0||y==FG_H-1||x==0||x==FG_W-1){int pos=y*FG_W+x;if(!grid[pos]&&!visited[pos]){visited[pos]=1;stack[sp++]=pos;}}}while(sp>0){int pos=stack[--sp];int py=pos/FG_W,px2=pos%FG_W;const int dx[]={0,0,1,-1},dy[]={1,-1,0,0};for(int d=0;d<4;d++){int ny=py+dy[d],nx=px2+dx[d];if(ny<0||ny>=FG_H||nx<0||nx>=FG_W)continue;int npos=ny*FG_W+nx;if(!visited[npos]&&!grid[npos]){visited[npos]=1;stack[sp++]=npos;}}}int sum_y=0,count=0;for(int y=1;y<FG_H-1;y++)for(int x=1;x<FG_W-1;x++){int pos=y*FG_W+x;if(!grid[pos]&&!visited[pos]){sum_y+=(y-1);count++;}}return count>0?(int16_t)(sum_y/count):-1;}
static void hprofile(const int8_t*tern,int16_t*prof){for(int y=0;y<IMG_H;y++){int c=0;for(int x=0;x<IMG_W;x++)c+=(tern[y*IMG_W+x]>0);prof[y]=(int16_t)c;}for(int y=IMG_H;y<HP_PAD;y++)prof[y]=0;}
static void div_features(const int8_t*hg,const int8_t*vg,int16_t*ns,int16_t*cy){int neg_sum=0,neg_ysum=0,neg_cnt=0;for(int y=0;y<IMG_H;y++)for(int x=0;x<IMG_W;x++){int dh=(int)hg[y*IMG_W+x]-(x>0?(int)hg[y*IMG_W+x-1]:0);int dv=(int)vg[y*IMG_W+x]-(y>0?(int)vg[(y-1)*IMG_W+x]:0);int d=dh+dv;if(d<0){neg_sum+=d;neg_ysum+=y;neg_cnt++;}}*ns=(int16_t)(neg_sum<-32767?-32767:neg_sum);*cy=neg_cnt>0?(int16_t)(neg_ysum/neg_cnt):-1;}
static void grid_div_fn(const int8_t*hg,const int8_t*vg,int16_t*out){int nr=8;int regions[8]={0};for(int y=0;y<IMG_H;y++)for(int x=0;x<IMG_W;x++){int dh=(int)hg[y*IMG_W+x]-(x>0?(int)hg[y*IMG_W+x-1]:0);int dv=(int)vg[y*IMG_W+x]-(y>0?(int)vg[(y-1)*IMG_W+x]:0);int d=dh+dv;if(d<0){int ry=y*2/IMG_H;int rx=x*4/IMG_W;regions[ry*4+rx]+=d;}}for(int i=0;i<nr;i++)out[i]=(int16_t)(regions[i]<-32767?-32767:regions[i]);}
static int cmp_i16(const void*a,const void*b){return *(const int16_t*)a-*(const int16_t*)b;}
static int compute_mad(const cand_t*cands,int nc){if(nc<3)return 1000;int16_t vals[TOP_K];for(int j=0;j<nc;j++)vals[j]=divneg_train[cands[j].id];qsort(vals,(size_t)nc,sizeof(int16_t),cmp_i16);int16_t med=vals[nc/2];int16_t devs[TOP_K];for(int j=0;j<nc;j++)devs[j]=(int16_t)abs(vals[j]-med);qsort(devs,(size_t)nc,sizeof(int16_t),cmp_i16);return devs[nc/2]>0?devs[nc/2]:1;}

int main(int argc, char** argv) {
    if(argc > 1) data_dir = argv[1];
    load_data();
    tern_train = aligned_alloc(32, (size_t)TRAIN_N*PADDED); tern_test = aligned_alloc(32, (size_t)TEST_N*PADDED);
    hgrad_train = aligned_alloc(32, (size_t)TRAIN_N*PADDED); hgrad_test = aligned_alloc(32, (size_t)TEST_N*PADDED);
    vgrad_train = aligned_alloc(32, (size_t)TRAIN_N*PADDED); vgrad_test = aligned_alloc(32, (size_t)TEST_N*PADDED);
    quant_tern(raw_train_img, tern_train, TRAIN_N); quant_tern(raw_test_img,  tern_test,  TEST_N);
    gradients(tern_train, hgrad_train, vgrad_train, TRAIN_N); gradients(tern_test, hgrad_test, vgrad_test, TEST_N);
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
    build_index();
    
    memset(px_map, 0, sizeof(px_map)); memset(gm_map, 0, sizeof(gm_map));
    uint8_t psig[N_BLOCKS], gsig[TOTAL_BINS];
    for(int i=0; i<TRAIN_N; i++) {
        int l=train_labels[i]; get_px_sig(tern_train+(size_t)i*PADDED, psig); get_gm_sig(hgrad_train+(size_t)i*PADDED, vgrad_train+(size_t)i*PADDED, gsig);
        for(int k=0;k<N_BLOCKS;k++) px_map[k][psig[k]][l]++;
        for(int k=0;k<TOTAL_BINS;k++) gm_map[k][qc(gsig[k])][l]++;
    }

    cent_train=malloc((size_t)TRAIN_N*2); cent_test=malloc((size_t)TEST_N*2);
    for(int i=0; i<TRAIN_N; i++) cent_train[i] = enclosed_centroid(tern_train + (size_t)i*PADDED);
    for(int i=0; i<TEST_N; i++) cent_test[i] = enclosed_centroid(tern_test + (size_t)i*PADDED);
    divneg_train=malloc((size_t)TRAIN_N*2); divneg_test=malloc((size_t)TEST_N*2);
    divneg_cy_train=malloc((size_t)TRAIN_N*2); divneg_cy_test=malloc((size_t)TEST_N*2);
    for(int i=0; i<TRAIN_N; i++) div_features(hgrad_train+(size_t)i*PADDED, vgrad_train+(size_t)i*PADDED, &divneg_train[i], &divneg_cy_train[i]);
    for(int i=0; i<TEST_N; i++) div_features(hgrad_test+(size_t)i*PADDED, vgrad_test+(size_t)i*PADDED, &divneg_test[i], &divneg_cy_test[i]);
    gdiv_train=malloc((size_t)TRAIN_N*8*2); gdiv_test=malloc((size_t)TEST_N*8*2);
    for(int i=0; i<TRAIN_N; i++) grid_div_fn(hgrad_train+(size_t)i*PADDED, vgrad_train+(size_t)i*PADDED, gdiv_train + (size_t)i*8);
    for(int i=0; i<TEST_N; i++) grid_div_fn(hgrad_test+(size_t)i*PADDED, vgrad_test+(size_t)i*PADDED, gdiv_test + (size_t)i*8);
    hprof_train=malloc((size_t)TRAIN_N*HP_PAD*2); hprof_test=malloc((size_t)TEST_N*HP_PAD*2);
    for(int i=0; i<TRAIN_N; i++) hprofile(tern_train+(size_t)i*PADDED, hprof_train + (size_t)i*HP_PAD);
    for(int i=0; i<TEST_N; i++) hprofile(tern_test+(size_t)i*PADDED, hprof_test + (size_t)i*HP_PAD);

    printf("\n=== SSTT Router v1: Stable Production Edition ===\n\n");
    int correct=0, t1_count=0, t2_count=0, t3_count=0, t1_corr=0, t2_corr=0, t3_corr=0;
    uint32_t *vbuf = calloc(TRAIN_N, 4);
    double t_start = now_sec();
    for(int i=0; i<TEST_N; i++) {
        memset(vbuf, 0, TRAIN_N * 4);
        const uint8_t *sig = joint_te + (size_t)i * SIG_PAD;
        for (int k = 0; k < N_BLOCKS; k++) {
            uint8_t bv = sig[k]; if (bv == bg) continue;
            uint32_t off = idx_off[k][bv]; uint16_t sz = idx_sz[k][bv];
            const uint32_t *ids = idx_pool + off; for (uint16_t j = 0; j < sz; j++) vbuf[ids[j]] += ig_w[k];
        }
        cand_t cands[TOP_K]; int nc = select_top_k(vbuf, TRAIN_N, cands, TOP_K);
        int counts[N_CLASSES] = {0};
        for(int j=0; j<nc; j++) counts[train_labels[cands[j].id]]++;
        int best_c = 0; for(int c=1; c<N_CLASSES; c++) if(counts[c] > counts[best_c]) best_c = c;
        int top_count = counts[best_c];

        int prediction = -1;
        if (top_count >= 198) {
            prediction = best_c; t1_count++; if(prediction == test_labels[i]) t1_corr++;
        } else if (top_count >= 180) {
            uint32_t scores[N_CLASSES] = {0};
            uint8_t psig2[N_BLOCKS], gsig2[TOTAL_BINS];
            get_px_sig(tern_test + (size_t)i*PADDED, psig2); get_gm_sig(hgrad_test + (size_t)i*PADDED, vgrad_test + (size_t)i*PADDED, gsig2);
            for(int k=0;k<N_BLOCKS;k++) if(psig2[k]!=0) for(int c=0;c<10;c++) scores[c]+=px_map[k][psig2[k]][c];
            for(int k=0;k<TOTAL_BINS;k++) if(qc(gsig2[k])!=0) for(int c=0;c<10;c++) scores[c]+=gm_map[k][qc(gsig2[k])][c]/2;
            prediction = 0; for(int c=1;c<10;c++) if(scores[c] > scores[prediction]) prediction = c;
            t2_count++; if(prediction == test_labels[i]) t2_corr++;
        } else {
            int mad = compute_mad(cands, nc);
            int wc=(int)(50*50/(50+mad)), wd=(int)(50*200/(50+mad)), wg=(int)(50*100/(50+mad)), wp=(int)(50*16/(50+mad));
            for(int j=0; j<nc; j++) {
                uint32_t cid = cands[j].id;
                int32_t dpx = tdot(tern_test+(size_t)i*PADDED, tern_train+(size_t)cid*PADDED);
                int32_t dvg = tdot(vgrad_test+(size_t)i*PADDED, vgrad_train+(size_t)cid*PADDED);
                int16_t cc=cent_train[cid], q_cent=cent_test[i]; int32_t cs = (q_cent<0&&cc<0)?50:(q_cent<0||cc<0)?-80:-(int32_t)abs(q_cent-cc);
                int32_t ds = -(int32_t)abs(divneg_test[i] - divneg_train[cid]);
                if(divneg_cy_test[i]>=0 && divneg_cy_train[cid]>=0) ds -= abs(divneg_cy_test[i] - divneg_cy_train[cid])*2;
                int32_t gs=0; for(int r=0;r<8;r++) gs += abs(gdiv_test[i*8+r]-gdiv_train[cid*8+r]);
                int32_t ps=0; for(int y=0;y<IMG_H;y++) ps += (int32_t)hprof_test[i*HP_PAD+y]*hprof_train[cid*HP_PAD+y];
                cands[j].combined = (int64_t)256*dpx + (int64_t)192*dvg + (int64_t)wc*cs + (int64_t)wd*ds + (int64_t)wg*(-gs) + (int64_t)wp*ps;
            }
            qsort(cands, (size_t)nc, sizeof(cand_t), cmp_comb_d);
            int64_t h[10] = {0}; 
            for(int j=0; j<15 && j<nc; j++) {
                uint8_t lbl = train_labels[cands[j].id];
                /* Add topo_w bonus to sequential evidence */
                int64_t evidence = cands[j].combined + 100 * (cands[j].div_sim + cands[j].gdiv_sim);
                h[lbl] += evidence * 2 / (2 + j);
            }
            prediction = 0; for(int c=1; c<10; c++) if(h[c] > h[prediction]) prediction = c;
            t3_count++; if(prediction == test_labels[i]) t3_corr++;
        }
        if (prediction == test_labels[i]) correct++;
        if ((i+1)%2000 == 0) fprintf(stderr, "  %d/%d\r", i+1, TEST_N);
    }
    printf("Accuracy: %.2f%% (%d/%d, Latency %.2f ms)\nCoverage: T1=%.1f%% T2=%.1f%% T3=%.1f%%\nAccuracy: T1=%.2f%% T2=%.2f%% T3=%.2f%%\n", 100.0*correct/TEST_N, correct, TEST_N, (now_sec()-t_start)*1000/TEST_N, 100.0*t1_count/TEST_N, 100.0*t2_count/TEST_N, 100.0*t3_count/TEST_N, t1_count?100.0*t1_corr/t1_count:0, t2_count?100.0*t2_corr/t2_count:0, t3_count?100.0*t3_corr/t3_count:0);
    return 0;
}
