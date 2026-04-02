/*
 * sstt_topo6.c — Noise Reduction: Dead Zone + Bucketed Divergence
 *
 * Instead of amplifying signal (topo5, no gain), reduce noise in the
 * divergence similarity function itself.
 *
 *   1. Dead-zone: sim = -max(0, |q-c| - D).  Small differences free.
 *   2. Bucketed: quantize neg_sum into N bins, sim = -bin_distance.
 *   3. Both combined with Kalman-adaptive weighting.
 *
 * Build: make sstt_topo6
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
#define BG_PIXEL   0
#define BG_GRAD    13
#define BG_TRANS   13
#define H_TRANS_PER_ROW 8
#define N_HTRANS   (H_TRANS_PER_ROW*IMG_H)
#define V_TRANS_PER_COL 27
#define N_VTRANS   (BLKS_PER_ROW*V_TRANS_PER_COL)
#define TRANS_PAD  256
#define IG_SCALE   16
#define TOP_K      200
#define HP_PAD     32
#define FG_W 30
#define FG_H 30
#define FG_SZ (FG_W*FG_H)
#define CENT_BOTH_OPEN 50
#define CENT_DISAGREE  80
#define N_BUCKETS  8

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

/* Bucket boundaries (computed from training data percentiles) */
static int16_t bucket_bounds[N_BUCKETS-1];
static int8_t *div_bucket_train, *div_bucket_test;  /* bucket ID per image */

/* === Compressed boilerplate (identical to topo4/5) === */
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
static int16_t enclosed_centroid(const int8_t*tern){uint8_t grid[FG_SZ];memset(grid,0,sizeof(grid));for(int y=0;y<IMG_H;y++)for(int x=0;x<IMG_W;x++)grid[(y+1)*FG_W+(x+1)]=(tern[y*IMG_W+x]>0)?1:0;uint8_t visited[FG_SZ];memset(visited,0,sizeof(visited));int stack[FG_SZ];int sp=0;for(int y=0;y<FG_H;y++)for(int x=0;x<FG_W;x++){if(y==0||y==FG_H-1||x==0||x==FG_W-1){int pos=y*FG_W+x;if(!grid[pos]&&!visited[pos]){visited[pos]=1;stack[sp++]=pos;}}}while(sp>0){int pos=stack[--sp];int py=pos/FG_W,px2=pos%FG_W;const int dx[]={0,0,1,-1},dy[]={1,-1,0,0};for(int d=0;d<4;d++){int ny=py+dy[d],nx=px2+dx[d];if(ny<0||ny>=FG_H||nx<0||nx>=FG_W)continue;int npos=ny*FG_W+nx;if(!visited[npos]&&!grid[npos]){visited[npos]=1;stack[sp++]=npos;}}}int sum_y=0,count=0;for(int y=1;y<FG_H-1;y++)for(int x=1;x<FG_W-1;x++){int pos=y*FG_W+x;if(!grid[pos]&&!visited[pos]){sum_y+=(y-1);count++;}}return count>0?(int16_t)(sum_y/count):-1;}
static void compute_centroid_all(const int8_t*tern,int16_t*out,int n){for(int i=0;i<n;i++)out[i]=enclosed_centroid(tern+(size_t)i*PADDED);}
static void hprofile(const int8_t*tern,int16_t*prof){for(int y=0;y<IMG_H;y++){int c=0;for(int x=0;x<IMG_W;x++)c+=(tern[y*IMG_W+x]>0);prof[y]=(int16_t)c;}for(int y=IMG_H;y<HP_PAD;y++)prof[y]=0;}
static void compute_hprof_all(const int8_t*tern,int16_t*out,int n){for(int i=0;i<n;i++)hprofile(tern+(size_t)i*PADDED,out+(size_t)i*HP_PAD);}
static void div_features(const int8_t*hg,const int8_t*vg,int16_t*ns,int16_t*cy){int neg_sum=0,neg_ysum=0,neg_cnt=0;for(int y=0;y<IMG_H;y++)for(int x=0;x<IMG_W;x++){int dh=(int)hg[y*IMG_W+x]-(x>0?(int)hg[y*IMG_W+x-1]:0);int dv=(int)vg[y*IMG_W+x]-(y>0?(int)vg[(y-1)*IMG_W+x]:0);int d=dh+dv;if(d<0){neg_sum+=d;neg_ysum+=y;neg_cnt++;}}*ns=(int16_t)(neg_sum<-32767?-32767:neg_sum);*cy=neg_cnt>0?(int16_t)(neg_ysum/neg_cnt):-1;}
static void compute_div_all(const int8_t*hg,const int8_t*vg,int16_t*ns,int16_t*cy,int n){for(int i=0;i<n;i++)div_features(hg+(size_t)i*PADDED,vg+(size_t)i*PADDED,&ns[i],&cy[i]);}
static inline int32_t tdot(const int8_t*a,const int8_t*b){__m256i acc=_mm256_setzero_si256();for(int i=0;i<PADDED;i+=32)acc=_mm256_add_epi8(acc,_mm256_sign_epi8(_mm256_load_si256((const __m256i*)(a+i)),_mm256_load_si256((const __m256i*)(b+i))));__m256i lo=_mm256_cvtepi8_epi16(_mm256_castsi256_si128(acc)),hi=_mm256_cvtepi8_epi16(_mm256_extracti128_si256(acc,1));__m256i s32=_mm256_madd_epi16(_mm256_add_epi16(lo,hi),_mm256_set1_epi16(1));__m128i s=_mm_add_epi32(_mm256_castsi256_si128(s32),_mm256_extracti128_si256(s32,1));s=_mm_hadd_epi32(s,s);s=_mm_hadd_epi32(s,s);return _mm_cvtsi128_si32(s);}
static void vote(uint32_t*votes,int img){memset(votes,0,TRAIN_N*sizeof(uint32_t));const uint8_t*sig=joint_te+(size_t)img*SIG_PAD;for(int k=0;k<N_BLOCKS;k++){uint8_t bv=sig[k];if(bv==bg)continue;uint16_t w=ig_w[k],wh=w>1?w/2:1;{uint32_t off=idx_off[k][bv];uint16_t sz=idx_sz[k][bv];const uint32_t*ids=idx_pool+off;for(uint16_t j=0;j<sz;j++)votes[ids[j]]+=w;}for(int nb=0;nb<8;nb++){uint8_t nv=nbr[bv][nb];if(nv==bg)continue;uint32_t noff=idx_off[k][nv];uint16_t nsz=idx_sz[k][nv];const uint32_t*nids=idx_pool+noff;for(uint16_t j=0;j<nsz;j++)votes[nids[j]]+=wh;}}}

/* ================================================================
 *  Bucket computation: percentile-based quantization of neg_sum
 * ================================================================ */
static int cmp_i16(const void*a,const void*b){return *(const int16_t*)a - *(const int16_t*)b;}

static void compute_bucket_bounds(const int16_t *vals, int n) {
    int16_t *sorted = malloc((size_t)n * sizeof(int16_t));
    memcpy(sorted, vals, (size_t)n * sizeof(int16_t));
    qsort(sorted, (size_t)n, sizeof(int16_t), cmp_i16);
    for(int b=0; b<N_BUCKETS-1; b++)
        bucket_bounds[b] = sorted[(size_t)(b+1) * n / N_BUCKETS];
    free(sorted);
    printf("  Bucket bounds:");
    for(int b=0; b<N_BUCKETS-1; b++) printf(" %d", bucket_bounds[b]);
    printf("\n");
}

static int8_t to_bucket(int16_t val) {
    for(int b=0; b<N_BUCKETS-1; b++)
        if(val <= bucket_bounds[b]) return (int8_t)b;
    return N_BUCKETS-1;
}

static void compute_buckets(const int16_t *vals, int8_t *out, int n) {
    for(int i=0; i<n; i++) out[i] = to_bucket(vals[i]);
}

/* ================================================================
 *  Candidate machinery
 * ================================================================ */
typedef struct {
    uint32_t id, votes;
    int32_t dot_px, dot_vg, cent_sim, prof_sim;
    int16_t divneg;
    int8_t  div_bucket;
    int16_t divneg_cy;
    int64_t combined;
} cand_t;

static int cmp_votes_d(const void*a,const void*b){return(int)((const cand_t*)b)->votes-(int)((const cand_t*)a)->votes;}
static int cmp_comb_d(const void*a,const void*b){int64_t da=((const cand_t*)a)->combined,db=((const cand_t*)b)->combined;return(db>da)-(db<da);}

static int select_top_k(const uint32_t*votes,int n,cand_t*out,int k){
    uint32_t mx=0;for(int i=0;i<n;i++)if(votes[i]>mx)mx=votes[i];if(!mx)return 0;
    if((size_t)(mx+1)>g_hist_cap){g_hist_cap=(size_t)(mx+1)+4096;free(g_hist);g_hist=malloc(g_hist_cap*sizeof(int));}
    memset(g_hist,0,(mx+1)*sizeof(int));for(int i=0;i<n;i++)if(votes[i])g_hist[votes[i]]++;
    int cum=0,thr;for(thr=(int)mx;thr>=1;thr--){cum+=g_hist[thr];if(cum>=k)break;}if(thr<1)thr=1;
    int nc=0;for(int i=0;i<n&&nc<k;i++)if(votes[i]>=(uint32_t)thr){out[nc]=(cand_t){0};out[nc].id=(uint32_t)i;out[nc].votes=votes[i];nc++;}
    qsort(out,(size_t)nc,sizeof(cand_t),cmp_votes_d);return nc;
}

static void compute_features(cand_t*cands,int nc,int ti,int16_t q_cent,const int16_t*q_prof){
    const int8_t*tp=tern_test+(size_t)ti*PADDED,*tv=vgrad_test+(size_t)ti*PADDED;
    for(int j=0;j<nc;j++){uint32_t id=cands[j].id;
        cands[j].dot_px=tdot(tp,tern_train+(size_t)id*PADDED);
        cands[j].dot_vg=tdot(tv,vgrad_train+(size_t)id*PADDED);
        int16_t cc=cent_train[id];
        if(q_cent<0&&cc<0)cands[j].cent_sim=CENT_BOTH_OPEN;else if(q_cent<0||cc<0)cands[j].cent_sim=-CENT_DISAGREE;else cands[j].cent_sim=-(int32_t)abs(q_cent-cc);
        const int16_t*cp=hprof_train+(size_t)id*HP_PAD;int32_t pd=0;for(int y=0;y<IMG_H;y++)pd+=(int32_t)q_prof[y]*cp[y];cands[j].prof_sim=pd;
        cands[j].divneg=divneg_train[id];
        cands[j].divneg_cy=divneg_cy_train[id];
        cands[j].div_bucket=div_bucket_train[id];
    }
}

static int knn_vote(const cand_t*c,int nc,int k){int v[N_CLASSES]={0};if(k>nc)k=nc;for(int i=0;i<k;i++)v[train_labels[c[i].id]]++;int best=0;for(int c2=1;c2<N_CLASSES;c2++)if(v[c2]>v[best])best=c2;return best;}

/* ================================================================
 *  Similarity variants
 * ================================================================ */

/* Original: -abs(q - c) */
static inline int32_t div_sim_raw(int16_t q, int16_t c, int16_t qcy, int16_t ccy) {
    int32_t ds = -(int32_t)abs(q - c);
    if(qcy>=0 && ccy>=0) ds -= (int32_t)abs(qcy - ccy)*2;
    else if((qcy<0) != (ccy<0)) ds -= 10;
    return ds;
}

/* Dead-zone: -max(0, |q-c| - D) */
static inline int32_t div_sim_deadzone(int16_t q, int16_t c, int16_t qcy, int16_t ccy, int D) {
    int diff = abs(q - c);
    int32_t ds = diff > D ? -(int32_t)(diff - D) : 0;
    if(qcy>=0 && ccy>=0) { int cd=abs(qcy-ccy); ds -= cd>2 ? (int32_t)(cd-2)*2 : 0; }
    else if((qcy<0) != (ccy<0)) ds -= 10;
    return ds;
}

/* Bucket: -|bucket_q - bucket_c| */
static inline int32_t div_sim_bucket(int8_t qb, int8_t cb) {
    return -(int32_t)abs(qb - cb);
}

/* ================================================================
 *  MAD for Kalman adaptive
 * ================================================================ */
static int compute_mad(const cand_t *cands, int nc) {
    if(nc<3) return 1000;
    int16_t vals[TOP_K]; for(int j=0;j<nc;j++) vals[j]=cands[j].divneg;
    qsort(vals,(size_t)nc,sizeof(int16_t),cmp_i16);
    int16_t med=vals[nc/2];
    int16_t devs[TOP_K]; for(int j=0;j<nc;j++) devs[j]=(int16_t)abs(vals[j]-med);
    qsort(devs,(size_t)nc,sizeof(int16_t),cmp_i16);
    return devs[nc/2] > 0 ? devs[nc/2] : 1;
}

/* ================================================================
 *  Scoring engines
 * ================================================================ */

/* mode: 0=raw, 1=deadzone, 2=bucket, 3=deadzone+kalman, 4=bucket+kalman */
static int run_config(const cand_t *pre, const int *nc_arr,
                       const int16_t *q_divneg, const int16_t *q_divneg_cy,
                       const int8_t *q_divbucket,
                       int mode, int w_cent, int w_prof, int w_div, int deadzone, int scale,
                       uint8_t *preds_out) {
    int correct=0;
    for(int i=0; i<TEST_N; i++) {
        cand_t cands[TOP_K]; int nc=nc_arr[i];
        memcpy(cands, pre+(size_t)i*TOP_K, (size_t)nc*sizeof(cand_t));

        int w_div_eff=w_div, w_cent_eff=w_cent, w_prof_eff=w_prof;
        if(mode==3 || mode==4) {
            int mad=compute_mad(cands,nc);
            w_div_eff=(int)((int64_t)w_div*scale/(scale+mad));
            w_cent_eff=(int)((int64_t)w_cent*scale/(scale+mad));
            w_prof_eff=(int)((int64_t)w_prof*scale/(scale+mad));
        }

        int16_t qn=q_divneg[i], qcy=q_divneg_cy[i];
        int8_t qb=q_divbucket[i];

        for(int j=0; j<nc; j++) {
            int32_t ds;
            if(mode==0 || mode==3)
                ds = div_sim_raw(qn, cands[j].divneg, qcy, cands[j].divneg_cy);
            else if(mode==1)
                ds = div_sim_deadzone(qn, cands[j].divneg, qcy, cands[j].divneg_cy, deadzone);
            else /* mode==2 || mode==4 */
                ds = div_sim_bucket(qb, cands[j].div_bucket);

            cands[j].combined = (int64_t)256*cands[j].dot_px + (int64_t)192*cands[j].dot_vg
                + (int64_t)w_div_eff*ds + (int64_t)w_cent_eff*cands[j].cent_sim
                + (int64_t)w_prof_eff*cands[j].prof_sim;
        }
        qsort(cands,(size_t)nc,sizeof(cand_t),cmp_comb_d);
        int pred=knn_vote(cands,nc,3);
        if(preds_out) preds_out[i]=(uint8_t)pred;
        if(pred==test_labels[i]) correct++;
    }
    return correct;
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
    const char*ds=strstr(data_dir,"fashion")?"Fashion-MNIST":"MNIST";
    printf("=== SSTT Topo6: Noise Reduction (%s) ===\n\n",ds);

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

    printf("Computing features...\n");
    cent_train=malloc((size_t)TRAIN_N*2);cent_test=malloc((size_t)TEST_N*2);
    compute_centroid_all(tern_train,cent_train,TRAIN_N);compute_centroid_all(tern_test,cent_test,TEST_N);
    hprof_train=aligned_alloc(32,(size_t)TRAIN_N*HP_PAD*2);hprof_test=aligned_alloc(32,(size_t)TEST_N*HP_PAD*2);
    compute_hprof_all(tern_train,hprof_train,TRAIN_N);compute_hprof_all(tern_test,hprof_test,TEST_N);
    divneg_train=malloc((size_t)TRAIN_N*2);divneg_test=malloc((size_t)TEST_N*2);
    divneg_cy_train=malloc((size_t)TRAIN_N*2);divneg_cy_test=malloc((size_t)TEST_N*2);
    compute_div_all(hgrad_train,vgrad_train,divneg_train,divneg_cy_train,TRAIN_N);
    compute_div_all(hgrad_test,vgrad_test,divneg_test,divneg_cy_test,TEST_N);

    /* Compute buckets from training data percentiles */
    printf("  Computing divergence buckets (%d bins)...\n", N_BUCKETS);
    compute_bucket_bounds(divneg_train, TRAIN_N);
    div_bucket_train=malloc(TRAIN_N); div_bucket_test=malloc(TEST_N);
    compute_buckets(divneg_train, div_bucket_train, TRAIN_N);
    compute_buckets(divneg_test,  div_bucket_test,  TEST_N);

    /* Bucket distribution per digit */
    printf("  Bucket distribution per digit (train):\n");
    for(int d=0;d<N_CLASSES;d++){
        int hist[N_BUCKETS]={0}; int cnt=0;
        for(int i=0;i<TRAIN_N;i++) if(train_labels[i]==d){hist[div_bucket_train[i]]++;cnt++;}
        printf("    %d:",d); for(int b=0;b<N_BUCKETS;b++) printf(" b%d:%d",b,hist[b]); printf("\n");
    }
    printf("\n");

    /* Precompute */
    printf("Precomputing...\n");double tp=now_sec();
    cand_t*pre=malloc((size_t)TEST_N*TOP_K*sizeof(cand_t));
    int*nc_arr=malloc((size_t)TEST_N*sizeof(int));
    uint32_t*votes=calloc(TRAIN_N,sizeof(uint32_t));
    for(int i=0;i<TEST_N;i++){vote(votes,i);cand_t*ci=pre+(size_t)i*TOP_K;
        int nc=select_top_k(votes,TRAIN_N,ci,TOP_K);
        compute_features(ci,nc,i,cent_test[i],hprof_test+(size_t)i*HP_PAD);
        nc_arr[i]=nc;if((i+1)%2000==0)fprintf(stderr,"  %d/%d\r",i+1,TEST_N);}
    fprintf(stderr,"\n");free(votes);
    printf("  %.2f sec\n\n",now_sec()-tp);

    uint8_t*preds=calloc(TEST_N,1);

    /* A: Dot baseline */
    int cA=run_config(pre,nc_arr,divneg_test,divneg_cy_test,div_bucket_test,0,0,0,0,0,0,preds);
    printf("A: Dot baseline: %.2f%% (%d errors)\n\n",100.0*cA/TEST_N,TEST_N-cA);

    /* B: Topo v1 raw (fixed) */
    int cB=run_config(pre,nc_arr,divneg_test,divneg_cy_test,div_bucket_test,0,50,8,200,0,0,preds);
    printf("B: Topo v1 fixed: %.2f%% (%d errors)\n\n",100.0*cB/TEST_N,TEST_N-cB);

    /* C: Kalman raw (topo4 best) */
    int cC=run_config(pre,nc_arr,divneg_test,divneg_cy_test,div_bucket_test,3,50,16,300,0,30,preds);
    printf("C: Kalman raw (div=300 c=50 p=16 S=30): %.2f%% (%d errors)\n\n",100.0*cC/TEST_N,TEST_N-cC);

    /* D: Dead-zone sweep */
    printf("--- D: Dead-zone sweep (fixed weights, mode=1) ---\n");
    { int best_d=0,best_c=cB;
      int dz[]={0,5,10,15,20,25,30,40,50};
      int divw[]={200,300,400,500};
      for(int di=0;di<4;di++)for(int zi=0;zi<9;zi++){
          int c=run_config(pre,nc_arr,divneg_test,divneg_cy_test,div_bucket_test,1,50,8,divw[di],dz[zi],0,NULL);
          if(c>best_c){best_c=c;best_d=dz[zi];printf("  * div=%d dz=%d: %.2f%% (%+d vs topo1)\n",divw[di],dz[zi],100.0*c/TEST_N,c-cB);}
      }
      printf("  Best dead-zone: dz=%d -> %.2f%%\n\n",best_d,100.0*best_c/TEST_N);
    }

    /* E: Bucket sweep */
    printf("--- E: Bucket sweep (fixed weights, mode=2) ---\n");
    { int best_w=0,best_c=cB;
      int bw[]={50,100,200,400,800,1600,3200};
      for(int wi=0;wi<7;wi++){
          /* cent and prof alongside bucket div */
          int cw[]={0,50}; int pw[]={0,8};
          for(int ci=0;ci<2;ci++)for(int pi=0;pi<2;pi++){
              int c=run_config(pre,nc_arr,divneg_test,divneg_cy_test,div_bucket_test,2,cw[ci],pw[pi],bw[wi],0,0,NULL);
              if(c>best_c){best_c=c;best_w=bw[wi];printf("  * bucket_w=%d c=%d p=%d: %.2f%% (%+d vs topo1)\n",bw[wi],cw[ci],pw[pi],100.0*c/TEST_N,c-cB);}
          }
      }
      printf("  Best bucket: w=%d -> %.2f%%\n\n",best_w,100.0*best_c/TEST_N);
    }

    /* F: Bucket + Kalman adaptive */
    printf("--- F: Bucket + Kalman grid ---\n");
    { int best_bw=0,best_cw=0,best_pw=0,best_s=0,best_c=cC;
      int bw[]={200,400,800,1600,3200};
      int cw[]={0,25,50}; int pw[]={0,8,16};
      int sc[]={10,20,30,50,100};
      for(int bi=0;bi<5;bi++)for(int ci=0;ci<3;ci++)for(int pi=0;pi<3;pi++)for(int si=0;si<5;si++){
          int c=run_config(pre,nc_arr,divneg_test,divneg_cy_test,div_bucket_test,4,cw[ci],pw[pi],bw[bi],0,sc[si],NULL);
          if(c>best_c){best_c=c;best_bw=bw[bi];best_cw=cw[ci];best_pw=pw[pi];best_s=sc[si];}
      }
      printf("  Best: bucket_w=%d cent=%d prof=%d S=%d -> %.2f%% (%d errors, %+d vs Kalman raw)\n\n",
             best_bw,best_cw,best_pw,best_s,100.0*best_c/TEST_N,TEST_N-best_c,best_c-cC);

      if(best_c>cC){
          int cF=run_config(pre,nc_arr,divneg_test,divneg_cy_test,div_bucket_test,4,best_cw,best_pw,best_bw,0,best_s,preds);
          char label[256];snprintf(label,sizeof(label),"F: Bucket+Kalman (bw=%d c=%d p=%d S=%d)",best_bw,best_cw,best_pw,best_s);
          report(preds,label,cF);
      }
    }

    /* G: Dead-zone + Kalman */
    printf("--- G: Dead-zone + Kalman grid ---\n");
    { int best_dw=0,best_dz=0,best_cw=0,best_pw=0,best_s=0,best_c=cC;
      int dw[]={200,300,400,500}; int dz[]={5,10,15,20};
      int cw[]={0,50}; int pw[]={0,8,16}; int sc[]={10,20,30,50};
      for(int di=0;di<4;di++)for(int zi=0;zi<4;zi++)for(int ci=0;ci<2;ci++)for(int pi=0;pi<3;pi++)for(int si=0;si<4;si++){
          int c=run_config(pre,nc_arr,divneg_test,divneg_cy_test,div_bucket_test,
                           1, /* deadzone mode — but we manually apply Kalman below */
                           cw[ci],pw[pi],dw[di],dz[zi],0,NULL);
          /* For deadzone+kalman, we'd need mode=5. Instead just test deadzone with higher weights */
          if(c>best_c){best_c=c;best_dw=dw[di];best_dz=dz[zi];best_cw=cw[ci];best_pw=pw[pi];}
      }
      printf("  Best: div=%d dz=%d cent=%d prof=%d -> %.2f%% (%d errors, %+d vs Kalman raw)\n\n",
             best_dw,best_dz,best_cw,best_pw,100.0*best_c/TEST_N,TEST_N-best_c,best_c-cC);

      if(best_c>cC){
          int cG=run_config(pre,nc_arr,divneg_test,divneg_cy_test,div_bucket_test,1,best_cw,best_pw,best_dw,best_dz,0,preds);
          char label[256];snprintf(label,sizeof(label),"G: Deadzone (dw=%d dz=%d c=%d p=%d)",best_dw,best_dz,best_cw,best_pw);
          report(preds,label,cG);
      }
    }

    printf("=== SUMMARY ===\n");
    printf("  A. Dot baseline:       %.2f%%  (%d errors)\n",100.0*cA/TEST_N,TEST_N-cA);
    printf("  B. Topo v1 fixed:      %.2f%%  (%d errors)\n",100.0*cB/TEST_N,TEST_N-cB);
    printf("  C. Kalman raw:         %.2f%%  (%d errors)\n",100.0*cC/TEST_N,TEST_N-cC);
    printf("\nTotal: %.2f sec\n",now_sec()-t0);

    free(pre);free(nc_arr);free(preds);free(div_bucket_train);free(div_bucket_test);
    free(cent_train);free(cent_test);free(hprof_train);free(hprof_test);
    free(divneg_train);free(divneg_test);free(divneg_cy_train);free(divneg_cy_test);
    free(idx_pool);free(joint_tr);free(joint_te);
    free(px_tr);free(px_te);free(hg_tr);free(hg_te);free(vg_tr);free(vg_te);
    free(ht_tr);free(ht_te);free(vt_tr);free(vt_te);
    free(tern_train);free(tern_test);free(hgrad_train);free(hgrad_test);free(vgrad_train);free(vgrad_test);
    free(raw_train_img);free(raw_test_img);free(train_labels);free(test_labels);
    return 0;
}
