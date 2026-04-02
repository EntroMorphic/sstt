/*
 * sstt_hybrid_diagnose.c — Why does L2 re-ranking hurt?
 *
 * For each test image where hybrid(K=500→rerank200) gets it WRONG
 * but plain K=500 gets it RIGHT, dump:
 *   - The true label and both predictions
 *   - How many correct-class candidates survive the SAD filter
 *   - Where correct-class candidates rank in SAD vs vote order
 *   - The SAD distribution of correct vs wrong-class candidates
 *
 * This is a diagnostic, not a classifier.
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
#define TOP_K 500
#define HP_PAD 32
#define FG_W 30
#define FG_H 30
#define FG_SZ (FG_W*FG_H)
#define CENT_BOTH_OPEN 50
#define CENT_DISAGREE 80
#define MAX_REGIONS 16
#define RAW_PAD 800

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
static int16_t *gdiv_train,*gdiv_test;
static uint8_t *raw_pad_train,*raw_pad_test;

/* === All boilerplate from topo9_val (compressed) === */
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
static void build_index(void){long vc[BYTE_VALS]={0};for(int i=0;i<TRAIN_N;i++){const uint8_t*s=joint_tr+(size_t)i*SIG_PAD;for(int k=0;k<N_BLOCKS;k++)vc[s[k]]++;}bg=0;long mc=0;for(int v=0;v<BYTE_VALS;v++)if(vc[v]>mc){mc=vc[v];bg=(uint8_t)v;}compute_ig(joint_tr,BYTE_VALS,bg,ig_w);for(int v=0;v<BYTE_VALS;v++)for(int b=0;b<8;b++)nbr[v][b]=(uint8_t)(v^(1<<b));memset(idx_sz,0,sizeof(idx_sz));for(int i=0;i<TRAIN_N;i++){const uint8_t*s=joint_tr+(size_t)i*SIG_PAD;for(int k=0;k<N_BLOCKS;k++)if(s[k]!=bg)idx_sz[k][s[k]]++;}uint32_t tot=0;for(int k=0;k<N_BLOCKS;k++)for(int v=0;v<BYTE_VALS;v++){idx_off[k][v]=tot;tot+=idx_sz[k][v];}idx_pool=malloc((size_t)tot*sizeof(uint32_t));uint32_t(*wp)[BYTE_VALS]=malloc((size_t)N_BLOCKS*BYTE_VALS*4);memcpy(wp,idx_off,(size_t)N_BLOCKS*BYTE_VALS*4);for(int i=0;i<TRAIN_N;i++){const uint8_t*s=joint_tr+(size_t)i*SIG_PAD;for(int k=0;k<N_BLOCKS;k++)if(s[k]!=bg)idx_pool[wp[k][s[k]]++]=(uint32_t)i;}free(wp);}
static int16_t enclosed_centroid(const int8_t*tern){uint8_t grid[FG_SZ];memset(grid,0,sizeof(grid));for(int y=0;y<IMG_H;y++)for(int x=0;x<IMG_W;x++)grid[(y+1)*FG_W+(x+1)]=(tern[y*IMG_W+x]>0)?1:0;uint8_t visited[FG_SZ];memset(visited,0,sizeof(visited));int stack[FG_SZ];int sp=0;for(int y=0;y<FG_H;y++)for(int x=0;x<FG_W;x++){if(y==0||y==FG_H-1||x==0||x==FG_W-1){int pos=y*FG_W+x;if(!grid[pos]&&!visited[pos]){visited[pos]=1;stack[sp++]=pos;}}}while(sp>0){int pos=stack[--sp];int py=pos/FG_W,px2=pos%FG_W;const int dx[]={0,0,1,-1},dy[]={1,-1,0,0};for(int d=0;d<4;d++){int ny=py+dy[d],nx=px2+dx[d];if(ny<0||ny>=FG_H||nx<0||nx>=FG_W)continue;int npos=ny*FG_W+nx;if(!visited[npos]&&!grid[npos]){visited[npos]=1;stack[sp++]=npos;}}}int sum_y=0,count=0;for(int y=1;y<FG_H-1;y++)for(int x=1;x<FG_W-1;x++){int pos=y*FG_W+x;if(!grid[pos]&&!visited[pos]){sum_y+=(y-1);count++;}}return count>0?(int16_t)(sum_y/count):-1;}
static void hprofile(const int8_t*tern,int16_t*prof){for(int y=0;y<IMG_H;y++){int c=0;for(int x=0;x<IMG_W;x++)c+=(tern[y*IMG_W+x]>0);prof[y]=(int16_t)c;}for(int y=IMG_H;y<HP_PAD;y++)prof[y]=0;}
static void div_features(const int8_t*hg,const int8_t*vg,int16_t*ns,int16_t*cy){int neg_sum=0,neg_ysum=0,neg_cnt=0;for(int y=0;y<IMG_H;y++)for(int x=0;x<IMG_W;x++){int dh=(int)hg[y*IMG_W+x]-(x>0?(int)hg[y*IMG_W+x-1]:0);int dv=(int)vg[y*IMG_W+x]-(y>0?(int)vg[(y-1)*IMG_W+x]:0);int d=dh+dv;if(d<0){neg_sum+=d;neg_ysum+=y;neg_cnt++;}}*ns=(int16_t)(neg_sum<-32767?-32767:neg_sum);*cy=neg_cnt>0?(int16_t)(neg_ysum/neg_cnt):-1;}
static void grid_div(const int8_t*hg,const int8_t*vg,int grow,int gcol,int16_t*out){int nr=grow*gcol;int regions[MAX_REGIONS];memset(regions,0,sizeof(int)*nr);for(int y=0;y<IMG_H;y++)for(int x=0;x<IMG_W;x++){int dh=(int)hg[y*IMG_W+x]-(x>0?(int)hg[y*IMG_W+x-1]:0);int dv=(int)vg[y*IMG_W+x]-(y>0?(int)vg[(y-1)*IMG_W+x]:0);int d=dh+dv;if(d<0){int ry=y*grow/IMG_H;int rx=x*gcol/IMG_W;if(ry>=grow)ry=grow-1;if(rx>=gcol)rx=gcol-1;regions[ry*gcol+rx]+=d;}}for(int i=0;i<nr;i++)out[i]=(int16_t)(regions[i]<-32767?-32767:regions[i]);}
static inline int32_t tdot(const int8_t*a,const int8_t*b){__m256i acc=_mm256_setzero_si256();for(int i=0;i<PADDED;i+=32)acc=_mm256_add_epi8(acc,_mm256_sign_epi8(_mm256_load_si256((const __m256i*)(a+i)),_mm256_load_si256((const __m256i*)(b+i))));__m256i lo=_mm256_cvtepi8_epi16(_mm256_castsi256_si128(acc)),hi=_mm256_cvtepi8_epi16(_mm256_extracti128_si256(acc,1));__m256i s32=_mm256_madd_epi16(_mm256_add_epi16(lo,hi),_mm256_set1_epi16(1));__m128i s=_mm_add_epi32(_mm256_castsi256_si128(s32),_mm256_extracti128_si256(s32,1));s=_mm_hadd_epi32(s,s);s=_mm_hadd_epi32(s,s);return _mm_cvtsi128_si32(s);}
static void vote(uint32_t*votes,int img){memset(votes,0,TRAIN_N*sizeof(uint32_t));const uint8_t*sig=joint_te+(size_t)img*SIG_PAD;for(int k=0;k<N_BLOCKS;k++){uint8_t bv=sig[k];if(bv==bg)continue;uint16_t w=ig_w[k],wh=w>1?w/2:1;{uint32_t off=idx_off[k][bv];uint16_t sz=idx_sz[k][bv];const uint32_t*ids=idx_pool+off;for(uint16_t j=0;j<sz;j++)votes[ids[j]]+=w;}for(int nb=0;nb<8;nb++){uint8_t nv=nbr[bv][nb];if(nv==bg)continue;uint32_t noff=idx_off[k][nv];uint16_t nsz=idx_sz[k][nv];const uint32_t*nids=idx_pool+noff;for(uint16_t j=0;j<nsz;j++)votes[nids[j]]+=wh;}}}

typedef struct {
    uint32_t id, votes;
    int32_t dot_px, dot_vg, div_sim, cent_sim, prof_sim, gdiv_sim;
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
static int compute_mad(const cand_t*cands,int nc){if(nc<3)return 1000;int16_t*vals=malloc(nc*2);for(int j=0;j<nc;j++)vals[j]=divneg_train[cands[j].id];qsort(vals,(size_t)nc,sizeof(int16_t),cmp_i16);int16_t med=vals[nc/2];for(int j=0;j<nc;j++)vals[j]=(int16_t)abs(vals[j]-med);qsort(vals,(size_t)nc,sizeof(int16_t),cmp_i16);int r=vals[nc/2]>0?vals[nc/2]:1;free(vals);return r;}

static inline uint32_t sad_avx2(const uint8_t*a,const uint8_t*b){
    __m256i sum=_mm256_setzero_si256();
    for(int i=0;i<RAW_PAD;i+=32)
        sum=_mm256_add_epi64(sum,_mm256_sad_epu8(
            _mm256_load_si256((const __m256i*)(a+i)),
            _mm256_load_si256((const __m256i*)(b+i))));
    __m128i lo=_mm256_castsi256_si128(sum),hi=_mm256_extracti128_si256(sum,1);
    __m128i s=_mm_add_epi64(lo,hi);
    return(uint32_t)(_mm_extract_epi64(s,0)+_mm_extract_epi64(s,1));
}

static void pad_raw(void){
    raw_pad_train=aligned_alloc(32,(size_t)TRAIN_N*RAW_PAD);
    raw_pad_test=aligned_alloc(32,(size_t)TEST_N*RAW_PAD);
    for(int i=0;i<TRAIN_N;i++){memcpy(raw_pad_train+(size_t)i*RAW_PAD,raw_train_img+(size_t)i*PIXELS,PIXELS);memset(raw_pad_train+(size_t)i*RAW_PAD+PIXELS,0,RAW_PAD-PIXELS);}
    for(int i=0;i<TEST_N;i++){memcpy(raw_pad_test+(size_t)i*RAW_PAD,raw_test_img+(size_t)i*PIXELS,PIXELS);memset(raw_pad_test+(size_t)i*RAW_PAD+PIXELS,0,RAW_PAD-PIXELS);}
}

static int rank_and_predict(cand_t*cands,int nc,int ti,
                            int w_c,int w_p,int w_d,int w_g,int sc,int nreg){
    const int16_t*qi=gdiv_test+(size_t)ti*MAX_REGIONS;
    for(int j=0;j<nc;j++){
        const int16_t*ci=gdiv_train+(size_t)cands[j].id*MAX_REGIONS;
        int32_t l=0;for(int r=0;r<nreg;r++)l+=abs(qi[r]-ci[r]);
        cands[j].gdiv_sim=-l;
    }
    int mad=compute_mad(cands,nc);int64_t s=(int64_t)sc;
    int wc=(int)(s*w_c/(s+mad)),wp=(int)(s*w_p/(s+mad)),wd=(int)(s*w_d/(s+mad)),wg=(int)(s*w_g/(s+mad));
    for(int j=0;j<nc;j++)
        cands[j].combined=(int64_t)256*cands[j].dot_px+(int64_t)192*cands[j].dot_vg
            +(int64_t)wc*cands[j].cent_sim+(int64_t)wp*cands[j].prof_sim
            +(int64_t)wd*cands[j].div_sim+(int64_t)wg*cands[j].gdiv_sim;
    qsort(cands,(size_t)nc,sizeof(cand_t),cmp_comb_d);
    return knn_vote(cands,nc,3);
}

int main(int argc,char**argv){
    if(argc>1){data_dir=argv[1];size_t l=strlen(data_dir);if(l&&data_dir[l-1]!='/'){char*buf=malloc(l+2);memcpy(buf,data_dir,l);buf[l]='/';buf[l+1]='\0';data_dir=buf;}}

    printf("=== SSTT Hybrid Diagnostic: Why Does L2 Re-Ranking Hurt? ===\n\n");

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
    build_index();
    cent_train=malloc((size_t)TRAIN_N*2);cent_test=malloc((size_t)TEST_N*2);
    for(int i=0;i<TRAIN_N;i++)cent_train[i]=enclosed_centroid(tern_train+(size_t)i*PADDED);
    for(int i=0;i<TEST_N;i++)cent_test[i]=enclosed_centroid(tern_test+(size_t)i*PADDED);
    hprof_train=aligned_alloc(32,(size_t)TRAIN_N*HP_PAD*2);hprof_test=aligned_alloc(32,(size_t)TEST_N*HP_PAD*2);
    for(int i=0;i<TRAIN_N;i++)hprofile(tern_train+(size_t)i*PADDED,hprof_train+(size_t)i*HP_PAD);
    for(int i=0;i<TEST_N;i++)hprofile(tern_test+(size_t)i*PADDED,hprof_test+(size_t)i*HP_PAD);
    divneg_train=malloc((size_t)TRAIN_N*2);divneg_test=malloc((size_t)TEST_N*2);
    divneg_cy_train=malloc((size_t)TRAIN_N*2);divneg_cy_test=malloc((size_t)TEST_N*2);
    for(int i=0;i<TRAIN_N;i++)div_features(hgrad_train+(size_t)i*PADDED,vgrad_train+(size_t)i*PADDED,&divneg_train[i],&divneg_cy_train[i]);
    for(int i=0;i<TEST_N;i++)div_features(hgrad_test+(size_t)i*PADDED,vgrad_test+(size_t)i*PADDED,&divneg_test[i],&divneg_cy_test[i]);
    gdiv_train=malloc((size_t)TRAIN_N*MAX_REGIONS*2);gdiv_test=malloc((size_t)TEST_N*MAX_REGIONS*2);
    for(int i=0;i<TRAIN_N;i++)grid_div(hgrad_train+(size_t)i*PADDED,vgrad_train+(size_t)i*PADDED,2,4,gdiv_train+(size_t)i*MAX_REGIONS);
    for(int i=0;i<TEST_N;i++)grid_div(hgrad_test+(size_t)i*PADDED,vgrad_test+(size_t)i*PADDED,2,4,gdiv_test+(size_t)i*MAX_REGIONS);
    pad_raw();

    int w_c=50,w_p=16,w_d=200,w_g=100,sc=50,nreg=8;
    uint32_t*vbuf=calloc(TRAIN_N,sizeof(uint32_t));

    int n_disagree=0, n_hybrid_lost=0, n_hybrid_won=0;
    int total_correct_removed=0, total_wrong_removed=0;

    printf("Scanning 10K test images for hybrid vs K=500 disagreements...\n\n");

    for(int i=0;i<TEST_N;i++){
        uint8_t true_label=test_labels[i];
        vote(vbuf,i);

        /* Get K=500 candidates */
        cand_t cands500[TOP_K];
        int nc=select_top_k(vbuf,TRAIN_N,cands500,TOP_K);
        compute_base(cands500,nc,i,cent_test[i],hprof_test+(size_t)i*HP_PAD,divneg_test[i],divneg_cy_test[i]);

        /* Run K=500 ranking */
        cand_t c500[TOP_K];
        memcpy(c500,cands500,(size_t)nc*sizeof(cand_t));
        int pred500=rank_and_predict(c500,nc,i,w_c,w_p,w_d,w_g,sc,nreg);

        /* Compute SAD for all 500 and sort */
        typedef struct{uint32_t sad;int idx;}se_t;
        se_t se[TOP_K];
        const uint8_t*qraw=raw_pad_test+(size_t)i*RAW_PAD;
        for(int j=0;j<nc;j++){
            se[j].sad=sad_avx2(qraw,raw_pad_train+(size_t)cands500[j].id*RAW_PAD);
            se[j].idx=j;
        }
        /* Sort by SAD ascending */
        for(int a=0;a<nc-1;a++)for(int b=a+1;b<nc;b++)if(se[b].sad<se[a].sad){se_t tmp=se[a];se[a]=se[b];se[b]=tmp;}

        /* Take top 200 by SAD */
        int kr=200;if(kr>nc)kr=nc;
        cand_t c200[TOP_K];
        for(int j=0;j<kr;j++)c200[j]=cands500[se[j].idx];

        int pred_hybrid=rank_and_predict(c200,kr,i,w_c,w_p,w_d,w_g,sc,nreg);

        if(pred500!=pred_hybrid){
            n_disagree++;
            int k500_right=(pred500==true_label);
            int hybrid_right=(pred_hybrid==true_label);

            if(k500_right&&!hybrid_right) n_hybrid_lost++;
            if(!k500_right&&hybrid_right) n_hybrid_won++;

            if(k500_right&&!hybrid_right){
                /* This is the interesting case: hybrid lost. What did the SAD filter remove? */
                int correct_in_500=0,correct_in_200=0;
                int correct_removed=0,wrong_removed=0;
                uint32_t min_sad_correct=UINT32_MAX,max_sad_correct=0;
                uint32_t min_sad_wrong=UINT32_MAX,max_sad_wrong=0;

                /* Count correct-class candidates in full 500 */
                for(int j=0;j<nc;j++)
                    if(train_labels[cands500[j].id]==true_label)correct_in_500++;

                /* Count correct-class in top-200 by SAD */
                int kept[TOP_K]={0};
                for(int j=0;j<kr;j++)kept[se[j].idx]=1;
                for(int j=0;j<nc;j++){
                    uint8_t lbl=train_labels[cands500[j].id];
                    if(!kept[j]){
                        if(lbl==true_label)correct_removed++;
                        else wrong_removed++;
                    }
                    if(lbl==true_label){
                        if(se[j].sad<min_sad_correct)min_sad_correct=se[j].sad; /* Not right — need original SAD */
                    }
                }
                /* Actually compute SAD stats properly */
                min_sad_correct=UINT32_MAX;max_sad_correct=0;
                min_sad_wrong=UINT32_MAX;max_sad_wrong=0;
                for(int j=0;j<nc;j++){
                    uint32_t sd=sad_avx2(qraw,raw_pad_train+(size_t)cands500[j].id*RAW_PAD);
                    if(train_labels[cands500[j].id]==true_label){
                        if(sd<min_sad_correct)min_sad_correct=sd;
                        if(sd>max_sad_correct)max_sad_correct=sd;
                    } else {
                        if(sd<min_sad_wrong)min_sad_wrong=sd;
                        if(sd>max_sad_wrong)max_sad_wrong=sd;
                    }
                }
                for(int j=0;j<kr;j++)
                    if(train_labels[cands500[se[j].idx].id]==true_label)correct_in_200++;

                total_correct_removed+=correct_removed;
                total_wrong_removed+=wrong_removed;

                printf("  Image %5d: true=%d  K500→%d(right)  hybrid→%d(wrong)\n",
                       i,true_label,pred500,pred_hybrid);
                printf("    Correct-class candidates: %d in K=500, %d survived SAD filter, %d removed\n",
                       correct_in_500,correct_in_200,correct_removed);
                printf("    Wrong-class removed: %d\n",wrong_removed);
                printf("    SAD range (correct class): %u - %u\n",min_sad_correct,max_sad_correct);
                printf("    SAD range (wrong class):   %u - %u\n",min_sad_wrong,max_sad_wrong);
                /* Show vote rank of removed correct-class candidates */
                printf("    Removed correct-class vote ranks:");
                for(int j=0;j<nc;j++)
                    if(!kept[j]&&train_labels[cands500[j].id]==true_label)
                        printf(" %d",j);
                printf("\n\n");
            }
        }
    }

    printf("=== SUMMARY ===\n");
    printf("Total disagreements: %d\n",n_disagree);
    printf("Hybrid lost (K500 right, hybrid wrong): %d\n",n_hybrid_lost);
    printf("Hybrid won  (K500 wrong, hybrid right): %d\n",n_hybrid_won);
    printf("Net: hybrid is %+d images vs K=500\n",n_hybrid_won-n_hybrid_lost);
    printf("Total correct-class candidates removed by SAD filter: %d\n",total_correct_removed);
    printf("Total wrong-class candidates removed by SAD filter: %d\n",total_wrong_removed);

    return 0;
}
