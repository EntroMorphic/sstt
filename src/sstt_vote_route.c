/*
 * sstt_vote_route.c — Free Routing: Predict Confidence from Vote Phase
 *
 * Tests whether the class concentration of the top-200 candidates
 * (available for free after the vote phase, before any dot products)
 * can predict classification success and route images:
 *
 *   High concentration → skip ranking, output plurality class directly
 *   Low concentration  → run full ranking pipeline
 *
 * This eliminates 87% of compute (the ranking step) for the easy majority.
 *
 * Build: make sstt_vote_route
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
#define FG_W 30
#define FG_H 30
#define FG_SZ (FG_W*FG_H)
#define CENT_BOTH_OPEN 50
#define CENT_DISAGREE 80
#define MAX_REGIONS 16

static const char *data_dir="data/";
static double now_sec(void){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return ts.tv_sec+ts.tv_nsec*1e-9;}

static uint8_t *raw_train_img,*raw_test_img,*train_labels,*test_labels;
static int8_t *tern_train,*tern_test,*hgrad_train,*hgrad_test,*vgrad_train,*vgrad_test;
static uint8_t *px_tr,*px_te,*hg_tr,*hg_te,*vg_tr,*vg_te,*ht_tr,*ht_te,*vt_tr,*vt_te,*joint_tr,*joint_te;
static uint16_t ig_w[N_BLOCKS];static uint8_t nbr[BYTE_VALS][8];
static uint32_t idx_off[N_BLOCKS][BYTE_VALS];static uint16_t idx_sz[N_BLOCKS][BYTE_VALS];
static uint32_t *idx_pool;static uint8_t bg;static int *g_hist=NULL;static size_t g_hist_cap=0;
static int16_t *cent_train,*cent_test;
static int16_t *divneg_train,*divneg_test,*divneg_cy_train,*divneg_cy_test;
static int16_t *gdiv_train,*gdiv_test;

/* All boilerplate */
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
static void vote(uint32_t*votes,int img){memset(votes,0,TRAIN_N*sizeof(uint32_t));const uint8_t*sig=joint_te+(size_t)img*SIG_PAD;for(int k=0;k<N_BLOCKS;k++){uint8_t bv=sig[k];if(bv==bg)continue;uint16_t w=ig_w[k],wh=w>1?w/2:1;{uint32_t off=idx_off[k][bv];uint16_t sz=idx_sz[k][bv];const uint32_t*ids=idx_pool+off;for(uint16_t j=0;j<sz;j++)votes[ids[j]]+=w;}for(int nb=0;nb<8;nb++){uint8_t nv=nbr[bv][nb];if(nv==bg)continue;uint32_t noff=idx_off[k][nv];uint16_t nsz=idx_sz[k][nv];const uint32_t*nids=idx_pool+noff;for(uint16_t j=0;j<nsz;j++)votes[nids[j]]+=wh;}}}

typedef struct {uint32_t id,votes;} lite_cand_t;
static int cmp_lite_d(const void*a,const void*b){return(int)((const lite_cand_t*)b)->votes-(int)((const lite_cand_t*)a)->votes;}

static int select_lite(const uint32_t*votes,int n,lite_cand_t*out,int k){
    uint32_t mx=0;for(int i=0;i<n;i++)if(votes[i]>mx)mx=votes[i];if(!mx)return 0;
    if((size_t)(mx+1)>g_hist_cap){g_hist_cap=(size_t)(mx+1)+4096;free(g_hist);g_hist=malloc(g_hist_cap*sizeof(int));}
    memset(g_hist,0,(mx+1)*sizeof(int));for(int i=0;i<n;i++)if(votes[i])g_hist[votes[i]]++;
    int cum=0,thr;for(thr=(int)mx;thr>=1;thr--){cum+=g_hist[thr];if(cum>=k)break;}if(thr<1)thr=1;
    int nc=0;for(int i=0;i<n&&nc<k;i++)if(votes[i]>=(uint32_t)thr){out[nc].id=(uint32_t)i;out[nc].votes=votes[i];nc++;}
    qsort(out,(size_t)nc,sizeof(lite_cand_t),cmp_lite_d);return nc;
}

/* ================================================================
 *  Main
 * ================================================================ */
int main(int argc,char**argv){
    double t0=now_sec();
    if(argc>1){data_dir=argv[1];size_t l=strlen(data_dir);if(l&&data_dir[l-1]!='/'){char*buf=malloc(l+2);memcpy(buf,data_dir,l);buf[l]='/';buf[l+1]='\0';data_dir=buf;}}
    int is_fashion=strstr(data_dir,"fashion")!=NULL;
    const char*ds=is_fashion?"Fashion-MNIST":"MNIST";
    printf("=== SSTT Vote-Phase Routing (%s) ===\n\n",ds);

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
    printf("  Setup: %.2f sec\n\n",now_sec()-t0);

    /* ================================================================
     *  For each test image: vote, get top-200, measure class concentration
     * ================================================================ */
    printf("Computing vote-phase class concentration...\n");
    double t_vote = now_sec();

    uint32_t *votes = calloc(TRAIN_N, sizeof(uint32_t));
    lite_cand_t cands[TOP_K];

    /* Per-image results */
    int *plurality_class = malloc(TEST_N * sizeof(int));
    int *plurality_count = malloc(TEST_N * sizeof(int));
    int *second_count = malloc(TEST_N * sizeof(int));
    double *concentration = malloc(TEST_N * sizeof(double));

    for(int i=0; i<TEST_N; i++){
        vote(votes, i);
        int nc = select_lite(votes, TRAIN_N, cands, TOP_K);

        /* Count class distribution of top-K candidates */
        int cls_count[N_CLASSES] = {0};
        for(int j=0; j<nc; j++) cls_count[train_labels[cands[j].id]]++;

        /* Find plurality and second-place */
        int best=0, best_cnt=0, second_cnt=0;
        for(int c=0; c<N_CLASSES; c++){
            if(cls_count[c] > best_cnt){
                second_cnt = best_cnt;
                best_cnt = cls_count[c];
                best = c;
            } else if(cls_count[c] > second_cnt){
                second_cnt = cls_count[c];
            }
        }
        plurality_class[i] = best;
        plurality_count[i] = best_cnt;
        second_count[i] = second_cnt;
        concentration[i] = nc > 0 ? (double)best_cnt / nc : 0;

        if((i+1)%2000==0) fprintf(stderr,"  %d/%d\r",i+1,TEST_N);
    }
    fprintf(stderr,"\n");
    free(votes);
    printf("  Vote phase: %.2f sec\n\n", now_sec()-t_vote);

    /* ================================================================
     *  Analysis 1: Vote-only accuracy (plurality class = prediction)
     * ================================================================ */
    int vote_correct = 0;
    for(int i=0; i<TEST_N; i++)
        if(plurality_class[i] == test_labels[i]) vote_correct++;
    printf("=== Vote-only accuracy (plurality of top-%d) ===\n", TOP_K);
    printf("  %.2f%% (%d correct, %d errors)\n\n", 100.0*vote_correct/TEST_N,
           vote_correct, TEST_N-vote_correct);

    /* ================================================================
     *  Analysis 2: Concentration as confidence predictor
     * ================================================================ */
    printf("=== Concentration as confidence predictor ===\n\n");
    printf("  %-20s %8s %8s %8s %8s\n", "Concentration", "Count", "Correct", "ErrRate", "Cumul");

    /* Sweep concentration thresholds */
    int thresholds[] = {200, 190, 180, 170, 160, 150, 140, 130, 120, 110, 100, 80, 60};
    int cum_count=0, cum_correct=0;
    for(int ti=0; ti<13; ti++){
        int thr = thresholds[ti]; /* out of 200 */
        int count=0, correct=0;
        for(int i=0; i<TEST_N; i++){
            if(plurality_count[i] >= thr){
                count++;
                if(plurality_class[i] == test_labels[i]) correct++;
            }
        }
        printf("  >= %3d/%d (%5.1f%%) %8d %8d %7.1f%%",
               thr, TOP_K, 100.0*thr/TOP_K, count, correct,
               count > 0 ? 100.0*(count-correct)/count : 0.0);
        printf("   cov=%.1f%%\n", 100.0*count/TEST_N);
    }

    /* ================================================================
     *  Analysis 3: Precision-coverage tradeoff (vote-only routing)
     * ================================================================ */
    printf("\n=== Routing: vote-only fast path + abstain ===\n\n");
    printf("  %-12s %8s %10s %10s %10s\n", "Threshold", "Accept", "FastAcc", "Coverage", "Reject");

    int route_thresholds[] = {195, 190, 185, 180, 175, 170, 160, 150, 140, 120, 100};
    for(int ti=0; ti<11; ti++){
        int thr = route_thresholds[ti];
        int accept=0, accept_correct=0, reject=0, reject_correct=0;
        for(int i=0; i<TEST_N; i++){
            if(plurality_count[i] >= thr){
                accept++;
                if(plurality_class[i] == test_labels[i]) accept_correct++;
            } else {
                reject++;
                if(plurality_class[i] == test_labels[i]) reject_correct++;
            }
        }
        printf("  >= %3d/200 %8d %9.2f%% %9.1f%% %8d (%.1f%% acc)\n",
               thr, accept, accept>0?100.0*accept_correct/accept:0.0,
               100.0*accept/TEST_N, reject,
               reject>0?100.0*reject_correct/reject:0.0);
    }

    /* ================================================================
     *  Analysis 4: Ratio-based routing (plurality / second-place)
     * ================================================================ */
    printf("\n=== Ratio-based routing (plurality / second-place) ===\n\n");
    printf("  %-12s %8s %10s %10s\n", "Ratio >=", "Accept", "FastAcc", "Coverage");

    double ratios[] = {20.0, 10.0, 5.0, 4.0, 3.0, 2.5, 2.0, 1.5, 1.0};
    for(int ri=0; ri<9; ri++){
        double r = ratios[ri];
        int accept=0, accept_correct=0;
        for(int i=0; i<TEST_N; i++){
            double img_ratio = second_count[i] > 0 ? (double)plurality_count[i]/second_count[i] : 999.0;
            if(img_ratio >= r){
                accept++;
                if(plurality_class[i] == test_labels[i]) accept_correct++;
            }
        }
        printf("  >= %5.1f    %8d %9.2f%% %9.1f%%\n",
               r, accept, accept>0?100.0*accept_correct/accept:0.0,
               100.0*accept/TEST_N);
    }

    /* ================================================================
     *  Analysis 5: MI between vote concentration and correctness
     * ================================================================ */
    printf("\n=== Mutual Information: vote concentration vs correctness ===\n\n");
    {
        /* Bucket concentration into 10 bins (0-19, 20-39, ..., 180-200) */
        #define CONC_BINS 10
        int ct[CONC_BINS]={0}, cc[CONC_BINS]={0};
        for(int i=0; i<TEST_N; i++){
            int b = plurality_count[i] / 20; if(b>=CONC_BINS) b=CONC_BINS-1;
            ct[b]++;
            if(plurality_class[i]==test_labels[i]) cc[b]++;
        }
        double p_c = (double)vote_correct/TEST_N, p_e = 1.0-p_c;
        double H_Y=0;
        if(p_c>0)H_Y-=p_c*log2(p_c);if(p_e>0)H_Y-=p_e*log2(p_e);
        double H_Y_Q=0;
        for(int b=0;b<CONC_BINS;b++){
            if(!ct[b])continue;
            double pq=(double)ct[b]/TEST_N;
            double pc2=(double)cc[b]/ct[b],pe2=1.0-pc2;
            double h=0;if(pc2>0)h-=pc2*log2(pc2);if(pe2>0)h-=pe2*log2(pe2);
            H_Y_Q+=pq*h;
        }
        printf("  H(Y) = %.4f bits\n", H_Y);
        printf("  H(Y|concentration) = %.4f bits\n", H_Y_Q);
        printf("  I = %.4f bits (%.1f%% reduction)\n", H_Y-H_Y_Q, 100*(H_Y-H_Y_Q)/H_Y);
    }

    /* ================================================================
     *  Analysis 6: Timing comparison
     * ================================================================ */
    printf("\n=== Timing: vote-only vs full pipeline ===\n\n");
    printf("  Vote phase (all 10K images): %.2f sec\n", now_sec()-t_vote);
    printf("  Per-image vote cost: %.0f us\n", (now_sec()-t_vote)*1e6/TEST_N);
    printf("  Full pipeline (from topo8): ~1 ms/image\n");
    printf("  Speedup for fast path: ~%.0fx\n\n", 1000.0/((now_sec()-t_vote)*1e6/TEST_N));

    printf("Total: %.2f sec\n", now_sec()-t0);
    free(plurality_class);free(plurality_count);free(second_count);free(concentration);
    free(idx_pool);free(joint_tr);free(joint_te);
    free(px_tr);free(px_te);free(hg_tr);free(hg_te);free(vg_tr);free(vg_te);
    free(ht_tr);free(ht_te);free(vt_tr);free(vt_te);
    free(tern_train);free(tern_test);free(hgrad_train);free(hgrad_test);
    free(vgrad_train);free(vgrad_test);
    free(raw_train_img);free(raw_test_img);free(train_labels);free(test_labels);
    return 0;
}
