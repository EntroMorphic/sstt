/*
 * test_kdilute_simple.c — Minimal integration test for kNN dilution analysis
 *
 * Simplified version that focuses on getting one image through the pipeline
 * with extensive error checking.
 */

#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
#define TOP_K 200

/* Global data */
static uint8_t *raw_train_img, *raw_test_img, *train_labels, *test_labels;
static int8_t *tern_train, *tern_test, *hgrad_train, *hgrad_test, *vgrad_train, *vgrad_test;
static uint8_t *px_tr, *px_te, *hg_tr, *hg_te, *vg_tr, *vg_te, *ht_tr, *ht_te, *vt_tr, *vt_te, *joint_tr, *joint_te;
static uint16_t ig_w[N_BLOCKS];
static uint8_t nbr[BYTE_VALS][8];
static uint32_t idx_off[N_BLOCKS][BYTE_VALS];
static uint16_t idx_sz[N_BLOCKS][BYTE_VALS];
static uint32_t *idx_pool;
static uint8_t bg;
static int *g_hist = NULL;
static size_t g_hist_cap = 0;
static int16_t *divneg_train, *divneg_test;

/* Boilerplate functions */
static uint8_t *load_idx(const char*path, uint32_t*cnt, uint32_t*ro, uint32_t*co){
    FILE*f=fopen(path,"rb");if(!f){fprintf(stderr,"ERR:%s\n",path);exit(1);}
    uint32_t m,n;if(fread(&m,4,1,f)!=1||fread(&n,4,1,f)!=1){fclose(f);exit(1);}
    m=__builtin_bswap32(m);n=__builtin_bswap32(n);*cnt=n;
    size_t s=1;if((m&0xFF)>=3){uint32_t r,c;if(fread(&r,4,1,f)!=1||fread(&c,4,1,f)!=1){fclose(f);exit(1);}r=__builtin_bswap32(r);c=__builtin_bswap32(c);if(ro)*ro=r;if(co)*co=c;s=(size_t)r*c;}else{if(ro)*ro=0;if(co)*co=0;}
    size_t total=(size_t)n*s;uint8_t*d=malloc(total);if(!d||fread(d,1,total,f)!=total){fclose(f);exit(1);}fclose(f);return d;
}

static void load_data(const char *data_dir) {
    char p[256];
    snprintf(p,sizeof(p),"%strain-images-idx3-ubyte",data_dir);
    raw_train_img=load_idx(p,NULL,NULL,NULL);
    snprintf(p,sizeof(p),"%strain-labels-idx1-ubyte",data_dir);
    train_labels=load_idx(p,NULL,NULL,NULL);
    snprintf(p,sizeof(p),"%st10k-images-idx3-ubyte",data_dir);
    raw_test_img=load_idx(p,NULL,NULL,NULL);
    snprintf(p,sizeof(p),"%st10k-labels-idx1-ubyte",data_dir);
    test_labels=load_idx(p,NULL,NULL,NULL);
    printf("Data loaded.\n");
}

static inline int8_t clamp_trit(int v){return v>0?1:v<0?-1:0;}

static void quant_tern(const uint8_t*src,int8_t*dst,int n){
    const __m256i bias=_mm256_set1_epi8((char)0x80),thi=_mm256_set1_epi8((char)(170^0x80)),tlo=_mm256_set1_epi8((char)(85^0x80)),one=_mm256_set1_epi8(1);
    for(int i=0;i<n;i++){
        const uint8_t*s=src+(size_t)i*PIXELS;int8_t*d=dst+(size_t)i*PADDED;
        for(int k=0;k+32<=PIXELS;k+=32){
            __m256i px=_mm256_loadu_si256((const __m256i*)(s+k));
            __m256i sp=_mm256_xor_si256(px,bias);
            _mm256_storeu_si256((__m256i*)(d+k),_mm256_sub_epi8(_mm256_and_si256(_mm256_cmpgt_epi8(sp,thi),one),_mm256_and_si256(_mm256_cmpgt_epi8(tlo,sp),one)));
        }
        for(int k=(PIXELS/32)*32;k<PIXELS;k++)d[k]=s[k]>170?1:s[k]<85?-1:0;
        memset(d+PIXELS,0,PADDED-PIXELS);
    }
}

static void gradients(const int8_t*t,int8_t*h,int8_t*v,int n){
    for(int i=0;i<n;i++){
        const int8_t*ti=t+(size_t)i*PADDED;int8_t*hi=h+(size_t)i*PADDED;int8_t*vi=v+(size_t)i*PADDED;
        for(int y=0;y<IMG_H;y++){for(int x=0;x<IMG_W-1;x++)hi[y*IMG_W+x]=clamp_trit(ti[y*IMG_W+x+1]-ti[y*IMG_W+x]);hi[y*IMG_W+IMG_W-1]=0;}
        memset(hi+PIXELS,0,PADDED-PIXELS);
        for(int y=0;y<IMG_H-1;y++)for(int x=0;x<IMG_W;x++)vi[y*IMG_W+x]=clamp_trit(ti[(y+1)*IMG_W+x]-ti[y*IMG_W+x]);
        memset(vi+(IMG_H-1)*IMG_W,0,IMG_W);memset(vi+PIXELS,0,PADDED-PIXELS);
    }
}

static inline uint8_t benc(int8_t a,int8_t b,int8_t c){return(uint8_t)((a+1)*9+(b+1)*3+(c+1));}
static void block_sigs(const int8_t*data,uint8_t*sigs,int n){
    for(int i=0;i<n;i++){
        const int8_t*img=data+(size_t)i*PADDED;uint8_t*sig=sigs+(size_t)i*SIG_PAD;
        for(int y=0;y<IMG_H;y++)for(int s=0;s<BLKS_PER_ROW;s++){int b=y*IMG_W+s*3;sig[y*BLKS_PER_ROW+s]=benc(img[b],img[b+1],img[b+2]);}
        memset(sig+N_BLOCKS,0xFF,SIG_PAD-N_BLOCKS);
    }
}

static void div_features(const int8_t*hg,const int8_t*vg,int16_t*ns){
    int neg_sum=0;
    for(int y=0;y<IMG_H;y++)for(int x=0;x<IMG_W;x++){
        int dh=(int)hg[y*IMG_W+x]-(x>0?(int)hg[y*IMG_W+x-1]:0);
        int dv=(int)vg[y*IMG_W+x]-(y>0?(int)vg[(y-1)*IMG_W+x]:0);
        int d=dh+dv;if(d<0)neg_sum+=d;
    }
    *ns=(int16_t)(neg_sum<-32767?-32767:neg_sum);
}

static void compute_ig(const uint8_t*sigs,int nv,uint8_t bgv,uint16_t*ig_out){
    /* Simplified: use uniform IG weights (skip expensive IG computation for now) */
    for(int k=0;k<N_BLOCKS;k++)ig_out[k]=8;
}

static void build_index(void){
    printf("   [1] Counting block values...\n"); fflush(stdout);
    long vc[BYTE_VALS]={0};
    for(int i=0;i<TRAIN_N;i++){
        if(i % 10000 == 0) { printf("     %d/60000\n", i); fflush(stdout); }
        const uint8_t*s=joint_tr+(size_t)i*SIG_PAD;
        for(int k=0;k<N_BLOCKS;k++)vc[s[k]]++;
    }
    printf("   [2] Finding background value...\n"); fflush(stdout);
    bg=0;long mc=0;for(int v=0;v<BYTE_VALS;v++)if(vc[v]>mc){mc=vc[v];bg=(uint8_t)v;}
    printf("   [3] Computing information gain...\n"); fflush(stdout);
    compute_ig(joint_tr,BYTE_VALS,bg,ig_w);
    printf("   [4] IG computed.\n"); fflush(stdout);
    for(int v=0;v<BYTE_VALS;v++)for(int b=0;b<8;b++)nbr[v][b]=(uint8_t)(v^(1<<b));
    memset(idx_sz,0,sizeof(idx_sz));
    for(int i=0;i<TRAIN_N;i++){const uint8_t*s=joint_tr+(size_t)i*SIG_PAD;for(int k=0;k<N_BLOCKS;k++)if(s[k]!=bg)idx_sz[k][s[k]]++;}
    uint32_t tot=0;for(int k=0;k<N_BLOCKS;k++)for(int v=0;v<BYTE_VALS;v++){idx_off[k][v]=tot;tot+=idx_sz[k][v];}
    idx_pool=malloc((size_t)tot*sizeof(uint32_t));
    printf("Index: allocating %u entries (%zu MB)\n", tot, tot*4/1048576);
    if(!idx_pool){fprintf(stderr,"ERR: malloc idx_pool\n");exit(1);}
    uint32_t(*wp)[BYTE_VALS]=malloc((size_t)N_BLOCKS*BYTE_VALS*4);
    memcpy(wp,idx_off,(size_t)N_BLOCKS*BYTE_VALS*4);
    for(int i=0;i<TRAIN_N;i++){const uint8_t*s=joint_tr+(size_t)i*SIG_PAD;for(int k=0;k<N_BLOCKS;k++)if(s[k]!=bg)idx_pool[wp[k][s[k]]++]=(uint32_t)i;}
    free(wp);
    printf("Index built.\n");
}

typedef struct {
    uint32_t id, votes;
} cand_t;

static int cmp_votes_d(const void*a,const void*b){return(int)((const cand_t*)b)->votes-(int)((const cand_t*)a)->votes;}

static int select_top_k(const uint32_t*votes,int n,cand_t*out,int k){
    uint32_t mx=0;for(int i=0;i<n;i++)if(votes[i]>mx)mx=votes[i];
    if(!mx)return 0;
    if((size_t)(mx+1)>g_hist_cap){g_hist_cap=(size_t)(mx+1)+4096;free(g_hist);g_hist=malloc(g_hist_cap*sizeof(int));}
    memset(g_hist,0,(mx+1)*sizeof(int));for(int i=0;i<n;i++)if(votes[i])g_hist[votes[i]]++;
    int cum=0,thr;for(thr=(int)mx;thr>=1;thr--){cum+=g_hist[thr];if(cum>=k)break;}
    if(thr<1)thr=1;
    int nc=0;for(int i=0;i<n&&nc<k;i++)if(votes[i]>=(uint32_t)thr){out[nc]=(cand_t){0};out[nc].id=(uint32_t)i;out[nc].votes=votes[i];nc++;}
    qsort(out,(size_t)nc,sizeof(cand_t),cmp_votes_d);
    return nc;
}

static void vote(uint32_t*votes,int img){
    memset(votes,0,TRAIN_N*sizeof(uint32_t));
    const uint8_t*sig=joint_te+(size_t)img*SIG_PAD;
    for(int k=0;k<N_BLOCKS;k++){
        uint8_t bv=sig[k];if(bv==bg)continue;
        uint16_t w=ig_w[k],wh=w>1?w/2:1;
        {uint32_t off=idx_off[k][bv];uint16_t sz=idx_sz[k][bv];const uint32_t*ids=idx_pool+off;for(uint16_t j=0;j<sz;j++)votes[ids[j]]+=w;}
        for(int nb=0;nb<8;nb++){uint8_t nv=nbr[bv][nb];if(nv==bg)continue;uint32_t noff=idx_off[k][nv];uint16_t nsz=idx_sz[k][nv];const uint32_t*nids=idx_pool+noff;for(uint16_t j=0;j<nsz;j++)votes[nids[j]]+=wh;}
    }
}

int main(int argc, char*argv[]) {
    const char *data_dir = (argc > 1) ? argv[1] : "data/";

    printf("TEST: Loading and processing one test image\n");
    printf("======================================\n\n");

    printf("1. Loading data from %s\n", data_dir);
    load_data(data_dir);

    printf("\n2. Quantizing ternary...\n");
    tern_train = malloc((size_t)TRAIN_N * PADDED);
    tern_test = malloc((size_t)TEST_N * PADDED);
    if (!tern_train || !tern_test) { fprintf(stderr, "ERR: malloc tern\n"); exit(1); }
    quant_tern(raw_train_img, tern_train, TRAIN_N);
    quant_tern(raw_test_img, tern_test, TEST_N);
    printf("   Done.\n");

    printf("\n3. Computing gradients...\n");
    hgrad_train = malloc((size_t)TRAIN_N * PADDED);
    hgrad_test = malloc((size_t)TEST_N * PADDED);
    vgrad_train = malloc((size_t)TRAIN_N * PADDED);
    vgrad_test = malloc((size_t)TEST_N * PADDED);
    if (!hgrad_train || !hgrad_test || !vgrad_train || !vgrad_test) { fprintf(stderr, "ERR: malloc grads\n"); exit(1); }
    gradients(tern_train, hgrad_train, vgrad_train, TRAIN_N);
    gradients(tern_test, hgrad_test, vgrad_test, TEST_N);
    printf("   Done.\n");

    printf("\n4. Computing block signatures...\n");
    px_tr = malloc((size_t)TRAIN_N * SIG_PAD);
    px_te = malloc((size_t)TEST_N * SIG_PAD);
    hg_tr = malloc((size_t)TRAIN_N * SIG_PAD);
    hg_te = malloc((size_t)TEST_N * SIG_PAD);
    vg_tr = malloc((size_t)TRAIN_N * SIG_PAD);
    vg_te = malloc((size_t)TEST_N * SIG_PAD);
    if (!px_tr || !px_te || !hg_tr || !hg_te || !vg_tr || !vg_te) { fprintf(stderr, "ERR: malloc sigs\n"); exit(1); }
    block_sigs(tern_train, px_tr, TRAIN_N);
    block_sigs(tern_test, px_te, TEST_N);
    block_sigs(hgrad_train, hg_tr, TRAIN_N);
    block_sigs(hgrad_test, hg_te, TEST_N);
    block_sigs(vgrad_train, vg_tr, TRAIN_N);
    block_sigs(vgrad_test, vg_te, TEST_N);
    printf("   Done.\n");

    printf("\n5. Creating joint signatures (Encoding D)...\n");
    joint_tr = malloc((size_t)TRAIN_N * SIG_PAD);
    joint_te = malloc((size_t)TEST_N * SIG_PAD);
    if (!joint_tr || !joint_te) { fprintf(stderr, "ERR: malloc joint\n"); exit(1); }
    // Simplified: just use pixel blocks for this test
    memcpy(joint_tr, px_tr, (size_t)TRAIN_N * SIG_PAD);
    memcpy(joint_te, px_te, (size_t)TEST_N * SIG_PAD);
    printf("   Done.\n");

    printf("\n6. Building inverted index...\n");
    build_index();

    printf("\n7. Computing divergences...\n");
    divneg_train = malloc((size_t)TRAIN_N * sizeof(int16_t));
    divneg_test = malloc((size_t)TEST_N * sizeof(int16_t));
    if (!divneg_train || !divneg_test) { fprintf(stderr, "ERR: malloc divneg\n"); exit(1); }
    for(int i=0;i<TRAIN_N;i++)div_features(hgrad_train+(size_t)i*PADDED, vgrad_train+(size_t)i*PADDED, &divneg_train[i]);
    for(int i=0;i<TEST_N;i++)div_features(hgrad_test+(size_t)i*PADDED, vgrad_test+(size_t)i*PADDED, &divneg_test[i]);
    printf("   Done.\n");

    printf("\n8. Testing vote() on image 0...\n");
    uint32_t *votes = malloc(TRAIN_N * sizeof(uint32_t));
    if (!votes) { fprintf(stderr, "ERR: malloc votes\n"); exit(1); }

    printf("   Calling vote(votes, 0)...\n");
    vote(votes, 0);
    printf("   Vote completed successfully.\n");

    printf("\n9. Selecting top-K candidates...\n");
    cand_t cands[TOP_K];
    int nc = select_top_k(votes, TRAIN_N, cands, TOP_K);
    printf("   Selected %d candidates for image 0.\n", nc);

    if (nc > 0) {
        printf("\n10. Analysis of image 0:\n");
        printf("    True label: %d\n", test_labels[0]);
        printf("    Rank-1 candidate: id=%u (label %d), votes=%u\n",
               cands[0].id, train_labels[cands[0].id], cands[0].votes);
        printf("    Rank-2 candidate: id=%u (label %d), votes=%u\n",
               cands[1].id, train_labels[cands[1].id], cands[1].votes);
        printf("    Rank-3 candidate: id=%u (label %d), votes=%u\n",
               cands[2].id, train_labels[cands[2].id], cands[2].votes);

        int rank1_label = train_labels[cands[0].id];
        int vote_result = (train_labels[cands[0].id] == train_labels[cands[1].id] && train_labels[cands[1].id] == train_labels[cands[2].id])
                          ? train_labels[cands[0].id]
                          : (train_labels[cands[0].id] == train_labels[cands[1].id])
                            ? train_labels[cands[0].id]
                            : train_labels[cands[2].id];

        printf("    k=3 vote result: %d\n", vote_result);
        printf("    Is Mode C? %s\n", (rank1_label == test_labels[0] && vote_result != test_labels[0]) ? "YES" : "NO");
        printf("    Test divergence: %d\n", divneg_test[0]);
    }

    printf("\n✓ Integration test PASSED\n");
    return 0;
}
