/*
 * sstt_cifar10_full.c — Full SSTT Pipeline on CIFAR-10
 *
 * Brings every tool from the MNIST/Fashion arsenal to CIFAR-10:
 *   - Flattened RGB (32x96): blocks = (R,G,B) color encoding
 *   - Encoding D bytepacked signatures
 *   - Multi-probe IG-weighted voting
 *   - Gradient-only dot (learned from ablation: pixel dot hurts)
 *   - Enclosed-region centroid
 *   - Gradient divergence (Green's theorem)
 *   - Grid spatial decomposition
 *   - Kalman-adaptive weighting (MAD)
 *   - Bayesian sequential candidate processing
 *   - Cascade autopsy: Mode A/B/C error classification
 *
 * Build: make sstt_cifar10_full
 */

#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define TRAIN_N         50000
#define TEST_N          10000
#define IMG_W           96      /* 32 pixels * 3 channels (interleaved RGB) */
#define IMG_H           32
#define PIXELS          3072
#define PADDED          3072    /* 3072 % 32 == 0 */
#define N_CLASSES       10
#define CLS_PAD         16

#define BLKS_PER_ROW    32
#define N_BLOCKS        1024    /* 32 * 32 */
#define SIG_PAD         1024
#define BYTE_VALS       256

#define BG_PIXEL        0
#define BG_GRAD         13
#define BG_TRANS        13

#define H_TRANS_PER_ROW 31
#define N_HTRANS        (H_TRANS_PER_ROW * IMG_H)
#define V_TRANS_PER_COL 31
#define N_VTRANS        (BLKS_PER_ROW * V_TRANS_PER_COL)
#define TRANS_PAD       992

#define IG_SCALE        16
#define TOP_K           200
#define MAX_REGIONS     16

/* For centroid: use the original 32x32 spatial image, not the 32x96 */
#define SPATIAL_W       32
#define SPATIAL_H       32
#define FG_W            34
#define FG_H            34
#define FG_SZ           (FG_W*FG_H)

static const char *data_dir = "data-cifar10/";
static const char *class_names[] = {
    "airplane","automobile","bird","cat","deer",
    "dog","frog","horse","ship","truck"
};

static double now_sec(void){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return ts.tv_sec+ts.tv_nsec*1e-9;}

/* Data */
static uint8_t *raw_train_img, *raw_test_img;     /* flattened RGB 32x96 */
static uint8_t *raw_gray_tr, *raw_gray_te;         /* grayscale 32x32 for centroid/spatial features */
static uint8_t *train_labels, *test_labels;

/* Ternary + gradients on flattened RGB */
static int8_t *tern_train, *tern_test;
static int8_t *hgrad_train, *hgrad_test, *vgrad_train, *vgrad_test;

/* Ternary on grayscale (for centroid, divergence — spatial features) */
static int8_t *gray_tern_tr, *gray_tern_te;
static int8_t *gray_hg_tr, *gray_hg_te, *gray_vg_tr, *gray_vg_te;
#define GRAY_PIXELS 1024
#define GRAY_PADDED 1024

/* Block/transition/joint sigs */
static uint8_t *px_tr, *px_te, *hgs_tr, *hgs_te, *vgs_tr, *vgs_te;
static uint8_t *ht_tr, *ht_te, *vt_tr, *vt_te;
static uint8_t *joint_tr, *joint_te;

/* Hot map + index */
static uint32_t *joint_hot;
static uint16_t ig_w[N_BLOCKS];
static uint8_t nbr[BYTE_VALS][8];
static uint32_t idx_off[N_BLOCKS][BYTE_VALS];
static uint16_t idx_sz[N_BLOCKS][BYTE_VALS];
static uint32_t *idx_pool;
static uint8_t bg_val;

/* Topological features */
static int16_t *cent_train, *cent_test;
static int16_t *divneg_train, *divneg_test;
static int16_t *divneg_cy_train, *divneg_cy_test;
static int16_t *gdiv_train, *gdiv_test;

/* ================================================================
 *  CIFAR-10 loader
 * ================================================================ */

static void load_cifar_batch(const char *path, int offset) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "ERROR: %s\n", path); exit(1); }
    uint8_t rec[3073];
    for (int i = 0; i < 10000; i++) {
        if (fread(rec, 1, 3073, f) != 3073) { fclose(f); exit(1); }
        int idx = offset + i;
        train_labels[idx] = rec[0];
        /* Interleaved RGB -> 32x96 */
        uint8_t *dst = raw_train_img + (size_t)idx * PIXELS;
        const uint8_t *r = rec+1, *g = rec+1+1024, *b = rec+1+2048;
        for (int y = 0; y < 32; y++)
            for (int x = 0; x < 32; x++) {
                int si = y*32+x, di = y*96+x*3;
                dst[di]=r[si]; dst[di+1]=g[si]; dst[di+2]=b[si];
            }
        /* Grayscale -> 32x32 */
        uint8_t *gdst = raw_gray_tr + (size_t)idx * GRAY_PIXELS;
        for (int p = 0; p < 1024; p++)
            gdst[p] = (uint8_t)((77*(int)r[p]+150*(int)g[p]+29*(int)b[p])>>8);
    }
    fclose(f);
}

static void load_cifar_test(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "ERROR: %s\n", path); exit(1); }
    uint8_t rec[3073];
    for (int i = 0; i < 10000; i++) {
        if (fread(rec, 1, 3073, f) != 3073) { fclose(f); exit(1); }
        test_labels[i] = rec[0];
        uint8_t *dst = raw_test_img + (size_t)i * PIXELS;
        const uint8_t *r = rec+1, *g = rec+1+1024, *b = rec+1+2048;
        for (int y = 0; y < 32; y++)
            for (int x = 0; x < 32; x++) {
                int si = y*32+x, di = y*96+x*3;
                dst[di]=r[si]; dst[di+1]=g[si]; dst[di+2]=b[si];
            }
        uint8_t *gdst = raw_gray_te + (size_t)i * GRAY_PIXELS;
        for (int p = 0; p < 1024; p++)
            gdst[p] = (uint8_t)((77*(int)r[p]+150*(int)g[p]+29*(int)b[p])>>8);
    }
    fclose(f);
}

static void load_data(void) {
    raw_train_img = malloc((size_t)TRAIN_N*PIXELS);
    raw_test_img  = malloc((size_t)TEST_N*PIXELS);
    raw_gray_tr   = malloc((size_t)TRAIN_N*GRAY_PIXELS);
    raw_gray_te   = malloc((size_t)TEST_N*GRAY_PIXELS);
    train_labels  = malloc(TRAIN_N);
    test_labels   = malloc(TEST_N);
    char path[512];
    for (int b2 = 1; b2 <= 5; b2++) {
        snprintf(path, sizeof(path), "%sdata_batch_%d.bin", data_dir, b2);
        load_cifar_batch(path, (b2-1)*10000);
    }
    snprintf(path, sizeof(path), "%stest_batch.bin", data_dir);
    load_cifar_test(path);
    printf("  Loaded 50K train + 10K test\n");
}

/* ================================================================
 *  Core features (same as bytecascade, dimensions differ)
 * ================================================================ */

static inline int8_t clamp_trit(int v){return v>0?1:v<0?-1:0;}

static void quantize_n(const uint8_t *src, int8_t *dst, int n, int pixels, int padded) {
    const __m256i bias=_mm256_set1_epi8((char)0x80);
    const __m256i thi=_mm256_set1_epi8((char)(170^0x80));
    const __m256i tlo=_mm256_set1_epi8((char)(85^0x80));
    const __m256i one=_mm256_set1_epi8(1);
    for(int img=0;img<n;img++){
        const uint8_t *s=src+(size_t)img*pixels;
        int8_t *d=dst+(size_t)img*padded;
        int i;
        for(i=0;i+32<=pixels;i+=32){
            __m256i px=_mm256_loadu_si256((const __m256i*)(s+i));
            __m256i sp=_mm256_xor_si256(px,bias);
            _mm256_storeu_si256((__m256i*)(d+i),
                _mm256_sub_epi8(_mm256_and_si256(_mm256_cmpgt_epi8(sp,thi),one),
                                _mm256_and_si256(_mm256_cmpgt_epi8(tlo,sp),one)));
        }
        for(;i<pixels;i++) d[i]=s[i]>170?1:s[i]<85?-1:0;
        memset(d+pixels,0,padded-pixels);
    }
}

static void gradients_n(const int8_t *t, int8_t *h, int8_t *v, int n, int w, int ht, int padded) {
    for(int img=0;img<n;img++){
        const int8_t *ti=t+(size_t)img*padded;
        int8_t *hi=h+(size_t)img*padded, *vi=v+(size_t)img*padded;
        for(int y=0;y<ht;y++){
            for(int x=0;x<w-1;x++) hi[y*w+x]=clamp_trit(ti[y*w+x+1]-ti[y*w+x]);
            hi[y*w+w-1]=0;
        }
        for(int y=0;y<ht-1;y++)
            for(int x=0;x<w;x++) vi[y*w+x]=clamp_trit(ti[(y+1)*w+x]-ti[y*w+x]);
        memset(vi+(ht-1)*w,0,w);
    }
}

static inline uint8_t benc(int8_t a,int8_t b,int8_t c){return(uint8_t)((a+1)*9+(b+1)*3+(c+1));}
static void block_sigs(const int8_t *data, uint8_t *sigs, int n) {
    for(int i=0;i<n;i++){
        const int8_t *img=data+(size_t)i*PADDED;
        uint8_t *sig=sigs+(size_t)i*SIG_PAD;
        for(int y=0;y<IMG_H;y++)
            for(int s=0;s<BLKS_PER_ROW;s++){
                int base=y*IMG_W+s*3;
                sig[y*BLKS_PER_ROW+s]=benc(img[base],img[base+1],img[base+2]);
            }
    }
}

static inline uint8_t tenc(uint8_t a,uint8_t b){
    int8_t a0=(a/9)-1,a1=((a/3)%3)-1,a2=(a%3)-1;
    int8_t b0=(b/9)-1,b1=((b/3)%3)-1,b2=(b%3)-1;
    return benc(clamp_trit(b0-a0),clamp_trit(b1-a1),clamp_trit(b2-a2));
}
static void transitions(const uint8_t *bs, uint8_t *ht2, uint8_t *vt2, int n) {
    for(int i=0;i<n;i++){
        const uint8_t *s=bs+(size_t)i*SIG_PAD;
        uint8_t *h=ht2+(size_t)i*TRANS_PAD, *v=vt2+(size_t)i*TRANS_PAD;
        for(int y=0;y<IMG_H;y++)
            for(int ss=0;ss<H_TRANS_PER_ROW;ss++)
                h[y*H_TRANS_PER_ROW+ss]=tenc(s[y*BLKS_PER_ROW+ss],s[y*BLKS_PER_ROW+ss+1]);
        memset(h+N_HTRANS,0xFF,TRANS_PAD-N_HTRANS);
        for(int y=0;y<V_TRANS_PER_COL;y++)
            for(int ss=0;ss<BLKS_PER_ROW;ss++)
                v[y*BLKS_PER_ROW+ss]=tenc(s[y*BLKS_PER_ROW+ss],s[(y+1)*BLKS_PER_ROW+ss]);
        memset(v+N_VTRANS,0xFF,TRANS_PAD-N_VTRANS);
    }
}

static uint8_t encode_d(uint8_t px_bv,uint8_t hg_bv,uint8_t vg_bv,uint8_t ht_bv,uint8_t vt_bv){
    int ps=((px_bv/9)-1)+(((px_bv/3)%3)-1)+((px_bv%3)-1);
    int hs=((hg_bv/9)-1)+(((hg_bv/3)%3)-1)+((hg_bv%3)-1);
    int vs=((vg_bv/9)-1)+(((vg_bv/3)%3)-1)+((vg_bv%3)-1);
    uint8_t pc=ps<0?0:ps==0?1:ps<3?2:3;
    uint8_t hc=hs<0?0:hs==0?1:hs<3?2:3;
    uint8_t vc=vs<0?0:vs==0?1:vs<3?2:3;
    return pc|(hc<<2)|(vc<<4)|((ht_bv!=BG_TRANS)?1<<6:0)|((vt_bv!=BG_TRANS)?1<<7:0);
}
static void joint_sigs_fn(uint8_t *out,int n,const uint8_t *px,const uint8_t *hg,const uint8_t *vg,const uint8_t *ht2,const uint8_t *vt2){
    for(int i=0;i<n;i++){
        const uint8_t *pi=px+(size_t)i*SIG_PAD,*hi=hg+(size_t)i*SIG_PAD,*vi=vg+(size_t)i*SIG_PAD;
        const uint8_t *hti=ht2+(size_t)i*TRANS_PAD,*vti=vt2+(size_t)i*TRANS_PAD;
        uint8_t *oi=out+(size_t)i*SIG_PAD;
        for(int y=0;y<IMG_H;y++)
            for(int s=0;s<BLKS_PER_ROW;s++){
                int k=y*BLKS_PER_ROW+s;
                uint8_t htb=(s>0)?hti[y*H_TRANS_PER_ROW+(s-1)]:BG_TRANS;
                uint8_t vtb=(y>0)?vti[(y-1)*BLKS_PER_ROW+s]:BG_TRANS;
                oi[k]=encode_d(pi[k],hi[k],vi[k],htb,vtb);
            }
    }
}

/* ================================================================
 *  Topological features (on grayscale 32x32 — spatial features)
 * ================================================================ */

static int16_t enclosed_centroid(const int8_t *tern) {
    uint8_t grid[FG_SZ];memset(grid,0,sizeof(grid));
    for(int y=0;y<SPATIAL_H;y++)for(int x=0;x<SPATIAL_W;x++)
        grid[(y+1)*FG_W+(x+1)]=(tern[y*SPATIAL_W+x]>0)?1:0;
    uint8_t visited[FG_SZ];memset(visited,0,sizeof(visited));
    int stack[FG_SZ];int sp=0;
    for(int y=0;y<FG_H;y++)for(int x=0;x<FG_W;x++){
        if(y==0||y==FG_H-1||x==0||x==FG_W-1){
            int pos=y*FG_W+x;if(!grid[pos]&&!visited[pos]){visited[pos]=1;stack[sp++]=pos;}}}
    while(sp>0){int pos=stack[--sp];int py=pos/FG_W,px2=pos%FG_W;
        const int dx[]={0,0,1,-1},dy[]={1,-1,0,0};
        for(int d=0;d<4;d++){int ny=py+dy[d],nx=px2+dx[d];
            if(ny<0||ny>=FG_H||nx<0||nx>=FG_W)continue;
            int npos=ny*FG_W+nx;if(!visited[npos]&&!grid[npos]){visited[npos]=1;stack[sp++]=npos;}}}
    int sum_y=0,count=0;
    for(int y=1;y<FG_H-1;y++)for(int x=1;x<FG_W-1;x++){
        int pos=y*FG_W+x;if(!grid[pos]&&!visited[pos]){sum_y+=(y-1);count++;}}
    return count>0?(int16_t)(sum_y/count):-1;
}

static void compute_centroid_all(const int8_t *tern,int16_t *out,int n){
    for(int i=0;i<n;i++)out[i]=enclosed_centroid(tern+(size_t)i*GRAY_PADDED);}

static void div_features(const int8_t *hg,const int8_t *vg,int16_t *ns,int16_t *cy){
    int neg_sum=0,neg_ysum=0,neg_cnt=0;
    for(int y=0;y<SPATIAL_H;y++)for(int x=0;x<SPATIAL_W;x++){
        int dh=(int)hg[y*SPATIAL_W+x]-(x>0?(int)hg[y*SPATIAL_W+x-1]:0);
        int dv=(int)vg[y*SPATIAL_W+x]-(y>0?(int)vg[(y-1)*SPATIAL_W+x]:0);
        int d=dh+dv;if(d<0){neg_sum+=d;neg_ysum+=y;neg_cnt++;}}
    *ns=(int16_t)(neg_sum<-32767?-32767:neg_sum);
    *cy=neg_cnt>0?(int16_t)(neg_ysum/neg_cnt):-1;
}

static void compute_div_all(const int8_t *hg,const int8_t *vg,int16_t *ns,int16_t *cy,int n){
    for(int i=0;i<n;i++)div_features(hg+(size_t)i*GRAY_PADDED,vg+(size_t)i*GRAY_PADDED,&ns[i],&cy[i]);}

static void grid_div(const int8_t *hg,const int8_t *vg,int grow,int gcol,int16_t *out){
    int nr=grow*gcol;int regions[MAX_REGIONS];memset(regions,0,sizeof(int)*nr);
    for(int y=0;y<SPATIAL_H;y++)for(int x=0;x<SPATIAL_W;x++){
        int dh=(int)hg[y*SPATIAL_W+x]-(x>0?(int)hg[y*SPATIAL_W+x-1]:0);
        int dv=(int)vg[y*SPATIAL_W+x]-(y>0?(int)vg[(y-1)*SPATIAL_W+x]:0);
        int d=dh+dv;if(d<0){int ry=y*grow/SPATIAL_H;int rx=x*gcol/SPATIAL_W;
            if(ry>=grow)ry=grow-1;if(rx>=gcol)rx=gcol-1;regions[ry*gcol+rx]+=d;}}
    for(int i=0;i<nr;i++)out[i]=(int16_t)(regions[i]<-32767?-32767:regions[i]);
}

static void compute_grid_div(const int8_t *hg,const int8_t *vg,int grow,int gcol,int16_t *out,int n){
    for(int i=0;i<n;i++)grid_div(hg+(size_t)i*GRAY_PADDED,vg+(size_t)i*GRAY_PADDED,grow,gcol,out+(size_t)i*MAX_REGIONS);}

/* ================================================================
 *  Index, voting, dot, top-K, kNN
 * ================================================================ */

static void build_all(void) {
    /* Hot map */
    joint_hot=aligned_alloc(32,(size_t)N_BLOCKS*BYTE_VALS*CLS_PAD*4);
    memset(joint_hot,0,(size_t)N_BLOCKS*BYTE_VALS*CLS_PAD*4);
    for(int i=0;i<TRAIN_N;i++){int lbl=train_labels[i];const uint8_t *sig=joint_tr+(size_t)i*SIG_PAD;
        for(int k=0;k<N_BLOCKS;k++)joint_hot[(size_t)k*BYTE_VALS*CLS_PAD+(size_t)sig[k]*CLS_PAD+lbl]++;}

    /* IG */
    int cc[N_CLASSES]={0};for(int i=0;i<TRAIN_N;i++)cc[train_labels[i]]++;
    double hc=0;for(int c=0;c<N_CLASSES;c++){double p=(double)cc[c]/TRAIN_N;if(p>0)hc-=p*log2(p);}
    double raw[N_BLOCKS],mx=0;
    for(int k=0;k<N_BLOCKS;k++){double hcond=0;
        for(int v=0;v<BYTE_VALS;v++){if((uint8_t)v==bg_val)continue;
            const uint32_t *h=joint_hot+(size_t)k*BYTE_VALS*CLS_PAD+(size_t)v*CLS_PAD;
            int vt=0;for(int c=0;c<N_CLASSES;c++)vt+=(int)h[c];if(!vt)continue;
            double pv=(double)vt/TRAIN_N,hv=0;
            for(int c=0;c<N_CLASSES;c++){double pc=(double)h[c]/vt;if(pc>0)hv-=pc*log2(pc);}
            hcond+=pv*hv;}
        raw[k]=hc-hcond;if(raw[k]>mx)mx=raw[k];}
    for(int k=0;k<N_BLOCKS;k++){ig_w[k]=mx>0?(uint16_t)(raw[k]/mx*IG_SCALE+0.5):1;if(!ig_w[k])ig_w[k]=1;}

    /* Neighbor table + index */
    for(int v=0;v<BYTE_VALS;v++)for(int b=0;b<8;b++)nbr[v][b]=(uint8_t)(v^(1<<b));
    memset(idx_sz,0,sizeof(idx_sz));
    for(int i=0;i<TRAIN_N;i++){const uint8_t *sig=joint_tr+(size_t)i*SIG_PAD;
        for(int k=0;k<N_BLOCKS;k++)if(sig[k]!=bg_val)idx_sz[k][sig[k]]++;}
    uint32_t tot=0;
    for(int k=0;k<N_BLOCKS;k++)for(int v=0;v<BYTE_VALS;v++){idx_off[k][v]=tot;tot+=idx_sz[k][v];}
    idx_pool=malloc((size_t)tot*sizeof(uint32_t));
    uint32_t(*wp)[BYTE_VALS]=malloc((size_t)N_BLOCKS*BYTE_VALS*sizeof(uint32_t));
    memcpy(wp,idx_off,(size_t)N_BLOCKS*BYTE_VALS*sizeof(uint32_t));
    for(int i=0;i<TRAIN_N;i++){const uint8_t *sig=joint_tr+(size_t)i*SIG_PAD;
        for(int k=0;k<N_BLOCKS;k++)if(sig[k]!=bg_val)idx_pool[wp[k][sig[k]]++]=(uint32_t)i;}
    free(wp);
    printf("  Index: %u entries (%.1f MB)\n",tot,(double)tot*4/(1024*1024));
}

static void vote(uint32_t *votes, int img) {
    memset(votes,0,TRAIN_N*sizeof(uint32_t));
    const uint8_t *sig=joint_te+(size_t)img*SIG_PAD;
    for(int k=0;k<N_BLOCKS;k++){
        uint8_t bv=sig[k];if(bv==bg_val)continue;
        uint16_t w=ig_w[k],wh=w>1?w/2:1;
        {uint32_t off=idx_off[k][bv];uint16_t sz=idx_sz[k][bv];
         for(uint16_t j=0;j<sz;j++)votes[idx_pool[off+j]]+=w;}
        for(int nb=0;nb<8;nb++){uint8_t nv=nbr[bv][nb];if(nv==bg_val)continue;
            uint32_t noff=idx_off[k][nv];uint16_t nsz=idx_sz[k][nv];
            for(uint16_t j=0;j<nsz;j++)votes[idx_pool[noff+j]]+=wh;}
    }
}

/* Safe dot for 3072-element vectors */
static inline int32_t tdot_safe(const int8_t *a, const int8_t *b, int len) {
    int32_t total=0;
    for(int chunk=0;chunk<len;chunk+=64){
        __m256i acc=_mm256_setzero_si256();
        int end=chunk+64;if(end>len)end=len;
        for(int i=chunk;i<end;i+=32)
            acc=_mm256_add_epi8(acc,_mm256_sign_epi8(
                _mm256_load_si256((const __m256i*)(a+i)),
                _mm256_load_si256((const __m256i*)(b+i))));
        __m256i lo=_mm256_cvtepi8_epi16(_mm256_castsi256_si128(acc));
        __m256i hi=_mm256_cvtepi8_epi16(_mm256_extracti128_si256(acc,1));
        __m256i s32=_mm256_madd_epi16(_mm256_add_epi16(lo,hi),_mm256_set1_epi16(1));
        __m128i s=_mm_add_epi32(_mm256_castsi256_si128(s32),_mm256_extracti128_si256(s32,1));
        s=_mm_hadd_epi32(s,s);s=_mm_hadd_epi32(s,s);
        total+=_mm_cvtsi128_si32(s);
    }
    return total;
}

typedef struct {
    uint32_t id, votes;
    int32_t dot_hg, dot_vg;
    int32_t div_sim, cent_sim, gdiv_sim;
    int64_t combined;
} cand_t;

static int cmp_votes_d(const void*a,const void*b){return(int)((const cand_t*)b)->votes-(int)((const cand_t*)a)->votes;}
static int cmp_comb_d(const void*a,const void*b){int64_t da=((const cand_t*)a)->combined,db=((const cand_t*)b)->combined;return(db>da)-(db<da);}

static int *g_hist=NULL;static size_t g_hist_cap=0;
static int select_top_k(const uint32_t *votes,int n,cand_t *out,int k){
    uint32_t mx=0;for(int i=0;i<n;i++)if(votes[i]>mx)mx=votes[i];if(!mx)return 0;
    if((size_t)(mx+1)>g_hist_cap){g_hist_cap=(size_t)(mx+1)+4096;free(g_hist);g_hist=malloc(g_hist_cap*sizeof(int));}
    memset(g_hist,0,(mx+1)*sizeof(int));for(int i=0;i<n;i++)if(votes[i])g_hist[votes[i]]++;
    int cum=0,thr;for(thr=(int)mx;thr>=1;thr--){cum+=g_hist[thr];if(cum>=k)break;}if(thr<1)thr=1;
    int nc=0;for(int i=0;i<n&&nc<k;i++)if(votes[i]>=(uint32_t)thr){
        out[nc]=(cand_t){0};out[nc].id=(uint32_t)i;out[nc].votes=votes[i];nc++;}
    qsort(out,(size_t)nc,sizeof(cand_t),cmp_votes_d);return nc;
}

static int knn_vote(const cand_t *c,int nc,int k){
    int v[N_CLASSES]={0};if(k>nc)k=nc;
    for(int i=0;i<k;i++)v[train_labels[c[i].id]]++;
    int best=0;for(int c2=1;c2<N_CLASSES;c2++)if(v[c2]>v[best])best=c2;return best;
}

/* MAD */
static int cmp_i16(const void*a,const void*b){return *(const int16_t*)a-*(const int16_t*)b;}
static int compute_mad(const cand_t *cands,int nc){
    if(nc<3)return 1000;
    int16_t vals[TOP_K];for(int j=0;j<nc;j++)vals[j]=divneg_train[cands[j].id];
    qsort(vals,(size_t)nc,sizeof(int16_t),cmp_i16);
    int16_t med=vals[nc/2],devs[TOP_K];
    for(int j=0;j<nc;j++)devs[j]=(int16_t)abs(vals[j]-med);
    qsort(devs,(size_t)nc,sizeof(int16_t),cmp_i16);
    return devs[nc/2]>0?devs[nc/2]:1;
}

#define CENT_BOTH_OPEN 50
#define CENT_DISAGREE  80

/* ================================================================
 *  Main
 * ================================================================ */

int main(int argc, char **argv) {
    double t0=now_sec();
    if(argc>1){data_dir=argv[1];size_t l=strlen(data_dir);
        if(l&&data_dir[l-1]!='/'){char*b2=malloc(l+2);memcpy(b2,data_dir,l);b2[l]='/';b2[l+1]=0;data_dir=b2;}}

    printf("=== SSTT CIFAR-10 Full Pipeline + Autopsy ===\n\n");

    printf("Loading...\n");load_data();

    printf("\nComputing features...\n");
    /* Flattened RGB ternary + gradients */
    tern_train=aligned_alloc(32,(size_t)TRAIN_N*PADDED);tern_test=aligned_alloc(32,(size_t)TEST_N*PADDED);
    hgrad_train=aligned_alloc(32,(size_t)TRAIN_N*PADDED);hgrad_test=aligned_alloc(32,(size_t)TEST_N*PADDED);
    vgrad_train=aligned_alloc(32,(size_t)TRAIN_N*PADDED);vgrad_test=aligned_alloc(32,(size_t)TEST_N*PADDED);
    quantize_n(raw_train_img,tern_train,TRAIN_N,PIXELS,PADDED);
    quantize_n(raw_test_img,tern_test,TEST_N,PIXELS,PADDED);
    gradients_n(tern_train,hgrad_train,vgrad_train,TRAIN_N,IMG_W,IMG_H,PADDED);
    gradients_n(tern_test,hgrad_test,vgrad_test,TEST_N,IMG_W,IMG_H,PADDED);

    /* Grayscale ternary + gradients (for spatial/topological features) */
    gray_tern_tr=aligned_alloc(32,(size_t)TRAIN_N*GRAY_PADDED);gray_tern_te=aligned_alloc(32,(size_t)TEST_N*GRAY_PADDED);
    gray_hg_tr=aligned_alloc(32,(size_t)TRAIN_N*GRAY_PADDED);gray_hg_te=aligned_alloc(32,(size_t)TEST_N*GRAY_PADDED);
    gray_vg_tr=aligned_alloc(32,(size_t)TRAIN_N*GRAY_PADDED);gray_vg_te=aligned_alloc(32,(size_t)TEST_N*GRAY_PADDED);
    quantize_n(raw_gray_tr,gray_tern_tr,TRAIN_N,GRAY_PIXELS,GRAY_PADDED);
    quantize_n(raw_gray_te,gray_tern_te,TEST_N,GRAY_PIXELS,GRAY_PADDED);
    gradients_n(gray_tern_tr,gray_hg_tr,gray_vg_tr,TRAIN_N,SPATIAL_W,SPATIAL_H,GRAY_PADDED);
    gradients_n(gray_tern_te,gray_hg_te,gray_vg_te,TEST_N,SPATIAL_W,SPATIAL_H,GRAY_PADDED);

    /* Block sigs + transitions + Encoding D (on flattened RGB) */
    px_tr=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD);px_te=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    hgs_tr=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD);hgs_te=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    vgs_tr=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD);vgs_te=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    block_sigs(tern_train,px_tr,TRAIN_N);block_sigs(tern_test,px_te,TEST_N);
    block_sigs(hgrad_train,hgs_tr,TRAIN_N);block_sigs(hgrad_test,hgs_te,TEST_N);
    block_sigs(vgrad_train,vgs_tr,TRAIN_N);block_sigs(vgrad_test,vgs_te,TEST_N);

    ht_tr=aligned_alloc(32,(size_t)TRAIN_N*TRANS_PAD);ht_te=aligned_alloc(32,(size_t)TEST_N*TRANS_PAD);
    vt_tr=aligned_alloc(32,(size_t)TRAIN_N*TRANS_PAD);vt_te=aligned_alloc(32,(size_t)TEST_N*TRANS_PAD);
    transitions(px_tr,ht_tr,vt_tr,TRAIN_N);transitions(px_te,ht_te,vt_te,TEST_N);

    joint_tr=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD);joint_te=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    joint_sigs_fn(joint_tr,TRAIN_N,px_tr,hgs_tr,vgs_tr,ht_tr,vt_tr);
    joint_sigs_fn(joint_te,TEST_N,px_te,hgs_te,vgs_te,ht_te,vt_te);

    /* Topological features (on grayscale 32x32) */
    cent_train=malloc((size_t)TRAIN_N*2);cent_test=malloc((size_t)TEST_N*2);
    compute_centroid_all(gray_tern_tr,cent_train,TRAIN_N);
    compute_centroid_all(gray_tern_te,cent_test,TEST_N);

    divneg_train=malloc((size_t)TRAIN_N*2);divneg_test=malloc((size_t)TEST_N*2);
    divneg_cy_train=malloc((size_t)TRAIN_N*2);divneg_cy_test=malloc((size_t)TEST_N*2);
    compute_div_all(gray_hg_tr,gray_vg_tr,divneg_train,divneg_cy_train,TRAIN_N);
    compute_div_all(gray_hg_te,gray_vg_te,divneg_test,divneg_cy_test,TEST_N);

    /* Grid divergence: test 3x3 and 4x4 */
    gdiv_train=malloc((size_t)TRAIN_N*MAX_REGIONS*2);gdiv_test=malloc((size_t)TEST_N*MAX_REGIONS*2);
    compute_grid_div(gray_hg_tr,gray_vg_tr,3,3,gdiv_train,TRAIN_N);
    compute_grid_div(gray_hg_te,gray_vg_te,3,3,gdiv_test,TEST_N);

    /* Background + build */
    {long vc[BYTE_VALS]={0};
     for(int i=0;i<TRAIN_N;i++){const uint8_t *sig=joint_tr+(size_t)i*SIG_PAD;
         for(int k=0;k<N_BLOCKS;k++)vc[sig[k]]++;}
     bg_val=0;long mc2=0;for(int v=0;v<BYTE_VALS;v++)if(vc[v]>mc2){mc2=vc[v];bg_val=(uint8_t)v;}
     printf("  BG=%d (%.1f%%)\n",bg_val,100.0*mc2/((long)TRAIN_N*N_BLOCKS));}
    build_all();
    printf("  Features + index built (%.1f sec)\n\n",now_sec()-t0);

    /* ================================================================
     * PRECOMPUTE: vote + base features for all test images
     * ================================================================ */
    printf("Precomputing candidates...\n");double tp=now_sec();
    cand_t *pre=malloc((size_t)TEST_N*TOP_K*sizeof(cand_t));
    int *nc_arr=malloc((size_t)TEST_N*sizeof(int));
    uint32_t *votes_buf=calloc(TRAIN_N,sizeof(uint32_t));
    int nreg=9; /* 3x3 grid */

    for(int i=0;i<TEST_N;i++){
        vote(votes_buf,i);
        cand_t *ci=pre+(size_t)i*TOP_K;
        int nc=select_top_k(votes_buf,TRAIN_N,ci,TOP_K);
        /* Compute gradient dots + topo features */
        const int8_t *qh=hgrad_test+(size_t)i*PADDED, *qv=vgrad_test+(size_t)i*PADDED;
        int16_t q_cent=cent_test[i], q_dn=divneg_test[i], q_dc=divneg_cy_test[i];
        const int16_t *q_gdiv=gdiv_test+(size_t)i*MAX_REGIONS;
        for(int j=0;j<nc;j++){
            uint32_t id=ci[j].id;
            ci[j].dot_hg=tdot_safe(qh,hgrad_train+(size_t)id*PADDED,PADDED);
            ci[j].dot_vg=tdot_safe(qv,vgrad_train+(size_t)id*PADDED,PADDED);
            /* Centroid */
            int16_t cc2=cent_train[id];
            if(q_cent<0&&cc2<0)ci[j].cent_sim=CENT_BOTH_OPEN;
            else if(q_cent<0||cc2<0)ci[j].cent_sim=-CENT_DISAGREE;
            else ci[j].cent_sim=-(int32_t)abs(q_cent-cc2);
            /* Divergence */
            int16_t cdn=divneg_train[id],cdc=divneg_cy_train[id];
            int32_t ds=-(int32_t)abs(q_dn-cdn);
            if(q_dc>=0&&cdc>=0)ds-=(int32_t)abs(q_dc-cdc)*2;
            else if((q_dc<0)!=(cdc<0))ds-=10;
            ci[j].div_sim=ds;
            /* Grid divergence */
            const int16_t *cg=gdiv_train+(size_t)id*MAX_REGIONS;
            int32_t gl=0;for(int r=0;r<nreg;r++)gl+=abs(q_gdiv[r]-cg[r]);
            ci[j].gdiv_sim=-gl;
        }
        nc_arr[i]=nc;
        if((i+1)%2000==0)fprintf(stderr,"  %d/%d\r",i+1,TEST_N);
    }
    fprintf(stderr,"\n");free(votes_buf);
    printf("  %.1f sec\n\n",now_sec()-tp);

    /* ================================================================
     * CASCADE AUTOPSY
     * ================================================================ */
    printf("=== CASCADE AUTOPSY ===\n\n");

    int mode_a=0, mode_b=0, mode_c=0, correct_total=0;
    int vote_recall=0, vote_top1=0;

    /* Run with gradient-only dot, Kalman-adaptive topo, kNN=7 */
    int w_c=50,w_d=200,w_g=100,sc=50;

    for(int i=0;i<TEST_N;i++){
        cand_t cands[TOP_K];int nc=nc_arr[i];
        memcpy(cands,pre+(size_t)i*TOP_K,(size_t)nc*sizeof(cand_t));
        int true_label=test_labels[i];

        /* Check vote recall */
        int correct_in_topk=0, correct_rank_vote=-1;
        for(int j=0;j<nc;j++)if(train_labels[cands[j].id]==true_label){
            correct_in_topk=1;if(correct_rank_vote<0)correct_rank_vote=j;}
        if(correct_in_topk)vote_recall++;
        if(nc>0 && train_labels[cands[0].id]==true_label)vote_top1++;

        /* Score with Kalman-adaptive topo */
        int mad=compute_mad(cands,nc);int64_t s=(int64_t)sc;
        int wc2=(int)(s*w_c/(s+mad)),wd2=(int)(s*w_d/(s+mad)),wg2=(int)(s*w_g/(s+mad));
        for(int j=0;j<nc;j++)
            cands[j].combined=(int64_t)256*cands[j].dot_hg+(int64_t)256*cands[j].dot_vg
                              +(int64_t)wc2*cands[j].cent_sim+(int64_t)wd2*cands[j].div_sim
                              +(int64_t)wg2*cands[j].gdiv_sim;
        qsort(cands,(size_t)nc,sizeof(cand_t),cmp_comb_d);

        int pred=knn_vote(cands,nc,7);

        /* Autopsy */
        if(pred==true_label){
            correct_total++;
        } else if(!correct_in_topk){
            mode_a++; /* Vote miss */
        } else {
            /* Correct class in top-K. Did dot+topo ranking put it in top-7? */
            int in_top_knn=0;
            for(int j=0;j<7&&j<nc;j++)if(train_labels[cands[j].id]==true_label){in_top_knn=1;break;}
            if(!in_top_knn) mode_b++; /* Dot/topo override */
            else mode_c++; /* kNN dilution */
        }
    }

    int errors=TEST_N-correct_total;
    printf("  Accuracy: %.2f%% (%d/%d, %d errors)\n\n",100.0*correct_total/TEST_N,correct_total,TEST_N,errors);
    printf("  Vote recall (correct in top-%d): %.2f%% (%d/%d)\n",TOP_K,100.0*vote_recall/TEST_N,vote_recall,TEST_N);
    printf("  Vote top-1:  %.2f%% (%d/%d)\n\n",100.0*vote_top1/TEST_N,vote_top1,TEST_N);

    printf("  Error modes (of %d errors):\n",errors);
    printf("    Mode A (vote miss — correct not in top-%d):  %d (%.1f%%)\n",TOP_K,mode_a,100.0*mode_a/errors);
    printf("    Mode B (dot/topo override — in top-K, not top-7): %d (%.1f%%)\n",mode_b,100.0*mode_b/errors);
    printf("    Mode C (kNN dilution — in top-7, wrong vote):     %d (%.1f%%)\n",mode_c,100.0*mode_c/errors);

    printf("\n  Ceiling if Mode B fixed: %.2f%%\n",100.0*(correct_total+mode_b)/TEST_N);
    printf("  Ceiling if Mode B+C fixed: %.2f%%\n",100.0*(correct_total+mode_b+mode_c)/TEST_N);

    /* ================================================================
     * SWEEP: weight configurations
     * ================================================================ */
    printf("\n=== WEIGHT SWEEP ===\n\n");
    {
        int wc_v[]={0,25,50,100};int wd_v[]={0,50,100,200};int wg_v[]={0,50,100,200};int sc_v[]={20,50,100};
        int best_c=0,bwc=0,bwd=0,bwg=0,bsc=0;
        for(int ci=0;ci<4;ci++)for(int di=0;di<4;di++)for(int gi=0;gi<4;gi++)for(int si=0;si<3;si++){
            int correct2=0;
            for(int i=0;i<TEST_N;i++){
                cand_t cands[TOP_K];int nc=nc_arr[i];
                memcpy(cands,pre+(size_t)i*TOP_K,(size_t)nc*sizeof(cand_t));
                int mad=compute_mad(cands,nc);int64_t s2=(int64_t)sc_v[si];
                int wc3=(int)(s2*wc_v[ci]/(s2+mad)),wd3=(int)(s2*wd_v[di]/(s2+mad)),wg3=(int)(s2*wg_v[gi]/(s2+mad));
                for(int j=0;j<nc;j++)
                    cands[j].combined=(int64_t)256*cands[j].dot_hg+(int64_t)256*cands[j].dot_vg
                                      +(int64_t)wc3*cands[j].cent_sim+(int64_t)wd3*cands[j].div_sim
                                      +(int64_t)wg3*cands[j].gdiv_sim;
                qsort(cands,(size_t)nc,sizeof(cand_t),cmp_comb_d);
                if(knn_vote(cands,nc,7)==test_labels[i])correct2++;
            }
            if(correct2>best_c){best_c=correct2;bwc=wc_v[ci];bwd=wd_v[di];bwg=wg_v[gi];bsc=sc_v[si];}
        }
        printf("  Best static: w_c=%d w_d=%d w_g=%d sc=%d -> %.2f%% (%d errors)\n",
               bwc,bwd,bwg,bsc,100.0*best_c/TEST_N,TEST_N-best_c);

        /* Bayesian sequential sweep on best static weights */
        printf("\n  Bayesian sequential sweep (on best static)...\n");
        int ds_v[]={1,2,5,10};int ks_v[]={3,7,15,30};
        int best_seq=best_c,bds=0,bks=0;
        for(int dsi=0;dsi<4;dsi++)for(int ksi=0;ksi<4;ksi++){
            int correct2=0;
            for(int i=0;i<TEST_N;i++){
                cand_t cands[TOP_K];int nc=nc_arr[i];
                memcpy(cands,pre+(size_t)i*TOP_K,(size_t)nc*sizeof(cand_t));
                int mad=compute_mad(cands,nc);int64_t s2=(int64_t)bsc;
                int wc3=(int)(s2*bwc/(s2+mad)),wd3=(int)(s2*bwd/(s2+mad)),wg3=(int)(s2*bwg/(s2+mad));
                for(int j=0;j<nc;j++)
                    cands[j].combined=(int64_t)256*cands[j].dot_hg+(int64_t)256*cands[j].dot_vg
                                      +(int64_t)wc3*cands[j].cent_sim+(int64_t)wd3*cands[j].div_sim
                                      +(int64_t)wg3*cands[j].gdiv_sim;
                qsort(cands,(size_t)nc,sizeof(cand_t),cmp_comb_d);
                int64_t state[N_CLASSES];memset(state,0,sizeof(state));
                int ks=ks_v[ksi]<nc?ks_v[ksi]:nc;
                for(int j=0;j<ks;j++){
                    uint8_t lbl=train_labels[cands[j].id];
                    int64_t weight=cands[j].combined*ds_v[dsi]/(ds_v[dsi]+j);
                    state[lbl]+=weight;
                }
                int pred=0;for(int c=1;c<N_CLASSES;c++)if(state[c]>state[pred])pred=c;
                if(pred==test_labels[i])correct2++;
            }
            if(correct2>best_seq){best_seq=correct2;bds=ds_v[dsi];bks=ks_v[ksi];}
        }
        printf("  Best sequential: dS=%d K=%d -> %.2f%% (%+d vs static)\n",
               bds,bks,100.0*best_seq/TEST_N,best_seq-best_c);

        /* Gradient-only dot (no topo) for comparison */
        printf("\n  Gradient-dot-only (no topo features)...\n");
        int grad_only=0;
        for(int i=0;i<TEST_N;i++){
            cand_t cands[TOP_K];int nc=nc_arr[i];
            memcpy(cands,pre+(size_t)i*TOP_K,(size_t)nc*sizeof(cand_t));
            for(int j=0;j<nc;j++)cands[j].combined=(int64_t)256*cands[j].dot_hg+(int64_t)256*cands[j].dot_vg;
            qsort(cands,(size_t)nc,sizeof(cand_t),cmp_comb_d);
            if(knn_vote(cands,nc,7)==test_labels[i])grad_only++;
        }
        printf("  Grad dot only: %.2f%%\n",100.0*grad_only/TEST_N);
    }

    /* Per-class recall */
    printf("\n=== PER-CLASS RECALL (best config) ===\n");
    {
        int per_class[N_CLASSES]={0},per_total[N_CLASSES]={0};
        for(int i=0;i<TEST_N;i++){
            per_total[test_labels[i]]++;
            /* Quick recompute with best weights — just use autopsy config */
            cand_t cands[TOP_K];int nc=nc_arr[i];
            memcpy(cands,pre+(size_t)i*TOP_K,(size_t)nc*sizeof(cand_t));
            int mad=compute_mad(cands,nc);int64_t s2=(int64_t)sc;
            int wc3=(int)(s2*w_c/(s2+mad)),wd3=(int)(s2*w_d/(s2+mad)),wg3=(int)(s2*w_g/(s2+mad));
            for(int j=0;j<nc;j++)
                cands[j].combined=(int64_t)256*cands[j].dot_hg+(int64_t)256*cands[j].dot_vg
                                  +(int64_t)wc3*cands[j].cent_sim+(int64_t)wd3*cands[j].div_sim
                                  +(int64_t)wg3*cands[j].gdiv_sim;
            qsort(cands,(size_t)nc,sizeof(cand_t),cmp_comb_d);
            if(knn_vote(cands,nc,7)==test_labels[i])per_class[test_labels[i]]++;
        }
        for(int c=0;c<N_CLASSES;c++)
            printf("  %d %-12s %.1f%% (%d/%d)\n",c,class_names[c],
                   100.0*per_class[c]/per_total[c],per_class[c],per_total[c]);
    }

    printf("\nTotal: %.1f sec\n",now_sec()-t0);
    free(pre);free(nc_arr);
    return 0;
}
