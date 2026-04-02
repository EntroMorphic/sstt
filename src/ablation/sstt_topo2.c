/*
 * sstt_topo2.c — Generalization Experiments: Divergence Dot, Histogram, Curl
 *
 * Follow-up to sstt_topo.c (97.11% MNIST, 84.24% Fashion).  Contribution 30
 * identified a tension between information and invariance:
 * the scalar divergence summary generalizes because it's position-invariant,
 * but it discards the spatial shape of the divergence field.
 *
 * Three experiments to extract more from the gradient field topology:
 *
 *   A. Divergence dot product — full 784-element ternary divergence field
 *      passed through the existing AVX2 dot.  Tests whether spatial structure
 *      generalizes or is domain-specific.
 *
 *   B. Divergence histogram — 9-bin count of divergence values {-4..+4}.
 *      Position-invariant, integer, counts-based.  The architecturally
 *      native middle ground between scalar and spatial.
 *
 *   C. Curl — discrete curl of the gradient field: rotational structure
 *      complementary to divergence's convergence/divergence structure.
 *      Scalar summary + histogram.
 *
 * All features compose with the sstt_topo.c baseline (cent + prof + div).
 *
 * Build: make sstt_topo2
 */

#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define TRAIN_N         60000
#define TEST_N          10000
#define IMG_W           28
#define IMG_H           28
#define PIXELS          784
#define PADDED          800
#define N_CLASSES       10

#define BLKS_PER_ROW    9
#define N_BLOCKS        252
#define SIG_PAD         256
#define BYTE_VALS       256

#define BG_PIXEL        0
#define BG_GRAD         13
#define BG_TRANS        13
#define H_TRANS_PER_ROW 8
#define N_HTRANS        (H_TRANS_PER_ROW * IMG_H)
#define V_TRANS_PER_COL 27
#define N_VTRANS        (BLKS_PER_ROW * V_TRANS_PER_COL)
#define TRANS_PAD       256

#define IG_SCALE        16
#define TOP_K           200
#define HP_PAD          32
#define DHIST_BINS      9       /* divergence values -4..+4 */
#define DHIST_PAD       16      /* pad to 16 for alignment */
#define FG_W            30
#define FG_H            30
#define FG_SZ           (FG_W * FG_H)

#define CENT_BOTH_OPEN  50
#define CENT_DISAGREE   80

static const char *data_dir = "data/";

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ================================================================
 *  Data
 * ================================================================ */
static uint8_t *raw_train_img, *raw_test_img;
static uint8_t *train_labels, *test_labels;
static int8_t  *tern_train, *tern_test;
static int8_t  *hgrad_train, *hgrad_test;
static int8_t  *vgrad_train, *vgrad_test;
static uint8_t *px_tr, *px_te, *hg_tr, *hg_te, *vg_tr, *vg_te;
static uint8_t *ht_tr, *ht_te, *vt_tr, *vt_te;
static uint8_t *joint_tr, *joint_te;

/* Bytepacked index */
static uint16_t  ig_w[N_BLOCKS];
static uint8_t   nbr[BYTE_VALS][8];
static uint32_t  idx_off[N_BLOCKS][BYTE_VALS];
static uint16_t  idx_sz [N_BLOCKS][BYTE_VALS];
static uint32_t *idx_pool;
static uint8_t   bg;
static int   *g_hist     = NULL;
static size_t g_hist_cap = 0;

/* --- Topo features from sstt_topo.c (carried forward) --- */
static int16_t *cent_train, *cent_test;
static int16_t *hprof_train, *hprof_test;
static int16_t *divneg_train, *divneg_test;
static int16_t *divneg_cy_train, *divneg_cy_test;

/* --- New features from contribution 30 --- */
static int8_t  *divtern_train, *divtern_test;     /* ternary divergence field [n][PADDED] */
static int16_t *divhist_train, *divhist_test;     /* divergence histogram [n][DHIST_PAD] */
static int16_t *curlneg_train, *curlneg_test;     /* curl neg sum */
static int16_t *curlhist_train, *curlhist_test;   /* curl histogram [n][DHIST_PAD] */

/* ================================================================
 *  Data loading + feature computation (from sstt_topo.c)
 * ================================================================ */
static uint8_t *load_idx(const char *path, uint32_t *cnt,
                          uint32_t *ro, uint32_t *co) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "ERR: %s\n", path); exit(1); }
    uint32_t m, n;
    if (fread(&m,4,1,f)!=1 || fread(&n,4,1,f)!=1) { fclose(f); exit(1); }
    m=__builtin_bswap32(m); n=__builtin_bswap32(n); *cnt=n;
    size_t s=1;
    if ((m&0xFF)>=3) {
        uint32_t r,c;
        if (fread(&r,4,1,f)!=1 || fread(&c,4,1,f)!=1) { fclose(f); exit(1); }
        r=__builtin_bswap32(r); c=__builtin_bswap32(c);
        if(ro)*ro=r; if(co)*co=c; s=(size_t)r*c;
    } else { if(ro)*ro=0; if(co)*co=0; }
    size_t total=(size_t)n*s;
    uint8_t *d=malloc(total);
    if (!d || fread(d,1,total,f)!=total) { fclose(f); exit(1); }
    fclose(f); return d;
}
static void load_data(void) {
    uint32_t n,r,c; char p[256];
    snprintf(p,sizeof(p),"%strain-images-idx3-ubyte",data_dir); raw_train_img=load_idx(p,&n,&r,&c);
    snprintf(p,sizeof(p),"%strain-labels-idx1-ubyte",data_dir); train_labels =load_idx(p,&n,NULL,NULL);
    snprintf(p,sizeof(p),"%st10k-images-idx3-ubyte", data_dir); raw_test_img =load_idx(p,&n,&r,&c);
    snprintf(p,sizeof(p),"%st10k-labels-idx1-ubyte", data_dir); test_labels  =load_idx(p,&n,NULL,NULL);
}

static inline int8_t clamp_trit(int v) { return v>0?1:v<0?-1:0; }

static void quant_tern(const uint8_t *src, int8_t *dst, int n) {
    const __m256i bias=_mm256_set1_epi8((char)0x80);
    const __m256i thi =_mm256_set1_epi8((char)(170^0x80));
    const __m256i tlo =_mm256_set1_epi8((char)(85^0x80));
    const __m256i one =_mm256_set1_epi8(1);
    for(int i=0;i<n;i++){
        const uint8_t*s=src+(size_t)i*PIXELS; int8_t*d=dst+(size_t)i*PADDED; int k;
        for(k=0;k+32<=PIXELS;k+=32){
            __m256i px=_mm256_loadu_si256((const __m256i*)(s+k));
            __m256i sp=_mm256_xor_si256(px,bias);
            _mm256_storeu_si256((__m256i*)(d+k),
                _mm256_sub_epi8(_mm256_and_si256(_mm256_cmpgt_epi8(sp,thi),one),
                                _mm256_and_si256(_mm256_cmpgt_epi8(tlo,sp),one)));
        }
        for(;k<PIXELS;k++) d[k]=s[k]>170?1:s[k]<85?-1:0;
        memset(d+PIXELS,0,PADDED-PIXELS);
    }
}

static void gradients(const int8_t *t,int8_t *h,int8_t *v,int n) {
    for(int i=0;i<n;i++){
        const int8_t*ti=t+(size_t)i*PADDED; int8_t*hi=h+(size_t)i*PADDED; int8_t*vi=v+(size_t)i*PADDED;
        for(int y=0;y<IMG_H;y++){
            for(int x=0;x<IMG_W-1;x++) hi[y*IMG_W+x]=clamp_trit(ti[y*IMG_W+x+1]-ti[y*IMG_W+x]);
            hi[y*IMG_W+IMG_W-1]=0;
        }
        memset(hi+PIXELS,0,PADDED-PIXELS);
        for(int y=0;y<IMG_H-1;y++) for(int x=0;x<IMG_W;x++) vi[y*IMG_W+x]=clamp_trit(ti[(y+1)*IMG_W+x]-ti[y*IMG_W+x]);
        memset(vi+(IMG_H-1)*IMG_W,0,IMG_W); memset(vi+PIXELS,0,PADDED-PIXELS);
    }
}

static inline uint8_t benc(int8_t a,int8_t b,int8_t c){return(uint8_t)((a+1)*9+(b+1)*3+(c+1));}
static void block_sigs(const int8_t*data,uint8_t*sigs,int n){
    for(int i=0;i<n;i++){
        const int8_t*img=data+(size_t)i*PADDED; uint8_t*sig=sigs+(size_t)i*SIG_PAD;
        for(int y=0;y<IMG_H;y++) for(int s=0;s<BLKS_PER_ROW;s++){int b=y*IMG_W+s*3;sig[y*BLKS_PER_ROW+s]=benc(img[b],img[b+1],img[b+2]);}
        memset(sig+N_BLOCKS,0xFF,SIG_PAD-N_BLOCKS);
    }
}
static inline uint8_t tenc(uint8_t a,uint8_t b){
    int8_t a0=(a/9)-1,a1=((a/3)%3)-1,a2=(a%3)-1,b0=(b/9)-1,b1=((b/3)%3)-1,b2=(b%3)-1;
    return benc(clamp_trit(b0-a0),clamp_trit(b1-a1),clamp_trit(b2-a2));
}
static void trans_fn(const uint8_t*bs,int str,uint8_t*ht,uint8_t*vt,int n){
    for(int i=0;i<n;i++){
        const uint8_t*s=bs+(size_t)i*str; uint8_t*h=ht+(size_t)i*TRANS_PAD; uint8_t*v=vt+(size_t)i*TRANS_PAD;
        for(int y=0;y<IMG_H;y++) for(int ss=0;ss<H_TRANS_PER_ROW;ss++) h[y*H_TRANS_PER_ROW+ss]=tenc(s[y*BLKS_PER_ROW+ss],s[y*BLKS_PER_ROW+ss+1]);
        memset(h+N_HTRANS,0xFF,TRANS_PAD-N_HTRANS);
        for(int y=0;y<V_TRANS_PER_COL;y++) for(int ss=0;ss<BLKS_PER_ROW;ss++) v[y*BLKS_PER_ROW+ss]=tenc(s[y*BLKS_PER_ROW+ss],s[(y+1)*BLKS_PER_ROW+ss]);
        memset(v+N_VTRANS,0xFF,TRANS_PAD-N_VTRANS);
    }
}
static uint8_t enc_d(uint8_t px,uint8_t hg,uint8_t vg,uint8_t ht,uint8_t vt){
    int ps=((px/9)-1)+(((px/3)%3)-1)+((px%3)-1);
    int hs=((hg/9)-1)+(((hg/3)%3)-1)+((hg%3)-1);
    int vs=((vg/9)-1)+(((vg/3)%3)-1)+((vg%3)-1);
    uint8_t pc=ps<0?0:ps==0?1:ps<3?2:3;
    uint8_t hc=hs<0?0:hs==0?1:hs<3?2:3;
    uint8_t vc=vs<0?0:vs==0?1:vs<3?2:3;
    return pc|(hc<<2)|(vc<<4)|((ht!=BG_TRANS)?1<<6:0)|((vt!=BG_TRANS)?1<<7:0);
}
static void joint_sigs_fn(uint8_t*out,int n,const uint8_t*px,const uint8_t*hg,const uint8_t*vg,const uint8_t*ht,const uint8_t*vt){
    for(int i=0;i<n;i++){
        const uint8_t*pi=px+(size_t)i*SIG_PAD,*hi=hg+(size_t)i*SIG_PAD,*vi=vg+(size_t)i*SIG_PAD;
        const uint8_t*hti=ht+(size_t)i*TRANS_PAD,*vti=vt+(size_t)i*TRANS_PAD; uint8_t*oi=out+(size_t)i*SIG_PAD;
        for(int y=0;y<IMG_H;y++) for(int s=0;s<BLKS_PER_ROW;s++){
            int k=y*BLKS_PER_ROW+s;
            uint8_t htb=s>0?hti[y*H_TRANS_PER_ROW+(s-1)]:BG_TRANS;
            uint8_t vtb=y>0?vti[(y-1)*BLKS_PER_ROW+s]:BG_TRANS;
            oi[k]=enc_d(pi[k],hi[k],vi[k],htb,vtb);
        }
        memset(oi+N_BLOCKS,0xFF,SIG_PAD-N_BLOCKS);
    }
}

/* IG + index */
static void compute_ig(const uint8_t *sigs, int n_vals, uint8_t bg_val, uint16_t *ig_out) {
    int cc[N_CLASSES]={0};
    for(int i=0;i<TRAIN_N;i++) cc[train_labels[i]]++;
    double hc=0;
    for(int c=0;c<N_CLASSES;c++){double p=(double)cc[c]/TRAIN_N;if(p>0)hc-=p*log2(p);}
    double raw[N_BLOCKS],mx=0;
    for(int k=0;k<N_BLOCKS;k++){
        int *cnt=calloc((size_t)n_vals*N_CLASSES,sizeof(int));
        int *vt =calloc(n_vals,sizeof(int));
        for(int i=0;i<TRAIN_N;i++){int v=sigs[(size_t)i*SIG_PAD+k];cnt[v*N_CLASSES+train_labels[i]]++;vt[v]++;}
        double hcond=0;
        for(int v=0;v<n_vals;v++){
            if(!vt[v]||v==(int)bg_val) continue;
            double pv=(double)vt[v]/TRAIN_N,hv=0;
            for(int c=0;c<N_CLASSES;c++){double pc=(double)cnt[v*N_CLASSES+c]/vt[v];if(pc>0)hv-=pc*log2(pc);}
            hcond+=pv*hv;
        }
        raw[k]=hc-hcond; if(raw[k]>mx) mx=raw[k];
        free(cnt); free(vt);
    }
    for(int k=0;k<N_BLOCKS;k++){ig_out[k]=mx>0?(uint16_t)(raw[k]/mx*IG_SCALE+0.5):1;if(!ig_out[k])ig_out[k]=1;}
}
static void build_index(void) {
    long vc[BYTE_VALS]={0};
    for(int i=0;i<TRAIN_N;i++){const uint8_t*s=joint_tr+(size_t)i*SIG_PAD;for(int k=0;k<N_BLOCKS;k++)vc[s[k]]++;}
    bg=0; long mc=0;
    for(int v=0;v<BYTE_VALS;v++) if(vc[v]>mc){mc=vc[v];bg=(uint8_t)v;}
    compute_ig(joint_tr,BYTE_VALS,bg,ig_w);
    for(int v=0;v<BYTE_VALS;v++) for(int b=0;b<8;b++) nbr[v][b]=(uint8_t)(v^(1<<b));
    memset(idx_sz,0,sizeof(idx_sz));
    for(int i=0;i<TRAIN_N;i++){const uint8_t*s=joint_tr+(size_t)i*SIG_PAD;for(int k=0;k<N_BLOCKS;k++)if(s[k]!=bg)idx_sz[k][s[k]]++;}
    uint32_t tot=0;
    for(int k=0;k<N_BLOCKS;k++) for(int v=0;v<BYTE_VALS;v++){idx_off[k][v]=tot;tot+=idx_sz[k][v];}
    idx_pool=malloc((size_t)tot*sizeof(uint32_t));
    uint32_t(*wp)[BYTE_VALS]=malloc((size_t)N_BLOCKS*BYTE_VALS*4);
    memcpy(wp,idx_off,(size_t)N_BLOCKS*BYTE_VALS*4);
    for(int i=0;i<TRAIN_N;i++){const uint8_t*s=joint_tr+(size_t)i*SIG_PAD;for(int k=0;k<N_BLOCKS;k++)if(s[k]!=bg)idx_pool[wp[k][s[k]]++]=(uint32_t)i;}
    free(wp);
    printf("  Index: %u entries (%.1f MB)\n",tot,(double)tot*4/1048576);
}

/* ================================================================
 *  Carried-forward topo features (centroid, profile, divergence scalar)
 * ================================================================ */
static int16_t enclosed_centroid(const int8_t *tern) {
    uint8_t grid[FG_SZ]; memset(grid,0,sizeof(grid));
    for(int y=0;y<IMG_H;y++) for(int x=0;x<IMG_W;x++) grid[(y+1)*FG_W+(x+1)]=(tern[y*IMG_W+x]>0)?1:0;
    uint8_t visited[FG_SZ]; memset(visited,0,sizeof(visited));
    int stack[FG_SZ]; int sp=0;
    for(int y=0;y<FG_H;y++) for(int x=0;x<FG_W;x++){
        if(y==0||y==FG_H-1||x==0||x==FG_W-1){int pos=y*FG_W+x;if(!grid[pos]&&!visited[pos]){visited[pos]=1;stack[sp++]=pos;}}
    }
    while(sp>0){int pos=stack[--sp];int py=pos/FG_W,px2=pos%FG_W;
        const int dx[]={0,0,1,-1},dy[]={1,-1,0,0};
        for(int d=0;d<4;d++){int ny=py+dy[d],nx=px2+dx[d];if(ny<0||ny>=FG_H||nx<0||nx>=FG_W)continue;
            int npos=ny*FG_W+nx;if(!visited[npos]&&!grid[npos]){visited[npos]=1;stack[sp++]=npos;}}}
    int sum_y=0,count=0;
    for(int y=1;y<FG_H-1;y++) for(int x=1;x<FG_W-1;x++){int pos=y*FG_W+x;if(!grid[pos]&&!visited[pos]){sum_y+=(y-1);count++;}}
    return count>0?(int16_t)(sum_y/count):-1;
}
static void compute_centroid_all(const int8_t *tern, int16_t *out, int n){for(int i=0;i<n;i++) out[i]=enclosed_centroid(tern+(size_t)i*PADDED);}
static void hprofile(const int8_t *tern, int16_t *prof){
    for(int y=0;y<IMG_H;y++){int c=0;for(int x=0;x<IMG_W;x++)c+=(tern[y*IMG_W+x]>0);prof[y]=(int16_t)c;}
    for(int y=IMG_H;y<HP_PAD;y++) prof[y]=0;
}
static void compute_hprof_all(const int8_t *tern, int16_t *out, int n){for(int i=0;i<n;i++) hprofile(tern+(size_t)i*PADDED,out+(size_t)i*HP_PAD);}
static void div_features(const int8_t *hg, const int8_t *vg, int16_t *ns, int16_t *cy){
    int neg_sum=0,neg_ysum=0,neg_cnt=0;
    for(int y=0;y<IMG_H;y++) for(int x=0;x<IMG_W;x++){
        int dh=(int)hg[y*IMG_W+x]-(x>0?(int)hg[y*IMG_W+x-1]:0);
        int dv=(int)vg[y*IMG_W+x]-(y>0?(int)vg[(y-1)*IMG_W+x]:0);
        int d=dh+dv; if(d<0){neg_sum+=d;neg_ysum+=y;neg_cnt++;}
    }
    *ns=(int16_t)(neg_sum<-32767?-32767:neg_sum);
    *cy=neg_cnt>0?(int16_t)(neg_ysum/neg_cnt):-1;
}
static void compute_div_all(const int8_t *hg, const int8_t *vg, int16_t *ns, int16_t *cy, int n){
    for(int i=0;i<n;i++) div_features(hg+(size_t)i*PADDED,vg+(size_t)i*PADDED,&ns[i],&cy[i]);
}

/* ================================================================
 *  NEW: Divergence dot product (Experiment A)
 *  Compute full divergence field, quantize to ternary, store for dot.
 * ================================================================ */
static void compute_divtern(const int8_t *hg, const int8_t *vg, int8_t *out, int n) {
    for(int i=0; i<n; i++){
        const int8_t *h = hg + (size_t)i*PADDED;
        const int8_t *v = vg + (size_t)i*PADDED;
        int8_t *d = out + (size_t)i*PADDED;
        for(int y=0; y<IMG_H; y++)
            for(int x=0; x<IMG_W; x++){
                int dh = (int)h[y*IMG_W+x] - (x>0 ? (int)h[y*IMG_W+x-1] : 0);
                int dv = (int)v[y*IMG_W+x] - (y>0 ? (int)v[(y-1)*IMG_W+x] : 0);
                d[y*IMG_W+x] = clamp_trit(dh + dv);
            }
        memset(d+PIXELS, 0, PADDED-PIXELS);
    }
}

/* ================================================================
 *  NEW: Divergence histogram (Experiment B)
 *  9 bins for values -4..+4.  Position-invariant.
 * ================================================================ */
static void compute_divhist(const int8_t *hg, const int8_t *vg, int16_t *out, int n) {
    for(int i=0; i<n; i++){
        const int8_t *h = hg + (size_t)i*PADDED;
        const int8_t *v = vg + (size_t)i*PADDED;
        int16_t *hist = out + (size_t)i*DHIST_PAD;
        memset(hist, 0, DHIST_PAD*sizeof(int16_t));
        for(int y=0; y<IMG_H; y++)
            for(int x=0; x<IMG_W; x++){
                int dh = (int)h[y*IMG_W+x] - (x>0 ? (int)h[y*IMG_W+x-1] : 0);
                int dv = (int)v[y*IMG_W+x] - (y>0 ? (int)v[(y-1)*IMG_W+x] : 0);
                int d = dh + dv;
                if(d >= -4 && d <= 4) hist[d+4]++;
            }
    }
}

/* ================================================================
 *  NEW: Curl of gradient field (Experiment C)
 *  curl(x,y) = dvg/dx - dhg/dy
 *  Scalar summary + histogram.
 * ================================================================ */
static void compute_curl(const int8_t *hg, const int8_t *vg,
                          int16_t *neg_sum_out, int16_t *hist_out, int n) {
    for(int i=0; i<n; i++){
        const int8_t *h = hg + (size_t)i*PADDED;
        const int8_t *v = vg + (size_t)i*PADDED;
        int16_t *hist = hist_out + (size_t)i*DHIST_PAD;
        memset(hist, 0, DHIST_PAD*sizeof(int16_t));
        int neg_sum = 0;
        for(int y=0; y<IMG_H; y++)
            for(int x=0; x<IMG_W; x++){
                /* curl = dvg/dx - dhg/dy */
                int dvgdx = (x<IMG_W-1 ? (int)v[y*IMG_W+x+1] : 0) - (int)v[y*IMG_W+x];
                int dhgdy = (y<IMG_H-1 ? (int)h[(y+1)*IMG_W+x] : 0) - (int)h[y*IMG_W+x];
                int c = dvgdx - dhgdy;
                if(c < 0) neg_sum += c;
                if(c >= -4 && c <= 4) hist[c+4]++;
            }
        neg_sum_out[i] = (int16_t)(neg_sum < -32767 ? -32767 : neg_sum);
    }
}

/* ================================================================
 *  AVX2 ternary dot
 * ================================================================ */
static inline int32_t tdot(const int8_t*a,const int8_t*b){
    __m256i acc=_mm256_setzero_si256();
    for(int i=0;i<PADDED;i+=32) acc=_mm256_add_epi8(acc,_mm256_sign_epi8(_mm256_load_si256((const __m256i*)(a+i)),_mm256_load_si256((const __m256i*)(b+i))));
    __m256i lo=_mm256_cvtepi8_epi16(_mm256_castsi256_si128(acc));
    __m256i hi=_mm256_cvtepi8_epi16(_mm256_extracti128_si256(acc,1));
    __m256i s32=_mm256_madd_epi16(_mm256_add_epi16(lo,hi),_mm256_set1_epi16(1));
    __m128i s=_mm_add_epi32(_mm256_castsi256_si128(s32),_mm256_extracti128_si256(s32,1));
    s=_mm_hadd_epi32(s,s); s=_mm_hadd_epi32(s,s); return _mm_cvtsi128_si32(s);
}

/* ================================================================
 *  Candidate machinery
 * ================================================================ */
typedef struct {
    uint32_t id, votes;
    int32_t  dot_px, dot_vg;           /* existing */
    int32_t  cent_sim, prof_sim;       /* from topo v1 */
    int32_t  div_sim;                  /* divergence scalar (from topo v1) */
    int32_t  divdot_sim;              /* NEW: divergence dot product */
    int32_t  divhist_sim;            /* NEW: divergence histogram L1 */
    int32_t  curl_sim;               /* NEW: curl scalar */
    int32_t  curlhist_sim;          /* NEW: curl histogram L1 */
    int64_t  combined;
} cand_t;

static int cmp_votes_d(const void*a,const void*b){return(int)((const cand_t*)b)->votes-(int)((const cand_t*)a)->votes;}
static int cmp_comb_d (const void*a,const void*b){int64_t da=((const cand_t*)a)->combined,db=((const cand_t*)b)->combined;return(db>da)-(db<da);}

static int select_top_k(const uint32_t*votes,int n,cand_t*out,int k){
    uint32_t mx=0; for(int i=0;i<n;i++) if(votes[i]>mx) mx=votes[i];
    if(!mx) return 0;
    if((size_t)(mx+1)>g_hist_cap){g_hist_cap=(size_t)(mx+1)+4096;free(g_hist);g_hist=malloc(g_hist_cap*sizeof(int));}
    memset(g_hist,0,(mx+1)*sizeof(int));
    for(int i=0;i<n;i++) if(votes[i]) g_hist[votes[i]]++;
    int cum=0,thr; for(thr=(int)mx;thr>=1;thr--){cum+=g_hist[thr];if(cum>=k)break;}
    if(thr<1)thr=1;
    int nc=0;
    for(int i=0;i<n&&nc<k;i++) if(votes[i]>=(uint32_t)thr){
        out[nc]=(cand_t){0}; out[nc].id=(uint32_t)i; out[nc].votes=votes[i]; nc++;
    }
    qsort(out,(size_t)nc,sizeof(cand_t),cmp_votes_d); return nc;
}

static void compute_features(cand_t *cands, int nc, int ti,
                              int16_t q_cent, const int16_t *q_prof,
                              int16_t q_divneg, int16_t q_divneg_cy,
                              const int16_t *q_divhist,
                              const int16_t *q_curlhist,
                              int16_t q_curlneg) {
    const int8_t *tp = tern_test  + (size_t)ti*PADDED;
    const int8_t *tv = vgrad_test + (size_t)ti*PADDED;
    const int8_t *qd = divtern_test + (size_t)ti*PADDED;

    for(int j=0; j<nc; j++){
        uint32_t id = cands[j].id;

        /* Dot products */
        cands[j].dot_px = tdot(tp, tern_train + (size_t)id*PADDED);
        cands[j].dot_vg = tdot(tv, vgrad_train + (size_t)id*PADDED);

        /* Centroid */
        int16_t c_cent = cent_train[id];
        if(q_cent<0 && c_cent<0) cands[j].cent_sim = CENT_BOTH_OPEN;
        else if(q_cent<0 || c_cent<0) cands[j].cent_sim = -CENT_DISAGREE;
        else cands[j].cent_sim = -(int32_t)abs(q_cent - c_cent);

        /* Profile dot */
        const int16_t *c_prof = hprof_train + (size_t)id*HP_PAD;
        int32_t pdot=0; for(int y=0;y<IMG_H;y++) pdot+=(int32_t)q_prof[y]*c_prof[y];
        cands[j].prof_sim = pdot;

        /* Divergence scalar */
        int16_t c_dn = divneg_train[id], c_dc = divneg_cy_train[id];
        int32_t ds = -(int32_t)abs(q_divneg - c_dn);
        if(q_divneg_cy>=0 && c_dc>=0) ds -= (int32_t)abs(q_divneg_cy - c_dc)*2;
        else if((q_divneg_cy<0) != (c_dc<0)) ds -= 10;
        cands[j].div_sim = ds;

        /* NEW: Divergence dot product */
        cands[j].divdot_sim = tdot(qd, divtern_train + (size_t)id*PADDED);

        /* NEW: Divergence histogram L1 distance (negated) */
        const int16_t *c_dh = divhist_train + (size_t)id*DHIST_PAD;
        int32_t hl=0; for(int b=0;b<DHIST_BINS;b++) hl += abs(q_divhist[b]-c_dh[b]);
        cands[j].divhist_sim = -hl;

        /* NEW: Curl scalar */
        cands[j].curl_sim = -(int32_t)abs(q_curlneg - curlneg_train[id]);

        /* NEW: Curl histogram L1 */
        const int16_t *c_ch = curlhist_train + (size_t)id*DHIST_PAD;
        int32_t cl=0; for(int b=0;b<DHIST_BINS;b++) cl += abs(q_curlhist[b]-c_ch[b]);
        cands[j].curlhist_sim = -cl;
    }
}

static int knn_vote(const cand_t *c, int nc, int k){
    int v[N_CLASSES]={0}; if(k>nc)k=nc;
    for(int i=0;i<k;i++) v[train_labels[c[i].id]]++;
    int best=0; for(int c2=1;c2<N_CLASSES;c2++) if(v[c2]>v[best]) best=c2;
    return best;
}

/* Vote */
static void vote(uint32_t *votes, int img){
    memset(votes,0,TRAIN_N*sizeof(uint32_t));
    const uint8_t*sig=joint_te+(size_t)img*SIG_PAD;
    for(int k=0;k<N_BLOCKS;k++){
        uint8_t bv=sig[k]; if(bv==bg) continue;
        uint16_t w=ig_w[k],wh=w>1?w/2:1;
        {uint32_t off=idx_off[k][bv];uint16_t sz=idx_sz[k][bv];const uint32_t*ids=idx_pool+off;for(uint16_t j=0;j<sz;j++)votes[ids[j]]+=w;}
        for(int nb=0;nb<8;nb++){uint8_t nv=nbr[bv][nb];if(nv==bg)continue;uint32_t noff=idx_off[k][nv];uint16_t nsz=idx_sz[k][nv];const uint32_t*nids=idx_pool+noff;for(uint16_t j=0;j<nsz;j++)votes[nids[j]]+=wh;}
    }
}

/* ================================================================
 *  Weight sweep engine
 * ================================================================ */
typedef struct {
    int w_px, w_vg, w_cent, w_prof, w_div, w_divdot, w_divhist, w_curl, w_curlhist;
} weights_t;

static int run_config(const cand_t *pre, const int *nc_arr, weights_t w, uint8_t *preds_out){
    int correct=0;
    for(int i=0;i<TEST_N;i++){
        cand_t cands[TOP_K]; int nc=nc_arr[i];
        memcpy(cands, pre+(size_t)i*TOP_K, (size_t)nc*sizeof(cand_t));
        for(int j=0;j<nc;j++)
            cands[j].combined = (int64_t)w.w_px       * cands[j].dot_px
                              + (int64_t)w.w_vg       * cands[j].dot_vg
                              + (int64_t)w.w_cent      * cands[j].cent_sim
                              + (int64_t)w.w_prof      * cands[j].prof_sim
                              + (int64_t)w.w_div       * cands[j].div_sim
                              + (int64_t)w.w_divdot    * cands[j].divdot_sim
                              + (int64_t)w.w_divhist   * cands[j].divhist_sim
                              + (int64_t)w.w_curl      * cands[j].curl_sim
                              + (int64_t)w.w_curlhist  * cands[j].curlhist_sim;
        qsort(cands,(size_t)nc,sizeof(cand_t),cmp_comb_d);
        int pred=knn_vote(cands,nc,3);
        if(preds_out) preds_out[i]=(uint8_t)pred;
        if(pred==test_labels[i]) correct++;
    }
    return correct;
}

/* Reporting */
static void report(const uint8_t *preds, const char *label, int correct){
    printf("--- %s ---\n", label);
    printf("  Accuracy: %.2f%%  (%d errors)\n\n", 100.0*correct/TEST_N, TEST_N-correct);
    int conf[N_CLASSES][N_CLASSES]; memset(conf,0,sizeof(conf));
    for(int i=0;i<TEST_N;i++) conf[test_labels[i]][preds[i]]++;
    printf("       ");for(int c=0;c<N_CLASSES;c++)printf(" %4d",c);
    printf("   | Recall\n  -----");for(int c=0;c<N_CLASSES;c++)(void)c,printf("-----");printf("---+-------\n");
    for(int r=0;r<N_CLASSES;r++){
        printf("    %d: ",r);int rt=0;
        for(int c=0;c<N_CLASSES;c++){printf(" %4d",conf[r][c]);rt+=conf[r][c];}
        printf("   | %5.1f%%\n",rt>0?100.0*conf[r][r]/rt:0.0);
    }
    typedef struct{int a,b,count;}pair_t;
    pair_t pairs[45];int np=0;
    for(int a=0;a<N_CLASSES;a++)for(int b=a+1;b<N_CLASSES;b++)pairs[np++]=(pair_t){a,b,conf[a][b]+conf[b][a]};
    for(int i=0;i<np-1;i++)for(int j=i+1;j<np;j++)if(pairs[j].count>pairs[i].count){pair_t t2=pairs[i];pairs[i]=pairs[j];pairs[j]=t2;}
    printf("  Top-5 confused:");for(int i=0;i<5&&i<np;i++)printf("  %d<->%d:%d",pairs[i].a,pairs[i].b,pairs[i].count);
    printf("\n\n");
}

/* ================================================================
 *  Main
 * ================================================================ */
int main(int argc, char **argv) {
    double t0 = now_sec();
    if(argc>1){data_dir=argv[1];size_t l=strlen(data_dir);if(l&&data_dir[l-1]!='/'){char*buf=malloc(l+2);memcpy(buf,data_dir,l);buf[l]='/';buf[l+1]='\0';data_dir=buf;}}
    const char *ds = strstr(data_dir,"fashion") ? "Fashion-MNIST" : "MNIST";
    printf("=== SSTT Topo2: Generalization Experiments (%s) ===\n\n", ds);

    load_data();
    tern_train=aligned_alloc(32,(size_t)TRAIN_N*PADDED); tern_test=aligned_alloc(32,(size_t)TEST_N*PADDED);
    hgrad_train=aligned_alloc(32,(size_t)TRAIN_N*PADDED); hgrad_test=aligned_alloc(32,(size_t)TEST_N*PADDED);
    vgrad_train=aligned_alloc(32,(size_t)TRAIN_N*PADDED); vgrad_test=aligned_alloc(32,(size_t)TEST_N*PADDED);
    quant_tern(raw_train_img,tern_train,TRAIN_N); quant_tern(raw_test_img,tern_test,TEST_N);
    gradients(tern_train,hgrad_train,vgrad_train,TRAIN_N); gradients(tern_test,hgrad_test,vgrad_test,TEST_N);

    px_tr=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD); px_te=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    hg_tr=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD); hg_te=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    vg_tr=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD); vg_te=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    block_sigs(tern_train,px_tr,TRAIN_N); block_sigs(tern_test,px_te,TEST_N);
    block_sigs(hgrad_train,hg_tr,TRAIN_N); block_sigs(hgrad_test,hg_te,TEST_N);
    block_sigs(vgrad_train,vg_tr,TRAIN_N); block_sigs(vgrad_test,vg_te,TEST_N);
    ht_tr=aligned_alloc(32,(size_t)TRAIN_N*TRANS_PAD); ht_te=aligned_alloc(32,(size_t)TEST_N*TRANS_PAD);
    vt_tr=aligned_alloc(32,(size_t)TRAIN_N*TRANS_PAD); vt_te=aligned_alloc(32,(size_t)TEST_N*TRANS_PAD);
    trans_fn(px_tr,SIG_PAD,ht_tr,vt_tr,TRAIN_N); trans_fn(px_te,SIG_PAD,ht_te,vt_te,TEST_N);
    joint_tr=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD); joint_te=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    joint_sigs_fn(joint_tr,TRAIN_N,px_tr,hg_tr,vg_tr,ht_tr,vt_tr);
    joint_sigs_fn(joint_te,TEST_N,px_te,hg_te,vg_te,ht_te,vt_te);
    printf("Building index...\n"); build_index();

    /* Carried-forward features */
    printf("Computing features...\n"); double t_feat=now_sec();
    cent_train=malloc((size_t)TRAIN_N*sizeof(int16_t)); cent_test=malloc((size_t)TEST_N*sizeof(int16_t));
    compute_centroid_all(tern_train,cent_train,TRAIN_N); compute_centroid_all(tern_test,cent_test,TEST_N);
    hprof_train=aligned_alloc(32,(size_t)TRAIN_N*HP_PAD*sizeof(int16_t));
    hprof_test=aligned_alloc(32,(size_t)TEST_N*HP_PAD*sizeof(int16_t));
    compute_hprof_all(tern_train,hprof_train,TRAIN_N); compute_hprof_all(tern_test,hprof_test,TEST_N);
    divneg_train=malloc((size_t)TRAIN_N*sizeof(int16_t)); divneg_test=malloc((size_t)TEST_N*sizeof(int16_t));
    divneg_cy_train=malloc((size_t)TRAIN_N*sizeof(int16_t)); divneg_cy_test=malloc((size_t)TEST_N*sizeof(int16_t));
    compute_div_all(hgrad_train,vgrad_train,divneg_train,divneg_cy_train,TRAIN_N);
    compute_div_all(hgrad_test,vgrad_test,divneg_test,divneg_cy_test,TEST_N);

    /* NEW features */
    divtern_train=aligned_alloc(32,(size_t)TRAIN_N*PADDED); divtern_test=aligned_alloc(32,(size_t)TEST_N*PADDED);
    compute_divtern(hgrad_train,vgrad_train,divtern_train,TRAIN_N);
    compute_divtern(hgrad_test,vgrad_test,divtern_test,TEST_N);
    printf("  Divergence ternary field: done\n");

    divhist_train=aligned_alloc(32,(size_t)TRAIN_N*DHIST_PAD*sizeof(int16_t));
    divhist_test=aligned_alloc(32,(size_t)TEST_N*DHIST_PAD*sizeof(int16_t));
    compute_divhist(hgrad_train,vgrad_train,divhist_train,TRAIN_N);
    compute_divhist(hgrad_test,vgrad_test,divhist_test,TEST_N);
    printf("  Divergence histogram: done\n");

    curlneg_train=malloc((size_t)TRAIN_N*sizeof(int16_t)); curlneg_test=malloc((size_t)TEST_N*sizeof(int16_t));
    curlhist_train=aligned_alloc(32,(size_t)TRAIN_N*DHIST_PAD*sizeof(int16_t));
    curlhist_test=aligned_alloc(32,(size_t)TEST_N*DHIST_PAD*sizeof(int16_t));
    compute_curl(hgrad_train,vgrad_train,curlneg_train,curlhist_train,TRAIN_N);
    compute_curl(hgrad_test,vgrad_test,curlneg_test,curlhist_test,TEST_N);
    printf("  Curl: done\n");
    printf("  Features: %.2f sec\n\n", now_sec()-t_feat);

    /* Precompute */
    printf("Precomputing votes + features...\n"); double t_pre=now_sec();
    cand_t *pre = malloc((size_t)TEST_N*TOP_K*sizeof(cand_t));
    int *nc_arr = malloc((size_t)TEST_N*sizeof(int));
    uint32_t *votes = calloc(TRAIN_N,sizeof(uint32_t));
    for(int i=0;i<TEST_N;i++){
        vote(votes,i);
        cand_t *ci=pre+(size_t)i*TOP_K;
        int nc=select_top_k(votes,TRAIN_N,ci,TOP_K);
        compute_features(ci,nc,i, cent_test[i], hprof_test+(size_t)i*HP_PAD,
                         divneg_test[i], divneg_cy_test[i],
                         divhist_test+(size_t)i*DHIST_PAD,
                         curlhist_test+(size_t)i*DHIST_PAD,
                         curlneg_test[i]);
        nc_arr[i]=nc;
        if((i+1)%2000==0) fprintf(stderr,"  precompute: %d/%d\r",i+1,TEST_N);
    }
    fprintf(stderr,"\n"); free(votes);
    printf("  Precompute: %.2f sec\n\n", now_sec()-t_pre);

    uint8_t *preds = calloc(TEST_N,1);

    /* Baseline weights from topo v1 */
    weights_t base = {256, 192, 0, 0, 0, 0, 0, 0, 0};
    weights_t topo1 = {256, 192, 50, 8, 200, 0, 0, 0, 0};

    int cBase = run_config(pre, nc_arr, base, preds);
    report(preds, "A: Dot baseline (256*px + 192*vg)", cBase);

    int cTopo1 = run_config(pre, nc_arr, topo1, preds);
    report(preds, "B: Topo v1 best (cent=50 prof=8 div=200)", cTopo1);

    /* Sweep each new feature individually on top of topo v1 */
    printf("--- C: Divergence dot sweep (on top of topo v1) ---\n");
    { weights_t w=topo1; int best_w=0,best_c=cTopo1;
      int ws[]={0,4,8,16,32,64,128}; int nw=7;
      for(int wi=0;wi<nw;wi++){w.w_divdot=ws[wi];int c=run_config(pre,nc_arr,w,NULL);
        printf("  w_divdot=%3d: %.2f%% (%d errors, %+d vs topo1)\n",ws[wi],100.0*c/TEST_N,TEST_N-c,c-cTopo1);
        if(c>best_c){best_c=c;best_w=ws[wi];}}
      printf("  Best: w_divdot=%d -> %.2f%%\n\n",best_w,100.0*best_c/TEST_N);
    }

    printf("--- D: Divergence histogram sweep (on top of topo v1) ---\n");
    { weights_t w=topo1; int best_w=0,best_c=cTopo1;
      int ws[]={0,1,2,4,8,16,32}; int nw=7;
      for(int wi=0;wi<nw;wi++){w.w_divhist=ws[wi];int c=run_config(pre,nc_arr,w,NULL);
        printf("  w_divhist=%3d: %.2f%% (%d errors, %+d vs topo1)\n",ws[wi],100.0*c/TEST_N,TEST_N-c,c-cTopo1);
        if(c>best_c){best_c=c;best_w=ws[wi];}}
      printf("  Best: w_divhist=%d -> %.2f%%\n\n",best_w,100.0*best_c/TEST_N);
    }

    printf("--- E: Curl scalar sweep (on top of topo v1) ---\n");
    { weights_t w=topo1; int best_w=0,best_c=cTopo1;
      int ws[]={0,25,50,100,200,400}; int nw=6;
      for(int wi=0;wi<nw;wi++){w.w_curl=ws[wi];int c=run_config(pre,nc_arr,w,NULL);
        printf("  w_curl=%4d: %.2f%% (%d errors, %+d vs topo1)\n",ws[wi],100.0*c/TEST_N,TEST_N-c,c-cTopo1);
        if(c>best_c){best_c=c;best_w=ws[wi];}}
      printf("  Best: w_curl=%d -> %.2f%%\n\n",best_w,100.0*best_c/TEST_N);
    }

    printf("--- E2: Curl histogram sweep (on top of topo v1) ---\n");
    { weights_t w=topo1; int best_w=0,best_c=cTopo1;
      int ws[]={0,1,2,4,8,16,32}; int nw=7;
      for(int wi=0;wi<nw;wi++){w.w_curlhist=ws[wi];int c=run_config(pre,nc_arr,w,NULL);
        printf("  w_curlhist=%3d: %.2f%% (%d errors, %+d vs topo1)\n",ws[wi],100.0*c/TEST_N,TEST_N-c,c-cTopo1);
        if(c>best_c){best_c=c;best_w=ws[wi];}}
      printf("  Best: w_curlhist=%d -> %.2f%%\n\n",best_w,100.0*best_c/TEST_N);
    }

    /* Combined grid: topo1 base + new features */
    printf("--- F: Combined grid search (topo1 + new features) ---\n");
    { int dd_vals[]={0,8,16,32,64};
      int dh_vals[]={0,2,4,8};
      int cu_vals[]={0,50,100,200};
      int ch_vals[]={0,2,4,8};
      int ndd=5,ndh=4,ncu=4,nch=4;
      weights_t best_w=topo1; int best_correct=cTopo1;
      for(int a=0;a<ndd;a++) for(int b=0;b<ndh;b++) for(int c=0;c<ncu;c++) for(int d=0;d<nch;d++){
          if(!dd_vals[a]&&!dh_vals[b]&&!cu_vals[c]&&!ch_vals[d]) continue;
          weights_t w=topo1; w.w_divdot=dd_vals[a]; w.w_divhist=dh_vals[b]; w.w_curl=cu_vals[c]; w.w_curlhist=ch_vals[d];
          int r=run_config(pre,nc_arr,w,NULL);
          if(r>best_correct){best_correct=r;best_w=w;}
      }
      printf("  Best: divdot=%d divhist=%d curl=%d curlhist=%d -> %.2f%% (%d errors, %+d vs topo1)\n\n",
             best_w.w_divdot,best_w.w_divhist,best_w.w_curl,best_w.w_curlhist,
             100.0*best_correct/TEST_N,TEST_N-best_correct,best_correct-cTopo1);

      int cF=run_config(pre,nc_arr,best_w,preds);
      char label[256];
      snprintf(label,sizeof(label),"G: Best combined (c=%d p=%d div=%d dd=%d dh=%d cu=%d ch=%d)",
               best_w.w_cent,best_w.w_prof,best_w.w_div,best_w.w_divdot,best_w.w_divhist,best_w.w_curl,best_w.w_curlhist);
      report(preds,label,cF);

      /* Per-pair delta vs topo1 */
      uint8_t *preds_t1=calloc(TEST_N,1);
      run_config(pre,nc_arr,topo1,preds_t1);
      printf("  Per-pair delta vs topo v1:\n");
      int pairs[][2]={{3,8},{4,9},{1,7},{3,5},{7,9}};
      for(int pi=0;pi<5;pi++){
          int a=pairs[pi][0],b=pairs[pi][1],e1=0,e2=0;
          for(int i=0;i<TEST_N;i++){int tl=test_labels[i];
              if(tl!=a&&tl!=b)continue;
              if(preds_t1[i]!=tl&&(preds_t1[i]==a||preds_t1[i]==b))e1++;
              if(preds[i]!=tl&&(preds[i]==a||preds[i]==b))e2++;}
          printf("    %d<->%d: %d -> %d (%+d)\n",a,b,e1,e2,e2-e1);
      }
      free(preds_t1);

      printf("\n=== SUMMARY ===\n");
      printf("  Dot baseline:   %.2f%%  (%d errors)\n",100.0*cBase/TEST_N,TEST_N-cBase);
      printf("  Topo v1:        %.2f%%  (%d errors)\n",100.0*cTopo1/TEST_N,TEST_N-cTopo1);
      printf("  Best combined:  %.2f%%  (%d errors, %+d vs topo1, %+d vs baseline)\n",
             100.0*cF/TEST_N,TEST_N-cF,cF-cTopo1,cF-cBase);
      printf("  Weights: px=%d vg=%d cent=%d prof=%d div=%d divdot=%d divhist=%d curl=%d curlhist=%d\n",
             best_w.w_px,best_w.w_vg,best_w.w_cent,best_w.w_prof,best_w.w_div,
             best_w.w_divdot,best_w.w_divhist,best_w.w_curl,best_w.w_curlhist);
    }

    printf("\nTotal: %.2f sec\n", now_sec()-t0);

    free(pre);free(nc_arr);free(preds);
    free(cent_train);free(cent_test);free(hprof_train);free(hprof_test);
    free(divneg_train);free(divneg_test);free(divneg_cy_train);free(divneg_cy_test);
    free(divtern_train);free(divtern_test);free(divhist_train);free(divhist_test);
    free(curlneg_train);free(curlneg_test);free(curlhist_train);free(curlhist_test);
    free(idx_pool);free(joint_tr);free(joint_te);
    free(px_tr);free(px_te);free(hg_tr);free(hg_te);free(vg_tr);free(vg_te);
    free(ht_tr);free(ht_te);free(vt_tr);free(vt_te);
    free(tern_train);free(tern_test);free(hgrad_train);free(hgrad_test);
    free(vgrad_train);free(vgrad_test);
    free(raw_train_img);free(raw_test_img);free(train_labels);free(test_labels);
    return 0;
}
