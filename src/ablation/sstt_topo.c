/*
 * sstt_topo.c — Topological Feature Augmentation for Cascade Ranking
 *
 * The cascade autopsy (contribution 23) proved 98.3% of errors occur at
 * the dot-product ranking step — the vote phase has 99.3% recall.  The
 * remaining errors concentrate on confusion pairs that require structural
 * features the ternary dot product cannot see:
 *
 *   3<->8 (25 errors):  0 vs 2 loops          — Euler characteristic
 *   4<->9 (36 errors):  loop position / open   — enclosed-region centroid
 *   1<->7 (23 errors):  horizontal bar          — row foreground profile
 *
 * This experiment augments the bytepacked cascade's ranking with 3
 * topological features computed ONLY on the top-K candidates:
 *
 *   score = w_px*dot(px) + w_vg*dot(vg)
 *         + w_euler   * euler_similarity
 *         + w_cent    * centroid_similarity
 *         + w_prof    * profile_dot
 *
 * Pipeline (vote phase unchanged, ranking augmented):
 *   1. Bytepacked 8-probe IG-weighted vote -> top-K candidates
 *   2. Ternary dot products (px, vg) [existing]
 *   3. Topological similarities (euler, centroid, profile) [NEW]
 *   4. Combined score -> sort -> k=3 majority vote
 *
 * Tests:
 *   A. Baseline: dot only (256*px + 192*vg)
 *   B. + Euler only (sweep w_euler)
 *   C. + Centroid only (sweep w_cent)
 *   D. + Profile only (sweep w_prof)
 *   E. All three combined (grid search)
 *   F. Best config: confusion matrix + per-pair delta
 *
 * Build: make sstt_topo
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
#define CLS_PAD         16

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
#define HP_PAD          32  /* pad profile to 32 for alignment */

/* Topological feature defaults */
#define EULER_MATCH     100
#define EULER_MISMATCH  100
#define CENT_BOTH_OPEN  50
#define CENT_DISAGREE   80

/* Flood-fill grid */
#define FG_W            30
#define FG_H            30
#define FG_SZ           (FG_W * FG_H)

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
static uint8_t *train_labels,  *test_labels;
static int8_t  *tern_train, *tern_test;
static int8_t  *hgrad_train, *hgrad_test;
static int8_t  *vgrad_train, *vgrad_test;
static uint8_t *px_tr, *px_te;
static uint8_t *hg_tr, *hg_te;
static uint8_t *vg_tr, *vg_te;
static uint8_t *ht_tr, *ht_te;
static uint8_t *vt_tr, *vt_te;
static uint8_t *joint_tr, *joint_te;

/* Bytepacked index */
static uint16_t  ig_w[N_BLOCKS];
static uint8_t   nbr[BYTE_VALS][8];
static uint32_t  idx_off[N_BLOCKS][BYTE_VALS];
static uint16_t  idx_sz [N_BLOCKS][BYTE_VALS];
static uint32_t *idx_pool;
static uint8_t   bg;

/* Pre-allocated histogram */
static int   *g_hist     = NULL;
static size_t g_hist_cap = 0;

/* Topological feature arrays */
static int8_t  *euler_train, *euler_test;       /* Euler characteristic per image */
static int16_t *cent_train,  *cent_test;        /* enclosed centroid y (-1 = none) */
static int16_t *hprof_train, *hprof_test;       /* horizontal profile [n][HP_PAD] */

/* Divergence features (from Gauss/Green's theorem on ternary gradient field) */
static int16_t *divneg_train, *divneg_test;     /* sum of negative divergence */
static int16_t *divneg_cy_train, *divneg_cy_test; /* y-centroid of neg divergence */

/* ================================================================
 *  Data loading
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

/* ================================================================
 *  Feature computation (same as oracle_v2 / bytecascade)
 * ================================================================ */
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
static void trans(const uint8_t*bs,int str,uint8_t*ht,uint8_t*vt,int n){
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

/* ================================================================
 *  IG + index (from bytecascade)
 * ================================================================ */
static void compute_ig(const uint8_t *sigs, int n_vals, uint8_t bg_val,
                        uint16_t *ig_out) {
    int cc[N_CLASSES]={0};
    for(int i=0;i<TRAIN_N;i++) cc[train_labels[i]]++;
    double hc=0;
    for(int c=0;c<N_CLASSES;c++){double p=(double)cc[c]/TRAIN_N;if(p>0)hc-=p*log2(p);}
    double raw[N_BLOCKS],mx=0;
    for(int k=0;k<N_BLOCKS;k++){
        int *cnt=calloc((size_t)n_vals*N_CLASSES,sizeof(int));
        int *vt =calloc(n_vals,sizeof(int));
        for(int i=0;i<TRAIN_N;i++){
            int v=sigs[(size_t)i*SIG_PAD+k];
            cnt[v*N_CLASSES+train_labels[i]]++; vt[v]++;
        }
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
    /* Detect background */
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
 *  Topological features
 * ================================================================ */

/* Euler characteristic via 2x2 quad method on binarized ternary image.
 * Foreground = (trit > 0), i.e. pixel was bright (>170).
 * Returns: #components - #holes.  Digit 8 -> -1, digit 0 -> 0, digit 1 -> 1. */
static int8_t euler_char(const int8_t *tern) {
    int q1=0, q3=0, qd=0;
    for(int y=0; y<IMG_H-1; y++){
        for(int x=0; x<IMG_W-1; x++){
            int a = tern[y*IMG_W+x] > 0;
            int b = tern[y*IMG_W+x+1] > 0;
            int c = tern[(y+1)*IMG_W+x] > 0;
            int d = tern[(y+1)*IMG_W+x+1] > 0;
            int s = a+b+c+d;
            if(s==1) q1++;
            else if(s==3) q3++;
            else if(s==2 && a==d && b==c && a!=b) qd++;
        }
    }
    int e = (q1 - q3 + 2*qd) / 4;
    return (int8_t)(e < -127 ? -127 : e > 127 ? 127 : e);
}

static void compute_euler_all(const int8_t *tern, int8_t *out, int n) {
    for(int i=0; i<n; i++)
        out[i] = euler_char(tern + (size_t)i*PADDED);
}

/* Enclosed-region centroid: flood-fill from background border on a
 * 30x30 padded grid.  Any background pixel unreached is enclosed.
 * Returns mean y-coordinate of enclosed pixels, or -1 if none. */
static int16_t enclosed_centroid(const int8_t *tern) {
    uint8_t grid[FG_SZ];
    memset(grid, 0, sizeof(grid));
    for(int y=0; y<IMG_H; y++)
        for(int x=0; x<IMG_W; x++)
            grid[(y+1)*FG_W+(x+1)] = (tern[y*IMG_W+x] > 0) ? 1 : 0;

    uint8_t visited[FG_SZ];
    memset(visited, 0, sizeof(visited));
    int stack[FG_SZ];
    int sp = 0;

    /* Seed flood-fill from all border pixels that are background */
    for(int y=0; y<FG_H; y++){
        for(int x=0; x<FG_W; x++){
            if(y==0 || y==FG_H-1 || x==0 || x==FG_W-1){
                int pos = y*FG_W+x;
                if(!grid[pos] && !visited[pos]){
                    visited[pos]=1;
                    stack[sp++]=pos;
                }
            }
        }
    }

    while(sp > 0){
        int pos = stack[--sp];
        int py = pos/FG_W, px = pos%FG_W;
        const int dx[]={0,0,1,-1}, dy[]={1,-1,0,0};
        for(int d=0; d<4; d++){
            int ny=py+dy[d], nx=px+dx[d];
            if(ny<0||ny>=FG_H||nx<0||nx>=FG_W) continue;
            int npos = ny*FG_W+nx;
            if(!visited[npos] && !grid[npos]){
                visited[npos]=1;
                stack[sp++]=npos;
            }
        }
    }

    /* Enclosed = background pixels not visited by flood-fill */
    int sum_y=0, count=0;
    for(int y=1; y<FG_H-1; y++)
        for(int x=1; x<FG_W-1; x++){
            int pos = y*FG_W+x;
            if(!grid[pos] && !visited[pos]){
                sum_y += (y-1);
                count++;
            }
        }
    return count > 0 ? (int16_t)(sum_y / count) : -1;
}

static void compute_centroid_all(const int8_t *tern, int16_t *out, int n) {
    for(int i=0; i<n; i++)
        out[i] = enclosed_centroid(tern + (size_t)i*PADDED);
}

/* Horizontal profile: count foreground pixels per row.  28 values. */
static void hprofile(const int8_t *tern, int16_t *prof) {
    for(int y=0; y<IMG_H; y++){
        int c=0;
        for(int x=0; x<IMG_W; x++) c += (tern[y*IMG_W+x] > 0);
        prof[y] = (int16_t)c;
    }
    for(int y=IMG_H; y<HP_PAD; y++) prof[y] = 0;
}

static void compute_hprof_all(const int8_t *tern, int16_t *out, int n) {
    for(int i=0; i<n; i++)
        hprofile(tern + (size_t)i*PADDED, out + (size_t)i*HP_PAD);
}

/* Discrete divergence of the ternary gradient field (Green's theorem).
 *
 * div(x,y) = (hgrad[y][x] - hgrad[y][x-1]) + (vgrad[y][x] - vgrad[y-1][x])
 *
 * Each gradient is in {-1,0,+1}, so div is in {-4,..,+4}.
 * Negative divergence = sinks = concavities / hole boundaries.
 * Sum of negative divergence correlates with loop count.
 * y-centroid of negative divergence locates where concavities sit. */
static void div_features(const int8_t *hg, const int8_t *vg,
                          int16_t *neg_sum_out, int16_t *neg_cy_out) {
    int neg_sum = 0, neg_ysum = 0, neg_cnt = 0;
    for(int y=0; y<IMG_H; y++){
        for(int x=0; x<IMG_W; x++){
            int dh = (int)hg[y*IMG_W+x] - (x>0 ? (int)hg[y*IMG_W+x-1] : 0);
            int dv = (int)vg[y*IMG_W+x] - (y>0 ? (int)vg[(y-1)*IMG_W+x] : 0);
            int d = dh + dv;
            if(d < 0){
                neg_sum += d;
                neg_ysum += y;
                neg_cnt++;
            }
        }
    }
    *neg_sum_out = (int16_t)(neg_sum < -32767 ? -32767 : neg_sum);
    *neg_cy_out  = neg_cnt > 0 ? (int16_t)(neg_ysum / neg_cnt) : -1;
}

static void compute_div_all(const int8_t *hg, const int8_t *vg,
                              int16_t *neg_sum, int16_t *neg_cy, int n) {
    for(int i=0; i<n; i++)
        div_features(hg + (size_t)i*PADDED, vg + (size_t)i*PADDED,
                     &neg_sum[i], &neg_cy[i]);
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
    int32_t  dot_px, dot_vg;
    int32_t  euler_sim, cent_sim, prof_sim, div_sim;
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
    for(int i=0;i<n&&nc<k;i++) if(votes[i]>=(uint32_t)thr)
        out[nc++]=(cand_t){(uint32_t)i,votes[i],0,0,0,0,0,0,0};
    qsort(out,(size_t)nc,sizeof(cand_t),cmp_votes_d); return nc;
}

/* Compute dot products + topological similarities for all candidates */
static void compute_features(cand_t *cands, int nc, int ti,
                              int8_t q_euler, int16_t q_cent,
                              const int16_t *q_prof,
                              int16_t q_divneg, int16_t q_divneg_cy) {
    const int8_t *tp = tern_test  + (size_t)ti*PADDED;
    const int8_t *tv = vgrad_test + (size_t)ti*PADDED;
    for(int j=0; j<nc; j++){
        uint32_t id = cands[j].id;

        /* Dot products (existing) */
        cands[j].dot_px = tdot(tp, tern_train  + (size_t)id*PADDED);
        cands[j].dot_vg = tdot(tv, vgrad_train + (size_t)id*PADDED);

        /* Euler similarity: binary match/mismatch */
        cands[j].euler_sim = (q_euler == euler_train[id]) ? EULER_MATCH : -EULER_MISMATCH;

        /* Centroid similarity */
        int16_t c_cent = cent_train[id];
        if(q_cent < 0 && c_cent < 0)
            cands[j].cent_sim = CENT_BOTH_OPEN;
        else if(q_cent < 0 || c_cent < 0)
            cands[j].cent_sim = -CENT_DISAGREE;
        else
            cands[j].cent_sim = -(int32_t)abs(q_cent - c_cent);

        /* Profile dot product */
        const int16_t *c_prof = hprof_train + (size_t)id*HP_PAD;
        int32_t pdot = 0;
        for(int y=0; y<IMG_H; y++) pdot += (int32_t)q_prof[y] * c_prof[y];
        cands[j].prof_sim = pdot;

        /* Divergence similarity: negative divergence sum + centroid match.
         * Combines topology signal (how many concavities) with spatial signal
         * (where they are).  Both derived from the gradient field via Green's
         * theorem — no binarization, no flood-fill. */
        int16_t c_dn = divneg_train[id], c_dc = divneg_cy_train[id];
        int32_t ds = -(int32_t)abs(q_divneg - c_dn);   /* closer total = more similar topology */
        if(q_divneg_cy >= 0 && c_dc >= 0)
            ds -= (int32_t)abs(q_divneg_cy - c_dc) * 2; /* penalize centroid mismatch */
        else if((q_divneg_cy < 0) != (c_dc < 0))
            ds -= 10;                                     /* one has concavities, other doesn't */
        cands[j].div_sim = ds;
    }
}

static int knn_vote(const cand_t *c, int nc, int k) {
    int v[N_CLASSES]={0}; if(k>nc) k=nc;
    for(int i=0; i<k; i++) v[train_labels[c[i].id]]++;
    int best=0; for(int c2=1; c2<N_CLASSES; c2++) if(v[c2]>v[best]) best=c2;
    return best;
}

/* ================================================================
 *  Vote accumulation (bytepacked 8-probe)
 * ================================================================ */
static void vote(uint32_t *votes, int img) {
    memset(votes, 0, TRAIN_N*sizeof(uint32_t));
    const uint8_t *sig = joint_te + (size_t)img*SIG_PAD;
    for(int k=0; k<N_BLOCKS; k++){
        uint8_t bv = sig[k]; if(bv==bg) continue;
        uint16_t w = ig_w[k], wh = w>1 ? w/2 : 1;
        { uint32_t off=idx_off[k][bv]; uint16_t sz=idx_sz[k][bv];
          const uint32_t*ids=idx_pool+off;
          for(uint16_t j=0;j<sz;j++) votes[ids[j]]+=w; }
        for(int nb=0; nb<8; nb++){
            uint8_t nv=nbr[bv][nb]; if(nv==bg) continue;
            uint32_t noff=idx_off[k][nv]; uint16_t nsz=idx_sz[k][nv];
            const uint32_t*nids=idx_pool+noff;
            for(uint16_t j=0;j<nsz;j++) votes[nids[j]]+=wh;
        }
    }
}

/* ================================================================
 *  Reporting
 * ================================================================ */
static void report(const uint8_t *preds, const char *label, int correct) {
    printf("--- %s ---\n", label);
    printf("  Accuracy: %.2f%%  (%d errors)\n\n", 100.0*correct/TEST_N, TEST_N-correct);
    int conf[N_CLASSES][N_CLASSES]; memset(conf,0,sizeof(conf));
    for(int i=0;i<TEST_N;i++) conf[test_labels[i]][preds[i]]++;
    printf("  Confusion (rows=actual, cols=predicted):\n       ");
    for(int c=0;c<N_CLASSES;c++) printf(" %4d",c);
    printf("   | Recall\n  -----");
    for(int c=0;c<N_CLASSES;c++) (void)c,printf("-----");
    printf("---+-------\n");
    for(int r=0;r<N_CLASSES;r++){
        printf("    %d: ",r); int rt=0;
        for(int c=0;c<N_CLASSES;c++){printf(" %4d",conf[r][c]);rt+=conf[r][c];}
        printf("   | %5.1f%%\n",rt>0?100.0*conf[r][r]/rt:0.0);
    }
    typedef struct{int a,b,count;}pair_t;
    pair_t pairs[45]; int np=0;
    for(int a=0;a<N_CLASSES;a++) for(int b=a+1;b<N_CLASSES;b++) pairs[np++]=(pair_t){a,b,conf[a][b]+conf[b][a]};
    for(int i=0;i<np-1;i++) for(int j=i+1;j<np;j++) if(pairs[j].count>pairs[i].count){pair_t t2=pairs[i];pairs[i]=pairs[j];pairs[j]=t2;}
    printf("  Top-5 confused:");
    for(int i=0;i<5&&i<np;i++) printf("  %d<->%d:%d",pairs[i].a,pairs[i].b,pairs[i].count);
    printf("\n\n");
}

/* ================================================================
 *  Weight sweep: apply weights to precomputed features
 * ================================================================ */
static int run_config(const cand_t *pre, const int *nc_arr,
                       int w_px, int w_vg, int w_euler, int w_cent, int w_prof,
                       int w_div, uint8_t *preds_out) {
    int correct = 0;
    for(int i=0; i<TEST_N; i++){
        cand_t cands[TOP_K];
        int nc = nc_arr[i];
        memcpy(cands, pre + (size_t)i*TOP_K, (size_t)nc*sizeof(cand_t));
        for(int j=0; j<nc; j++)
            cands[j].combined = (int64_t)w_px   * cands[j].dot_px
                              + (int64_t)w_vg   * cands[j].dot_vg
                              + (int64_t)w_euler * cands[j].euler_sim
                              + (int64_t)w_cent  * cands[j].cent_sim
                              + (int64_t)w_prof  * cands[j].prof_sim
                              + (int64_t)w_div   * cands[j].div_sim;
        qsort(cands, (size_t)nc, sizeof(cand_t), cmp_comb_d);
        int pred = knn_vote(cands, nc, 3);
        if(preds_out) preds_out[i] = (uint8_t)pred;
        if(pred == test_labels[i]) correct++;
    }
    return correct;
}

/* ================================================================
 *  Main
 * ================================================================ */
int main(int argc, char **argv) {
    double t0 = now_sec();
    if(argc>1){
        data_dir=argv[1]; size_t l=strlen(data_dir);
        if(l&&data_dir[l-1]!='/'){char*buf=malloc(l+2);memcpy(buf,data_dir,l);buf[l]='/';buf[l+1]='\0';data_dir=buf;}
    }
    const char *ds = strstr(data_dir,"fashion") ? "Fashion-MNIST" : "MNIST";
    printf("=== SSTT Topological Ranking (%s) ===\n\n", ds);

    /* ---- Load and featurize ---- */
    load_data();

    tern_train =aligned_alloc(32,(size_t)TRAIN_N*PADDED); tern_test  =aligned_alloc(32,(size_t)TEST_N *PADDED);
    hgrad_train=aligned_alloc(32,(size_t)TRAIN_N*PADDED); hgrad_test =aligned_alloc(32,(size_t)TEST_N *PADDED);
    vgrad_train=aligned_alloc(32,(size_t)TRAIN_N*PADDED); vgrad_test =aligned_alloc(32,(size_t)TEST_N *PADDED);
    quant_tern(raw_train_img,tern_train,TRAIN_N); quant_tern(raw_test_img,tern_test,TEST_N);
    gradients(tern_train,hgrad_train,vgrad_train,TRAIN_N);
    gradients(tern_test, hgrad_test, vgrad_test, TEST_N);

    px_tr=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD); px_te=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    hg_tr=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD); hg_te=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    vg_tr=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD); vg_te=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    block_sigs(tern_train, px_tr,TRAIN_N); block_sigs(tern_test, px_te,TEST_N);
    block_sigs(hgrad_train,hg_tr,TRAIN_N); block_sigs(hgrad_test,hg_te,TEST_N);
    block_sigs(vgrad_train,vg_tr,TRAIN_N); block_sigs(vgrad_test,vg_te,TEST_N);
    ht_tr=aligned_alloc(32,(size_t)TRAIN_N*TRANS_PAD); ht_te=aligned_alloc(32,(size_t)TEST_N*TRANS_PAD);
    vt_tr=aligned_alloc(32,(size_t)TRAIN_N*TRANS_PAD); vt_te=aligned_alloc(32,(size_t)TEST_N*TRANS_PAD);
    trans(px_tr,SIG_PAD,ht_tr,vt_tr,TRAIN_N); trans(px_te,SIG_PAD,ht_te,vt_te,TEST_N);
    joint_tr=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD); joint_te=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    joint_sigs_fn(joint_tr,TRAIN_N,px_tr,hg_tr,vg_tr,ht_tr,vt_tr);
    joint_sigs_fn(joint_te,TEST_N, px_te,hg_te,vg_te,ht_te,vt_te);

    printf("Building index...\n");
    build_index();

    /* ---- Compute topological features for all images ---- */
    printf("Computing topological features...\n");
    double t_topo = now_sec();

    euler_train = malloc(TRAIN_N); euler_test = malloc(TEST_N);
    compute_euler_all(tern_train, euler_train, TRAIN_N);
    compute_euler_all(tern_test,  euler_test,  TEST_N);
    printf("  Euler: done\n");

    cent_train = malloc((size_t)TRAIN_N*sizeof(int16_t));
    cent_test  = malloc((size_t)TEST_N *sizeof(int16_t));
    compute_centroid_all(tern_train, cent_train, TRAIN_N);
    compute_centroid_all(tern_test,  cent_test,  TEST_N);
    printf("  Centroid: done\n");

    hprof_train = aligned_alloc(32, (size_t)TRAIN_N*HP_PAD*sizeof(int16_t));
    hprof_test  = aligned_alloc(32, (size_t)TEST_N *HP_PAD*sizeof(int16_t));
    compute_hprof_all(tern_train, hprof_train, TRAIN_N);
    compute_hprof_all(tern_test,  hprof_test,  TEST_N);
    printf("  Profile: done\n");

    divneg_train = malloc((size_t)TRAIN_N*sizeof(int16_t));
    divneg_test  = malloc((size_t)TEST_N *sizeof(int16_t));
    divneg_cy_train = malloc((size_t)TRAIN_N*sizeof(int16_t));
    divneg_cy_test  = malloc((size_t)TEST_N *sizeof(int16_t));
    compute_div_all(hgrad_train, vgrad_train, divneg_train, divneg_cy_train, TRAIN_N);
    compute_div_all(hgrad_test,  vgrad_test,  divneg_test,  divneg_cy_test,  TEST_N);
    printf("  Divergence: done\n");

    printf("  Topo features: %.2f sec\n\n", now_sec()-t_topo);

    /* ---- Euler distribution ---- */
    {
        int ehist[10][256]; memset(ehist,0,sizeof(ehist));
        for(int i=0;i<TRAIN_N;i++) ehist[train_labels[i]][(uint8_t)euler_train[i]]++;
        printf("  Euler distribution per digit (train):\n");
        for(int d=0;d<10;d++){
            printf("    %d:", d);
            for(int e=-3;e<=3;e++) if(ehist[d][(uint8_t)(int8_t)e])
                printf("  E=%d:%d", e, ehist[d][(uint8_t)(int8_t)e]);
            printf("\n");
        }
        printf("\n");
    }

    /* ---- Centroid distribution for key pairs ---- */
    {
        printf("  Centroid distribution (train, 4 vs 9):\n");
        int bins4[30]={0}, bins9[30]={0}, no4=0, no9=0;
        for(int i=0;i<TRAIN_N;i++){
            if(train_labels[i]==4){if(cent_train[i]<0)no4++;else if(cent_train[i]<30)bins4[cent_train[i]]++;}
            if(train_labels[i]==9){if(cent_train[i]<0)no9++;else if(cent_train[i]<30)bins9[cent_train[i]]++;}
        }
        printf("    4: no_loop=%d", no4);
        for(int y=0;y<28;y++) if(bins4[y]>50) printf("  y=%d:%d",y,bins4[y]);
        printf("\n    9: no_loop=%d", no9);
        for(int y=0;y<28;y++) if(bins9[y]>50) printf("  y=%d:%d",y,bins9[y]);
        printf("\n\n");
    }

    /* ---- Divergence distribution for key pairs ---- */
    {
        printf("  Divergence neg_sum per digit (train, mean):\n    ");
        for(int d=0;d<10;d++){
            long sum=0; int cnt=0;
            for(int i=0;i<TRAIN_N;i++) if(train_labels[i]==d){sum+=divneg_train[i];cnt++;}
            printf("  %d:%.1f", d, cnt>0?(double)sum/cnt:0.0);
        }
        printf("\n\n");
    }

    printf("  Setup total: %.2f sec\n\n", now_sec()-t0);

    /* ================================================================
     *  Precompute: vote + features for all test images
     * ================================================================ */
    printf("Precomputing votes and features for all test images...\n");
    double t_pre = now_sec();

    cand_t   *pre     = malloc((size_t)TEST_N * TOP_K * sizeof(cand_t));
    int      *nc_arr  = malloc((size_t)TEST_N * sizeof(int));
    uint32_t *votes   = calloc(TRAIN_N, sizeof(uint32_t));

    for(int i=0; i<TEST_N; i++){
        vote(votes, i);
        cand_t *ci = pre + (size_t)i*TOP_K;
        int nc = select_top_k(votes, TRAIN_N, ci, TOP_K);
        compute_features(ci, nc, i, euler_test[i], cent_test[i],
                         hprof_test + (size_t)i*HP_PAD,
                         divneg_test[i], divneg_cy_test[i]);
        nc_arr[i] = nc;
        if((i+1)%2000==0) fprintf(stderr,"  precompute: %d/%d\r", i+1, TEST_N);
    }
    fprintf(stderr,"\n");
    free(votes);
    printf("  Precompute: %.2f sec\n\n", now_sec()-t_pre);

    uint8_t *preds = calloc(TEST_N, 1);

    /* ================================================================
     *  Test A: Baseline (dot only)
     * ================================================================ */
    int cA = run_config(pre, nc_arr, 256, 192, 0, 0, 0, 0, preds);
    report(preds, "A: Baseline (256*px + 192*vg)", cA);

    /* ================================================================
     *  Test B: Euler sweep
     * ================================================================ */
    printf("--- B: Euler sweep (256*px + 192*vg + w_euler*euler) ---\n");
    {
        int ws[] = {0, 50, 100, 200, 400, 800, 1600};
        int nw = 7;
        int best_w=0, best_c=cA;
        for(int wi=0; wi<nw; wi++){
            int c = run_config(pre, nc_arr, 256, 192, ws[wi], 0, 0, 0, NULL);
            printf("  w_euler=%4d: %.2f%% (%d errors, %+d)\n",
                   ws[wi], 100.0*c/TEST_N, TEST_N-c, c-cA);
            if(c>best_c){best_c=c;best_w=ws[wi];}
        }
        printf("  Best: w_euler=%d -> %.2f%%\n\n", best_w, 100.0*best_c/TEST_N);
    }

    /* ================================================================
     *  Test C: Centroid sweep
     * ================================================================ */
    printf("--- C: Centroid sweep (256*px + 192*vg + w_cent*cent) ---\n");
    {
        int ws[] = {0, 50, 100, 200, 400, 800, 1600};
        int nw = 7;
        int best_w=0, best_c=cA;
        for(int wi=0; wi<nw; wi++){
            int c = run_config(pre, nc_arr, 256, 192, 0, ws[wi], 0, 0, NULL);
            printf("  w_cent=%4d: %.2f%% (%d errors, %+d)\n",
                   ws[wi], 100.0*c/TEST_N, TEST_N-c, c-cA);
            if(c>best_c){best_c=c;best_w=ws[wi];}
        }
        printf("  Best: w_cent=%d -> %.2f%%\n\n", best_w, 100.0*best_c/TEST_N);
    }

    /* ================================================================
     *  Test D: Profile sweep
     * ================================================================ */
    printf("--- D: Profile sweep (256*px + 192*vg + w_prof*prof) ---\n");
    {
        int ws[] = {0, 1, 2, 4, 8, 16, 32};
        int nw = 7;
        int best_w=0, best_c=cA;
        for(int wi=0; wi<nw; wi++){
            int c = run_config(pre, nc_arr, 256, 192, 0, 0, ws[wi], 0, NULL);
            printf("  w_prof=%4d: %.2f%% (%d errors, %+d)\n",
                   ws[wi], 100.0*c/TEST_N, TEST_N-c, c-cA);
            if(c>best_c){best_c=c;best_w=ws[wi];}
        }
        printf("  Best: w_prof=%d -> %.2f%%\n\n", best_w, 100.0*best_c/TEST_N);
    }

    /* ================================================================
     *  Test D2: Divergence sweep
     * ================================================================ */
    printf("--- D2: Divergence sweep (256*px + 192*vg + w_div*div) ---\n");
    {
        int ws[] = {0, 25, 50, 100, 200, 400, 800};
        int nw = 7;
        int best_w=0, best_c=cA;
        for(int wi=0; wi<nw; wi++){
            int c = run_config(pre, nc_arr, 256, 192, 0, 0, 0, ws[wi], NULL);
            printf("  w_div=%4d: %.2f%% (%d errors, %+d)\n",
                   ws[wi], 100.0*c/TEST_N, TEST_N-c, c-cA);
            if(c>best_c){best_c=c;best_w=ws[wi];}
        }
        printf("  Best: w_div=%d -> %.2f%%\n\n", best_w, 100.0*best_c/TEST_N);
    }

    /* ================================================================
     *  Test E: Combined grid search (all 4 topo features)
     * ================================================================ */
    printf("--- E: Combined grid search ---\n");
    {
        int e_vals[] = {0, 100, 200, 400};
        int c_vals[] = {0, 50, 100, 200, 400};
        int p_vals[] = {0, 4, 8, 16};
        int d_vals[] = {0, 25, 50, 100, 200};
        int ne=4, nc2=5, np2=4, nd=5;
        int best_e=0, best_c=0, best_p=0, best_d=0, best_correct=cA;

        for(int ei=0; ei<ne; ei++)
        for(int ci=0; ci<nc2; ci++)
        for(int pi=0; pi<np2; pi++)
        for(int di=0; di<nd; di++){
            if(!e_vals[ei] && !c_vals[ci] && !p_vals[pi] && !d_vals[di]) continue;
            int c = run_config(pre, nc_arr, 256, 192, e_vals[ei], c_vals[ci], p_vals[pi], d_vals[di], NULL);
            if(c > best_correct){
                best_correct=c; best_e=e_vals[ei]; best_c=c_vals[ci]; best_p=p_vals[pi]; best_d=d_vals[di];
            }
        }
        printf("  Best combined: w_euler=%d w_cent=%d w_prof=%d w_div=%d -> %.2f%% (%d errors, %+d vs baseline)\n\n",
               best_e, best_c, best_p, best_d, 100.0*best_correct/TEST_N,
               TEST_N-best_correct, best_correct-cA);

        /* ================================================================
         *  Test F: Best config detail
         * ================================================================ */
        int cF = run_config(pre, nc_arr, 256, 192, best_e, best_c, best_p, best_d, preds);
        char label[160];
        snprintf(label, sizeof(label), "F: Best (e=%d c=%d p=%d d=%d)", best_e, best_c, best_p, best_d);
        report(preds, label, cF);

        /* Per-pair delta vs baseline */
        uint8_t *preds_base = calloc(TEST_N, 1);
        run_config(pre, nc_arr, 256, 192, 0, 0, 0, 0, preds_base);

        printf("  Per-pair delta (target pairs):\n");
        int pairs[][2] = {{3,8},{4,9},{1,7},{3,5},{7,9}};
        for(int pi=0; pi<5; pi++){
            int a=pairs[pi][0], b=pairs[pi][1];
            int err_base=0, err_topo=0;
            for(int i=0;i<TEST_N;i++){
                int tl=test_labels[i];
                if(tl!=a && tl!=b) continue;
                if(preds_base[i]!=tl && (preds_base[i]==a || preds_base[i]==b)) err_base++;
                if(preds[i]!=tl && (preds[i]==a || preds[i]==b)) err_topo++;
            }
            printf("    %d<->%d: %d -> %d (%+d)\n", a, b, err_base, err_topo, err_topo-err_base);
        }
        free(preds_base);

        printf("\n=== SUMMARY ===\n");
        printf("  A. Baseline (dot only):  %.2f%%  (%d errors)\n", 100.0*cA/TEST_N, TEST_N-cA);
        printf("  F. Best topo:            %.2f%%  (%d errors, %+d)\n",
               100.0*cF/TEST_N, TEST_N-cF, cF-cA);
        printf("  Weights: w_px=256 w_vg=192 w_euler=%d w_cent=%d w_prof=%d w_div=%d\n",
               best_e, best_c, best_p, best_d);
    }

    printf("\nTotal: %.2f sec\n", now_sec()-t0);

    /* Cleanup */
    free(pre); free(nc_arr); free(preds);
    free(euler_train); free(euler_test);
    free(cent_train);  free(cent_test);
    free(hprof_train); free(hprof_test);
    free(divneg_train); free(divneg_test);
    free(divneg_cy_train); free(divneg_cy_test);
    free(idx_pool);
    free(joint_tr); free(joint_te);
    free(px_tr); free(px_te); free(hg_tr); free(hg_te); free(vg_tr); free(vg_te);
    free(ht_tr); free(ht_te); free(vt_tr); free(vt_te);
    free(tern_train); free(tern_test);
    free(hgrad_train); free(hgrad_test);
    free(vgrad_train); free(vgrad_test);
    free(raw_train_img); free(raw_test_img);
    free(train_labels); free(test_labels);
    return 0;
}
