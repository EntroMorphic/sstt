/*
 * sstt_multi_threshold_topo9.c — Multi-Threshold Retrieval + Topo9 Ranking
 *
 * Integrates multi-threshold retrieval (5 quantization schemes with adaptive
 * percentiles) into the canonical topo9 ranking pipeline.
 *
 * Architecture:
 *   TRAINING: For each scheme s=1..S, build full pipeline (ternary, gradients,
 *     sigs, transitions, Encoding D, IG, index). Topo features (centroid,
 *     hprofile, divergence, grid_div) computed from scheme 1 ONLY.
 *
 *   INFERENCE: For each test image, vote with each scheme, extract top-50
 *     per scheme, merge candidates (by appearances then total votes),
 *     take top-200, compute topo features from scheme 1 data, run
 *     byte-for-byte identical topo9 ranking.
 *
 * When S=1, output must reproduce topo9's 97.27% baseline exactly.
 *
 * Build: gcc -O3 -mavx2 -march=native -Wall -Wextra -o build/sstt_multi_threshold_topo9 src/sstt_multi_threshold_topo9.c -lm
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

#define MAX_SCHEMES 5
#define CANDS_PER_SCHEME 50

static const char *data_dir="data/";
static double now_sec(void){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return ts.tv_sec+ts.tv_nsec*1e-9;}

static uint8_t *raw_train_img,*raw_test_img,*train_labels,*test_labels;

/* Scheme 1 (standard) ternary/gradient data — also used for topo features */
static int8_t *tern_train,*tern_test,*hgrad_train,*hgrad_test,*vgrad_train,*vgrad_test;

/* Topo features (scheme 1 only) */
static int16_t *cent_train,*cent_test,*hprof_train,*hprof_test;
static int16_t *divneg_train,*divneg_test,*divneg_cy_train,*divneg_cy_test;
static int16_t *gdiv_2x4_train,*gdiv_2x4_test;
static int16_t *gdiv_3x3_train,*gdiv_3x3_test;

static int *g_hist=NULL;static size_t g_hist_cap=0;

/* Per-scheme state */
typedef struct {
    int8_t *tern_train, *tern_test;
    int8_t *hgrad_train, *hgrad_test, *vgrad_train, *vgrad_test;
    uint8_t *px_tr, *px_te, *hg_tr, *hg_te, *vg_tr, *vg_te;
    uint8_t *ht_tr, *ht_te, *vt_tr, *vt_te;
    uint8_t *joint_tr, *joint_te;
    uint16_t ig_w[N_BLOCKS];
    uint8_t nbr[BYTE_VALS][8];
    uint32_t idx_off[N_BLOCKS][BYTE_VALS];
    uint16_t idx_sz[N_BLOCKS][BYTE_VALS];
    uint32_t *idx_pool;
    uint8_t bg;
    uint8_t lo_thr, hi_thr; /* fixed thresholds (0 = adaptive) */
    int adaptive_mode; /* 0=fixed, 1=P25/P75, 2=P10/P90 */
} scheme_t;

static scheme_t schemes[MAX_SCHEMES];
static int num_schemes = 1;

/* Merge candidate */
typedef struct { uint32_t id; uint32_t appearances; uint32_t total_votes; } merge_cand_t;

/* === All boilerplate (byte-for-byte from topo9.c) === */
static uint8_t *load_idx(const char*path,uint32_t*cnt,uint32_t*ro,uint32_t*co){FILE*f=fopen(path,"rb");if(!f){fprintf(stderr,"ERR:%s\n",path);exit(1);}uint32_t m,n;if(fread(&m,4,1,f)!=1||fread(&n,4,1,f)!=1){fclose(f);exit(1);}m=__builtin_bswap32(m);n=__builtin_bswap32(n);*cnt=n;size_t s=1;if((m&0xFF)>=3){uint32_t r,c;if(fread(&r,4,1,f)!=1||fread(&c,4,1,f)!=1){fclose(f);exit(1);}r=__builtin_bswap32(r);c=__builtin_bswap32(c);if(ro)*ro=r;if(co)*co=c;s=(size_t)r*c;}else{if(ro)*ro=0;if(co)*co=0;}size_t total=(size_t)n*s;uint8_t*d=malloc(total);if(!d||fread(d,1,total,f)!=total){fclose(f);exit(1);}fclose(f);return d;}
static void load_data(void){uint32_t n,r,c;char p[256];snprintf(p,sizeof(p),"%strain-images-idx3-ubyte",data_dir);raw_train_img=load_idx(p,&n,&r,&c);snprintf(p,sizeof(p),"%strain-labels-idx1-ubyte",data_dir);train_labels=load_idx(p,&n,NULL,NULL);snprintf(p,sizeof(p),"%st10k-images-idx3-ubyte",data_dir);raw_test_img=load_idx(p,&n,&r,&c);snprintf(p,sizeof(p),"%st10k-labels-idx1-ubyte",data_dir);test_labels=load_idx(p,&n,NULL,NULL);}
static inline int8_t clamp_trit(int v){return v>0?1:v<0?-1:0;}

/* Standard ternary quantization (fixed 85/170) */
static void quant_tern(const uint8_t*src,int8_t*dst,int n){const __m256i bias=_mm256_set1_epi8((char)0x80),thi=_mm256_set1_epi8((char)(170^0x80)),tlo=_mm256_set1_epi8((char)(85^0x80)),one=_mm256_set1_epi8(1);for(int i=0;i<n;i++){const uint8_t*s=src+(size_t)i*PIXELS;int8_t*d=dst+(size_t)i*PADDED;int k;for(k=0;k+32<=PIXELS;k+=32){__m256i px=_mm256_loadu_si256((const __m256i*)(s+k));__m256i sp=_mm256_xor_si256(px,bias);_mm256_storeu_si256((__m256i*)(d+k),_mm256_sub_epi8(_mm256_and_si256(_mm256_cmpgt_epi8(sp,thi),one),_mm256_and_si256(_mm256_cmpgt_epi8(tlo,sp),one)));}for(;k<PIXELS;k++)d[k]=s[k]>170?1:s[k]<85?-1:0;memset(d+PIXELS,0,PADDED-PIXELS);}}

/* Parameterized ternary quantization (arbitrary thresholds) */
static void quant_tern_thresh(const uint8_t*src,int8_t*dst,int n,uint8_t lo,uint8_t hi){
    const __m256i bias=_mm256_set1_epi8((char)0x80);
    const __m256i thi=_mm256_set1_epi8((char)(hi^0x80));
    const __m256i tlo=_mm256_set1_epi8((char)(lo^0x80));
    const __m256i one=_mm256_set1_epi8(1);
    for(int i=0;i<n;i++){
        const uint8_t*s=src+(size_t)i*PIXELS;int8_t*d=dst+(size_t)i*PADDED;int k;
        for(k=0;k+32<=PIXELS;k+=32){
            __m256i px=_mm256_loadu_si256((const __m256i*)(s+k));
            __m256i sp=_mm256_xor_si256(px,bias);
            _mm256_storeu_si256((__m256i*)(d+k),_mm256_sub_epi8(
                _mm256_and_si256(_mm256_cmpgt_epi8(sp,thi),one),
                _mm256_and_si256(_mm256_cmpgt_epi8(tlo,sp),one)));
        }
        for(;k<PIXELS;k++)d[k]=s[k]>hi?1:s[k]<lo?-1:0;
        memset(d+PIXELS,0,PADDED-PIXELS);
    }
}

/* Adaptive quantization: per-image percentile thresholds */
static void quant_tern_adaptive(const uint8_t*src,int8_t*dst,int n,int p_lo,int p_hi){
    for(int i=0;i<n;i++){
        const uint8_t*s=src+(size_t)i*PIXELS;int8_t*d=dst+(size_t)i*PADDED;
        /* Histogram of non-zero pixels */
        int hist[256];memset(hist,0,sizeof(hist));
        int nz=0;
        for(int k=0;k<PIXELS;k++)if(s[k]>0){hist[s[k]]++;nz++;}
        uint8_t lo=85,hi=170; /* fallback */
        if(nz>10){
            int target_lo=nz*p_lo/100, target_hi=nz*p_hi/100;
            int cum=0;
            for(int v=1;v<256;v++){cum+=hist[v];if(cum>=target_lo&&lo==85){lo=(uint8_t)v;}if(cum>=target_hi&&hi==170){hi=(uint8_t)v;break;}}
            if(lo>=hi){lo=85;hi=170;}
        }
        int k;
        for(k=0;k<PIXELS;k++)d[k]=s[k]>hi?1:s[k]<lo?-1:0;
        memset(d+PIXELS,0,PADDED-PIXELS);
    }
}

static void gradients(const int8_t*t,int8_t*h,int8_t*v,int n){for(int i=0;i<n;i++){const int8_t*ti=t+(size_t)i*PADDED;int8_t*hi=h+(size_t)i*PADDED;int8_t*vi=v+(size_t)i*PADDED;for(int y=0;y<IMG_H;y++){for(int x=0;x<IMG_W-1;x++)hi[y*IMG_W+x]=clamp_trit(ti[y*IMG_W+x+1]-ti[y*IMG_W+x]);hi[y*IMG_W+IMG_W-1]=0;}memset(hi+PIXELS,0,PADDED-PIXELS);for(int y=0;y<IMG_H-1;y++)for(int x=0;x<IMG_W;x++)vi[y*IMG_W+x]=clamp_trit(ti[(y+1)*IMG_W+x]-ti[y*IMG_W+x]);memset(vi+(IMG_H-1)*IMG_W,0,IMG_W);memset(vi+PIXELS,0,PADDED-PIXELS);}}
static inline uint8_t benc(int8_t a,int8_t b,int8_t c){return(uint8_t)((a+1)*9+(b+1)*3+(c+1));}
static void block_sigs(const int8_t*data,uint8_t*sigs,int n){for(int i=0;i<n;i++){const int8_t*img=data+(size_t)i*PADDED;uint8_t*sig=sigs+(size_t)i*SIG_PAD;for(int y=0;y<IMG_H;y++)for(int s=0;s<BLKS_PER_ROW;s++){int b=y*IMG_W+s*3;sig[y*BLKS_PER_ROW+s]=benc(img[b],img[b+1],img[b+2]);}memset(sig+N_BLOCKS,0xFF,SIG_PAD-N_BLOCKS);}}
static inline uint8_t tenc(uint8_t a,uint8_t b){int8_t a0=(a/9)-1,a1=((a/3)%3)-1,a2=(a%3)-1,b0=(b/9)-1,b1=((b/3)%3)-1,b2=(b%3)-1;return benc(clamp_trit(b0-a0),clamp_trit(b1-a1),clamp_trit(b2-a2));}
static void trans_fn(const uint8_t*bs,int str,uint8_t*ht,uint8_t*vt,int n){for(int i=0;i<n;i++){const uint8_t*s=bs+(size_t)i*str;uint8_t*h=ht+(size_t)i*TRANS_PAD;uint8_t*v=vt+(size_t)i*TRANS_PAD;for(int y=0;y<IMG_H;y++)for(int ss=0;ss<H_TRANS_PER_ROW;ss++)h[y*H_TRANS_PER_ROW+ss]=tenc(s[y*BLKS_PER_ROW+ss],s[y*BLKS_PER_ROW+ss+1]);memset(h+N_HTRANS,0xFF,TRANS_PAD-N_HTRANS);for(int y=0;y<V_TRANS_PER_COL;y++)for(int ss=0;ss<BLKS_PER_ROW;ss++)v[y*BLKS_PER_ROW+ss]=tenc(s[y*BLKS_PER_ROW+ss],s[(y+1)*BLKS_PER_ROW+ss]);memset(v+N_VTRANS,0xFF,TRANS_PAD-N_VTRANS);}}
static uint8_t enc_d(uint8_t px,uint8_t hg,uint8_t vg,uint8_t ht,uint8_t vt){int ps=((px/9)-1)+(((px/3)%3)-1)+((px%3)-1),hs=((hg/9)-1)+(((hg/3)%3)-1)+((hg%3)-1),vs=((vg/9)-1)+(((vg/3)%3)-1)+((vg%3)-1);uint8_t pc=ps<0?0:ps==0?1:ps<3?2:3,hc=hs<0?0:hs==0?1:hs<3?2:3,vc=vs<0?0:vs==0?1:vs<3?2:3;return pc|(hc<<2)|(vc<<4)|((ht!=BG_TRANS)?1<<6:0)|((vt!=BG_TRANS)?1<<7:0);}
static void joint_sigs_fn(uint8_t*out,int n,const uint8_t*px,const uint8_t*hg,const uint8_t*vg,const uint8_t*ht,const uint8_t*vt){for(int i=0;i<n;i++){const uint8_t*pi=px+(size_t)i*SIG_PAD,*hi=hg+(size_t)i*SIG_PAD,*vi=vg+(size_t)i*SIG_PAD;const uint8_t*hti=ht+(size_t)i*TRANS_PAD,*vti=vt+(size_t)i*TRANS_PAD;uint8_t*oi=out+(size_t)i*SIG_PAD;for(int y=0;y<IMG_H;y++)for(int s=0;s<BLKS_PER_ROW;s++){int k=y*BLKS_PER_ROW+s;uint8_t htb=s>0?hti[y*H_TRANS_PER_ROW+(s-1)]:BG_TRANS;uint8_t vtb=y>0?vti[(y-1)*BLKS_PER_ROW+s]:BG_TRANS;oi[k]=enc_d(pi[k],hi[k],vi[k],htb,vtb);}memset(oi+N_BLOCKS,0xFF,SIG_PAD-N_BLOCKS);}}
static void compute_ig(const uint8_t*sigs,int nv,uint8_t bgv,uint16_t*ig_out){int cc[N_CLASSES]={0};for(int i=0;i<TRAIN_N;i++)cc[train_labels[i]]++;double hc=0;for(int c=0;c<N_CLASSES;c++){double p=(double)cc[c]/TRAIN_N;if(p>0)hc-=p*log2(p);}double raw[N_BLOCKS],mx=0;for(int k=0;k<N_BLOCKS;k++){int*cnt=calloc((size_t)nv*N_CLASSES,sizeof(int));int*vt=calloc(nv,sizeof(int));for(int i=0;i<TRAIN_N;i++){int v=sigs[(size_t)i*SIG_PAD+k];cnt[v*N_CLASSES+train_labels[i]]++;vt[v]++;}double hcond=0;for(int v=0;v<nv;v++){if(!vt[v]||v==(int)bgv)continue;double pv=(double)vt[v]/TRAIN_N,hv=0;for(int c=0;c<N_CLASSES;c++){double pc=(double)cnt[v*N_CLASSES+c]/vt[v];if(pc>0)hv-=pc*log2(pc);}hcond+=pv*hv;}raw[k]=hc-hcond;if(raw[k]>mx)mx=raw[k];free(cnt);free(vt);}for(int k=0;k<N_BLOCKS;k++){ig_out[k]=mx>0?(uint16_t)(raw[k]/mx*IG_SCALE+0.5):1;if(!ig_out[k])ig_out[k]=1;}}

static void build_scheme_index(scheme_t *sc){
    long vc[BYTE_VALS]={0};
    for(int i=0;i<TRAIN_N;i++){const uint8_t*s=sc->joint_tr+(size_t)i*SIG_PAD;for(int k=0;k<N_BLOCKS;k++)vc[s[k]]++;}
    sc->bg=0;long mc=0;for(int v=0;v<BYTE_VALS;v++)if(vc[v]>mc){mc=vc[v];sc->bg=(uint8_t)v;}
    compute_ig(sc->joint_tr,BYTE_VALS,sc->bg,sc->ig_w);
    for(int v=0;v<BYTE_VALS;v++)for(int b=0;b<8;b++)sc->nbr[v][b]=(uint8_t)(v^(1<<b));
    memset(sc->idx_sz,0,sizeof(sc->idx_sz));
    for(int i=0;i<TRAIN_N;i++){const uint8_t*s=sc->joint_tr+(size_t)i*SIG_PAD;for(int k=0;k<N_BLOCKS;k++)if(s[k]!=sc->bg)sc->idx_sz[k][s[k]]++;}
    uint32_t tot=0;for(int k=0;k<N_BLOCKS;k++)for(int v=0;v<BYTE_VALS;v++){sc->idx_off[k][v]=tot;tot+=sc->idx_sz[k][v];}
    sc->idx_pool=malloc((size_t)tot*sizeof(uint32_t));
    uint32_t(*wp)[BYTE_VALS]=malloc((size_t)N_BLOCKS*BYTE_VALS*4);memcpy(wp,sc->idx_off,(size_t)N_BLOCKS*BYTE_VALS*4);
    for(int i=0;i<TRAIN_N;i++){const uint8_t*s=sc->joint_tr+(size_t)i*SIG_PAD;for(int k=0;k<N_BLOCKS;k++)if(s[k]!=sc->bg)sc->idx_pool[wp[k][s[k]]++]=(uint32_t)i;}
    free(wp);
    printf("    Scheme index: %u entries (%.1f MB)\n",tot,(double)tot*4/1048576);
}

/* Topo features (from scheme 1 only) — byte-for-byte from topo9.c */
static int16_t enclosed_centroid(const int8_t*tern){uint8_t grid[FG_SZ];memset(grid,0,sizeof(grid));for(int y=0;y<IMG_H;y++)for(int x=0;x<IMG_W;x++)grid[(y+1)*FG_W+(x+1)]=(tern[y*IMG_W+x]>0)?1:0;uint8_t visited[FG_SZ];memset(visited,0,sizeof(visited));int stack[FG_SZ];int sp=0;for(int y=0;y<FG_H;y++)for(int x=0;x<FG_W;x++){if(y==0||y==FG_H-1||x==0||x==FG_W-1){int pos=y*FG_W+x;if(!grid[pos]&&!visited[pos]){visited[pos]=1;stack[sp++]=pos;}}}while(sp>0){int pos=stack[--sp];int py=pos/FG_W,px2=pos%FG_W;const int dx[]={0,0,1,-1},dy[]={1,-1,0,0};for(int d=0;d<4;d++){int ny=py+dy[d],nx=px2+dx[d];if(ny<0||ny>=FG_H||nx<0||nx>=FG_W)continue;int npos=ny*FG_W+nx;if(!visited[npos]&&!grid[npos]){visited[npos]=1;stack[sp++]=npos;}}}int sum_y=0,count=0;for(int y=1;y<FG_H-1;y++)for(int x=1;x<FG_W-1;x++){int pos=y*FG_W+x;if(!grid[pos]&&!visited[pos]){sum_y+=(y-1);count++;}}return count>0?(int16_t)(sum_y/count):-1;}
static void compute_centroid_all(const int8_t*tern,int16_t*out,int n){for(int i=0;i<n;i++)out[i]=enclosed_centroid(tern+(size_t)i*PADDED);}
static void hprofile(const int8_t*tern,int16_t*prof){for(int y=0;y<IMG_H;y++){int c=0;for(int x=0;x<IMG_W;x++)c+=(tern[y*IMG_W+x]>0);prof[y]=(int16_t)c;}for(int y=IMG_H;y<HP_PAD;y++)prof[y]=0;}
static void compute_hprof_all(const int8_t*tern,int16_t*out,int n){for(int i=0;i<n;i++)hprofile(tern+(size_t)i*PADDED,out+(size_t)i*HP_PAD);}
static void div_features(const int8_t*hg,const int8_t*vg,int16_t*ns,int16_t*cy){int neg_sum=0,neg_ysum=0,neg_cnt=0;for(int y=0;y<IMG_H;y++)for(int x=0;x<IMG_W;x++){int dh=(int)hg[y*IMG_W+x]-(x>0?(int)hg[y*IMG_W+x-1]:0);int dv=(int)vg[y*IMG_W+x]-(y>0?(int)vg[(y-1)*IMG_W+x]:0);int d=dh+dv;if(d<0){neg_sum+=d;neg_ysum+=y;neg_cnt++;}}*ns=(int16_t)(neg_sum<-32767?-32767:neg_sum);*cy=neg_cnt>0?(int16_t)(neg_ysum/neg_cnt):-1;}
static void compute_div_all(const int8_t*hg,const int8_t*vg,int16_t*ns,int16_t*cy,int n){for(int i=0;i<n;i++)div_features(hg+(size_t)i*PADDED,vg+(size_t)i*PADDED,&ns[i],&cy[i]);}
static inline int32_t tdot(const int8_t*a,const int8_t*b){__m256i acc=_mm256_setzero_si256();for(int i=0;i<PADDED;i+=32)acc=_mm256_add_epi8(acc,_mm256_sign_epi8(_mm256_load_si256((const __m256i*)(a+i)),_mm256_load_si256((const __m256i*)(b+i))));__m256i lo=_mm256_cvtepi8_epi16(_mm256_castsi256_si128(acc)),hi=_mm256_cvtepi8_epi16(_mm256_extracti128_si256(acc,1));__m256i s32=_mm256_madd_epi16(_mm256_add_epi16(lo,hi),_mm256_set1_epi16(1));__m128i s=_mm_add_epi32(_mm256_castsi256_si128(s32),_mm256_extracti128_si256(s32,1));s=_mm_hadd_epi32(s,s);s=_mm_hadd_epi32(s,s);return _mm_cvtsi128_si32(s);}

/* Grid divergence — byte-for-byte from topo9.c */
static void grid_div(const int8_t*hg,const int8_t*vg,int grow,int gcol,int16_t*out){
    int nr=grow*gcol;int regions[MAX_REGIONS];memset(regions,0,sizeof(int)*nr);
    for(int y=0;y<IMG_H;y++)for(int x=0;x<IMG_W;x++){int dh=(int)hg[y*IMG_W+x]-(x>0?(int)hg[y*IMG_W+x-1]:0);int dv=(int)vg[y*IMG_W+x]-(y>0?(int)vg[(y-1)*IMG_W+x]:0);int d=dh+dv;if(d<0){int ry=y*grow/IMG_H;int rx=x*gcol/IMG_W;if(ry>=grow)ry=grow-1;if(rx>=gcol)rx=gcol-1;regions[ry*gcol+rx]+=d;}}
    for(int i=0;i<nr;i++)out[i]=(int16_t)(regions[i]<-32767?-32767:regions[i]);
}
static void compute_grid_div(const int8_t*hg,const int8_t*vg,int grow,int gcol,int16_t*out,int n){for(int i=0;i<n;i++)grid_div(hg+(size_t)i*PADDED,vg+(size_t)i*PADDED,grow,gcol,out+(size_t)i*MAX_REGIONS);}

/* ================================================================
 *  Per-scheme voting
 * ================================================================ */
static void vote_scheme(uint32_t *votes, int img, const scheme_t *sc) {
    memset(votes, 0, TRAIN_N * sizeof(uint32_t));
    const uint8_t *sig = sc->joint_te + (size_t)img * SIG_PAD;
    for (int k = 0; k < N_BLOCKS; k++) {
        uint8_t bv = sig[k];
        if (bv == sc->bg) continue;
        uint16_t w = sc->ig_w[k], wh = w > 1 ? w / 2 : 1;
        /* exact match */
        { uint32_t off=sc->idx_off[k][bv]; uint16_t sz=sc->idx_sz[k][bv];
          const uint32_t *ids=sc->idx_pool+off;
          for(uint16_t j=0;j<sz;j++) votes[ids[j]]+=w; }
        /* 8 bit-flip neighbors */
        for(int nb=0;nb<8;nb++){
            uint8_t nv=sc->nbr[bv][nb]; if(nv==sc->bg) continue;
            uint32_t noff=sc->idx_off[k][nv]; uint16_t nsz=sc->idx_sz[k][nv];
            const uint32_t *nids=sc->idx_pool+noff;
            for(uint16_t j=0;j<nsz;j++) votes[nids[j]]+=wh;
        }
    }
}

/* ================================================================
 *  Candidate structs and ranking — byte-for-byte from topo9.c
 * ================================================================ */
typedef struct {
    uint32_t id, votes;
    int32_t dot_px, dot_vg;
    int32_t div_sim, cent_sim, prof_sim;
    int32_t gdiv_sim;   /* grid div L1 (computed per-run for flexibility) */
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

/* MAD — byte-for-byte from topo9.c */
static int cmp_i16(const void*a,const void*b){return *(const int16_t*)a-*(const int16_t*)b;}
static int compute_mad(const cand_t*cands,int nc){if(nc<3)return 1000;int16_t vals[TOP_K];for(int j=0;j<nc;j++)vals[j]=divneg_train[cands[j].id];qsort(vals,(size_t)nc,sizeof(int16_t),cmp_i16);int16_t med=vals[nc/2];int16_t devs[TOP_K];for(int j=0;j<nc;j++)devs[j]=(int16_t)abs(vals[j]-med);qsort(devs,(size_t)nc,sizeof(int16_t),cmp_i16);return devs[nc/2]>0?devs[nc/2]:1;}

/* Forward declarations */
static int run_static_tuned(const cand_t*pre,const int*nc_arr,
                       const int16_t*qg,const int16_t*tg,int nreg,
                       int w_c,int w_p,int w_d,int w_g,int sc,
                       int T_MAD,int T_DIV,
                       uint8_t*preds);

/* ================================================================
 *  Static sort-then-vote — byte-for-byte from topo9.c
 * ================================================================ */
static int run_static(const cand_t*pre,const int*nc_arr,
                       const int16_t*qg,const int16_t*tg,int nreg,
                       int w_c,int w_p,int w_d,int w_g,int sc,
                       uint8_t*preds){
    return run_static_tuned(pre,nc_arr,qg,tg,nreg,w_c,w_p,w_d,w_g,sc,0,-100,preds);
}

static int run_static_tuned(const cand_t*pre,const int*nc_arr,
                       const int16_t*qg,const int16_t*tg,int nreg,
                       int w_c,int w_p,int w_d,int w_g,int sc,
                       int T_MAD,int T_DIV,
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
        int pred;
        // Method 4 disabled: testing showed NO improvement on real MNIST (doc 39)
        pred=knn_vote(cands,nc,3);
        if(preds)preds[i]=(uint8_t)pred;
        if(pred==test_labels[i])correct++;
    }
    return correct;
}

/* ================================================================
 *  Merge candidates across schemes
 * ================================================================ */
static int cmp_merge_d(const void*a,const void*b){
    const merge_cand_t*ma=(const merge_cand_t*)a,*mb=(const merge_cand_t*)b;
    if(mb->appearances!=ma->appearances) return (int)mb->appearances-(int)ma->appearances;
    return (mb->total_votes>ma->total_votes)-(mb->total_votes<ma->total_votes);
}

/* Merge top-CANDS_PER_SCHEME from each scheme into unified candidate list.
 * Returns number of merged candidates (up to TOP_K). */
static int merge_scheme_candidates(
    uint32_t votes_all[MAX_SCHEMES][TRAIN_N],
    int S,
    cand_t *out)
{
    /* Per-scheme: select top-CANDS_PER_SCHEME */
    cand_t scheme_tops[MAX_SCHEMES][CANDS_PER_SCHEME];
    int scheme_nc[MAX_SCHEMES];

    for(int s=0;s<S;s++){
        scheme_nc[s]=select_top_k(votes_all[s],TRAIN_N,scheme_tops[s],CANDS_PER_SCHEME);
    }

    /* Linear scan merge (60K is small) — use appearances/total_votes arrays */
    /* Use a flat array indexed by candidate ID, tracking appearances and total votes */
    static uint8_t seen_app[TRAIN_N];
    static uint32_t seen_votes[TRAIN_N];
    memset(seen_app,0,TRAIN_N);
    memset(seen_votes,0,TRAIN_N*sizeof(uint32_t));

    int unique_count=0;
    static uint32_t unique_ids[MAX_SCHEMES*CANDS_PER_SCHEME];

    for(int s=0;s<S;s++){
        for(int j=0;j<scheme_nc[s];j++){
            uint32_t id=scheme_tops[s][j].id;
            if(seen_app[id]==0) unique_ids[unique_count++]=id;
            seen_app[id]++;
            seen_votes[id]+=scheme_tops[s][j].votes;
        }
    }

    /* Build merge array */
    merge_cand_t *marr=malloc((size_t)unique_count*sizeof(merge_cand_t));
    for(int j=0;j<unique_count;j++){
        uint32_t id=unique_ids[j];
        marr[j]=(merge_cand_t){id,seen_app[id],seen_votes[id]};
    }
    qsort(marr,(size_t)unique_count,sizeof(merge_cand_t),cmp_merge_d);

    int nc=unique_count<TOP_K?unique_count:TOP_K;
    for(int j=0;j<nc;j++){
        out[j]=(cand_t){0};
        out[j].id=marr[j].id;
        out[j].votes=marr[j].total_votes;
    }
    /* Sort by votes descending for consistency with topo9 pipeline */
    qsort(out,(size_t)nc,sizeof(cand_t),cmp_votes_d);
    free(marr);
    return nc;
}

/* ================================================================
 *  Reporting
 * ================================================================ */
static void report(const uint8_t*preds,const char*label,int correct){
    printf("--- %s ---\n",label);printf("  Accuracy: %.2f%%  (%d errors)\n",100.0*correct/TEST_N,TEST_N-correct);
    int conf[N_CLASSES][N_CLASSES];memset(conf,0,sizeof(conf));for(int i=0;i<TEST_N;i++)conf[test_labels[i]][preds[i]]++;
    typedef struct{int a,b,count;}pair_t;pair_t pairs[45];int np=0;
    for(int a=0;a<N_CLASSES;a++)for(int b=a+1;b<N_CLASSES;b++)pairs[np++]=(pair_t){a,b,conf[a][b]+conf[b][a]};
    for(int i=0;i<np-1;i++)for(int j=i+1;j<np;j++)if(pairs[j].count>pairs[i].count){pair_t t2=pairs[i];pairs[i]=pairs[j];pairs[j]=t2;}
    printf("  Top-5:");for(int i=0;i<5&&i<np;i++)printf("  %d<->%d:%d",pairs[i].a,pairs[i].b,pairs[i].count);printf("\n\n");
}

/* Error mode analysis */
static void error_mode_analysis(const uint8_t*preds,const char*label){
    printf("  Error mode analysis (%s):\n",label);
    int mode_a=0,mode_b=0,mode_c=0;
    for(int i=0;i<TEST_N;i++){
        if(preds[i]==test_labels[i]) continue;
        int tl=test_labels[i],pl=preds[i];
        /* Mode A: confused between visually similar pairs */
        int pairs_a[][2]={{3,5},{4,9},{7,9},{3,8},{2,7},{1,7},{8,9},{0,6}};
        int is_a=0;
        for(int p=0;p<8;p++){
            if((tl==pairs_a[p][0]&&pl==pairs_a[p][1])||(tl==pairs_a[p][1]&&pl==pairs_a[p][0])){is_a=1;break;}
        }
        if(is_a){mode_a++;continue;}
        /* Mode B: off-by-one-stroke (e.g., 1->7 writing variation) */
        int pairs_b[][2]={{1,2},{2,3},{5,6},{6,8},{0,8}};
        int is_b=0;
        for(int p=0;p<5;p++){
            if((tl==pairs_b[p][0]&&pl==pairs_b[p][1])||(tl==pairs_b[p][1]&&pl==pairs_b[p][0])){is_b=1;break;}
        }
        if(is_b){mode_b++;continue;}
        /* Mode C: everything else */
        mode_c++;
    }
    int total_err=TEST_N-(int)(0); /* recount */
    total_err=0;for(int i=0;i<TEST_N;i++)if(preds[i]!=test_labels[i])total_err++;
    printf("    Mode A (visual similarity): %d (%.1f%%)\n",mode_a,100.0*mode_a/total_err);
    printf("    Mode B (stroke variation):  %d (%.1f%%)\n",mode_b,100.0*mode_b/total_err);
    printf("    Mode C (other):             %d (%.1f%%)\n",mode_c,100.0*mode_c/total_err);
    printf("    Total errors:               %d\n\n",total_err);
}

/* ================================================================
 *  Allocate and build one scheme
 * ================================================================ */
static void alloc_scheme(scheme_t *sc){
    sc->tern_train=aligned_alloc(32,(size_t)TRAIN_N*PADDED);
    sc->tern_test=aligned_alloc(32,(size_t)TEST_N*PADDED);
    sc->hgrad_train=aligned_alloc(32,(size_t)TRAIN_N*PADDED);
    sc->hgrad_test=aligned_alloc(32,(size_t)TEST_N*PADDED);
    sc->vgrad_train=aligned_alloc(32,(size_t)TRAIN_N*PADDED);
    sc->vgrad_test=aligned_alloc(32,(size_t)TEST_N*PADDED);
    sc->px_tr=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD);
    sc->px_te=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    sc->hg_tr=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD);
    sc->hg_te=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    sc->vg_tr=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD);
    sc->vg_te=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    sc->ht_tr=aligned_alloc(32,(size_t)TRAIN_N*TRANS_PAD);
    sc->ht_te=aligned_alloc(32,(size_t)TEST_N*TRANS_PAD);
    sc->vt_tr=aligned_alloc(32,(size_t)TRAIN_N*TRANS_PAD);
    sc->vt_te=aligned_alloc(32,(size_t)TEST_N*TRANS_PAD);
    sc->joint_tr=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD);
    sc->joint_te=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
}

static void build_scheme_pipeline(scheme_t *sc, int scheme_id){
    printf("  Building scheme %d pipeline", scheme_id+1);
    if(sc->adaptive_mode==0)
        printf(" (fixed %d/%d)...\n", sc->lo_thr, sc->hi_thr);
    else if(sc->adaptive_mode==1)
        printf(" (adaptive P25/P75)...\n");
    else
        printf(" (adaptive P10/P90)...\n");

    double ts=now_sec();

    /* Quantize */
    if(sc->adaptive_mode==0){
        if(sc->lo_thr==85 && sc->hi_thr==170){
            quant_tern(raw_train_img,sc->tern_train,TRAIN_N);
            quant_tern(raw_test_img,sc->tern_test,TEST_N);
        } else {
            quant_tern_thresh(raw_train_img,sc->tern_train,TRAIN_N,sc->lo_thr,sc->hi_thr);
            quant_tern_thresh(raw_test_img,sc->tern_test,TEST_N,sc->lo_thr,sc->hi_thr);
        }
    } else if(sc->adaptive_mode==1){
        quant_tern_adaptive(raw_train_img,sc->tern_train,TRAIN_N,25,75);
        quant_tern_adaptive(raw_test_img,sc->tern_test,TEST_N,25,75);
    } else {
        quant_tern_adaptive(raw_train_img,sc->tern_train,TRAIN_N,10,90);
        quant_tern_adaptive(raw_test_img,sc->tern_test,TEST_N,10,90);
    }

    gradients(sc->tern_train,sc->hgrad_train,sc->vgrad_train,TRAIN_N);
    gradients(sc->tern_test,sc->hgrad_test,sc->vgrad_test,TEST_N);
    block_sigs(sc->tern_train,sc->px_tr,TRAIN_N);block_sigs(sc->tern_test,sc->px_te,TEST_N);
    block_sigs(sc->hgrad_train,sc->hg_tr,TRAIN_N);block_sigs(sc->hgrad_test,sc->hg_te,TEST_N);
    block_sigs(sc->vgrad_train,sc->vg_tr,TRAIN_N);block_sigs(sc->vgrad_test,sc->vg_te,TEST_N);
    trans_fn(sc->px_tr,SIG_PAD,sc->ht_tr,sc->vt_tr,TRAIN_N);
    trans_fn(sc->px_te,SIG_PAD,sc->ht_te,sc->vt_te,TEST_N);
    joint_sigs_fn(sc->joint_tr,TRAIN_N,sc->px_tr,sc->hg_tr,sc->vg_tr,sc->ht_tr,sc->vt_tr);
    joint_sigs_fn(sc->joint_te,TEST_N,sc->px_te,sc->hg_te,sc->vg_te,sc->ht_te,sc->vt_te);
    build_scheme_index(sc);
    printf("    Scheme %d built in %.2f sec\n", scheme_id+1, now_sec()-ts);
}

static void free_scheme(scheme_t *sc){
    free(sc->tern_train);free(sc->tern_test);
    free(sc->hgrad_train);free(sc->hgrad_test);free(sc->vgrad_train);free(sc->vgrad_test);
    free(sc->px_tr);free(sc->px_te);free(sc->hg_tr);free(sc->hg_te);free(sc->vg_tr);free(sc->vg_te);
    free(sc->ht_tr);free(sc->ht_te);free(sc->vt_tr);free(sc->vt_te);
    free(sc->joint_tr);free(sc->joint_te);free(sc->idx_pool);
}

/* ================================================================
 *  Main
 * ================================================================ */
int main(int argc,char**argv){
    double t0=now_sec();
    if(argc>1){data_dir=argv[1];size_t l=strlen(data_dir);if(l&&data_dir[l-1]!='/'){char*buf=malloc(l+2);memcpy(buf,data_dir,l);buf[l]='/';buf[l+1]='\0';data_dir=buf;}}
    int is_fashion = strstr(data_dir,"fashion") != NULL;
    const char*ds=is_fashion?"Fashion-MNIST":"MNIST";
    printf("=== SSTT Multi-Threshold + Topo9 (%s) ===\n\n",ds);

    load_data();

    /* Configure quantization schemes */
    num_schemes = 5;
    schemes[0] = (scheme_t){0}; schemes[0].lo_thr=85;  schemes[0].hi_thr=170; schemes[0].adaptive_mode=0;
    schemes[1] = (scheme_t){0}; schemes[1].lo_thr=64;  schemes[1].hi_thr=192; schemes[1].adaptive_mode=0;
    schemes[2] = (scheme_t){0}; schemes[2].lo_thr=96;  schemes[2].hi_thr=160; schemes[2].adaptive_mode=0;
    schemes[3] = (scheme_t){0}; schemes[3].adaptive_mode=1; /* P25/P75 */
    schemes[4] = (scheme_t){0}; schemes[4].adaptive_mode=2; /* P10/P90 */

    /* Build all scheme pipelines */
    printf("Building %d scheme pipelines...\n", num_schemes);
    for(int s=0;s<num_schemes;s++){
        alloc_scheme(&schemes[s]);
        build_scheme_pipeline(&schemes[s], s);
    }

    /* Scheme 1's ternary/gradient data is the canonical data for topo features */
    tern_train = schemes[0].tern_train;
    tern_test  = schemes[0].tern_test;
    hgrad_train = schemes[0].hgrad_train;
    hgrad_test  = schemes[0].hgrad_test;
    vgrad_train = schemes[0].vgrad_train;
    vgrad_test  = schemes[0].vgrad_test;

    /* Compute topo features from scheme 1 only */
    printf("\nComputing topo features (scheme 1 only)...\n");double tf=now_sec();
    cent_train=malloc((size_t)TRAIN_N*2);cent_test=malloc((size_t)TEST_N*2);
    compute_centroid_all(tern_train,cent_train,TRAIN_N);compute_centroid_all(tern_test,cent_test,TEST_N);
    hprof_train=aligned_alloc(32,(size_t)TRAIN_N*HP_PAD*2);hprof_test=aligned_alloc(32,(size_t)TEST_N*HP_PAD*2);
    compute_hprof_all(tern_train,hprof_train,TRAIN_N);compute_hprof_all(tern_test,hprof_test,TEST_N);
    divneg_train=malloc((size_t)TRAIN_N*2);divneg_test=malloc((size_t)TEST_N*2);
    divneg_cy_train=malloc((size_t)TRAIN_N*2);divneg_cy_test=malloc((size_t)TEST_N*2);
    compute_div_all(hgrad_train,vgrad_train,divneg_train,divneg_cy_train,TRAIN_N);
    compute_div_all(hgrad_test,vgrad_test,divneg_test,divneg_cy_test,TEST_N);

    /* Both grid variants */
    gdiv_2x4_train=malloc((size_t)TRAIN_N*MAX_REGIONS*2);gdiv_2x4_test=malloc((size_t)TEST_N*MAX_REGIONS*2);
    compute_grid_div(hgrad_train,vgrad_train,2,4,gdiv_2x4_train,TRAIN_N);
    compute_grid_div(hgrad_test,vgrad_test,2,4,gdiv_2x4_test,TEST_N);
    gdiv_3x3_train=malloc((size_t)TRAIN_N*MAX_REGIONS*2);gdiv_3x3_test=malloc((size_t)TEST_N*MAX_REGIONS*2);
    compute_grid_div(hgrad_train,vgrad_train,3,3,gdiv_3x3_train,TRAIN_N);
    compute_grid_div(hgrad_test,vgrad_test,3,3,gdiv_3x3_test,TEST_N);
    printf("  Features: %.2f sec\n\n",now_sec()-tf);

    /* Select best grid for this dataset — identical to topo9.c */
    int16_t *qg, *tg; int nreg;
    int w_c, w_p, w_d, w_g, sc_val;
    if(is_fashion){
        qg=gdiv_3x3_test; tg=gdiv_3x3_train; nreg=9;
        w_c=25; w_p=0; w_d=200; w_g=200; sc_val=50;
    } else {
        qg=gdiv_2x4_test; tg=gdiv_2x4_train; nreg=8;
        w_c=50; w_p=16; w_d=200; w_g=100; sc_val=50;
    }

    /* ================================================================
     *  Test 1: Baseline — S=1 (standard only), must reproduce 97.27%
     * ================================================================ */
    printf("=== Test 1: Baseline (S=1, scheme 1 only) ===\n");
    {
        double tp=now_sec();
        cand_t*pre=malloc((size_t)TEST_N*TOP_K*sizeof(cand_t));
        int*nc_arr=malloc((size_t)TEST_N*sizeof(int));
        uint32_t*votes=calloc(TRAIN_N,sizeof(uint32_t));
        for(int i=0;i<TEST_N;i++){
            vote_scheme(votes,i,&schemes[0]);
            cand_t*ci=pre+(size_t)i*TOP_K;
            int nc=select_top_k(votes,TRAIN_N,ci,TOP_K);
            compute_base(ci,nc,i,cent_test[i],hprof_test+(size_t)i*HP_PAD,divneg_test[i],divneg_cy_test[i]);
            nc_arr[i]=nc;
            if((i+1)%2000==0)fprintf(stderr,"  %d/%d\r",i+1,TEST_N);
        }
        fprintf(stderr,"\n");free(votes);
        printf("  Precompute: %.2f sec\n",now_sec()-tp);

        uint8_t*preds=calloc(TEST_N,1);
        int cA=run_static(pre,nc_arr,qg,tg,nreg,w_c,w_p,w_d,w_g,sc_val,preds);
        report(preds,"Test 1: Baseline S=1 (must match topo9 97.27%)",cA);
        error_mode_analysis(preds,"Baseline S=1");
        free(pre);free(nc_arr);free(preds);
    }

    /* ================================================================
     *  Test 2: S=5 multi-threshold + topo9 ranking
     * ================================================================ */
    printf("=== Test 2: Multi-threshold S=5 + topo9 ranking ===\n");
    cand_t *pre_mt = NULL;
    int *nc_mt = NULL;
    uint8_t *preds_mt = NULL;
    int cMT = 0;
    {
        double tp=now_sec();
        pre_mt=malloc((size_t)TEST_N*TOP_K*sizeof(cand_t));
        nc_mt=malloc((size_t)TEST_N*sizeof(int));

        /* Allocate vote arrays for all schemes */
        uint32_t (*votes_all)[TRAIN_N] = malloc((size_t)num_schemes * TRAIN_N * sizeof(uint32_t));

        for(int i=0;i<TEST_N;i++){
            /* Vote with each scheme */
            for(int s=0;s<num_schemes;s++){
                vote_scheme(votes_all[s],i,&schemes[s]);
            }

            /* Merge candidates */
            cand_t*ci=pre_mt+(size_t)i*TOP_K;
            int nc=merge_scheme_candidates(votes_all,num_schemes,ci);

            /* Compute topo features using scheme 1 data */
            compute_base(ci,nc,i,cent_test[i],hprof_test+(size_t)i*HP_PAD,divneg_test[i],divneg_cy_test[i]);
            nc_mt[i]=nc;
            if((i+1)%2000==0)fprintf(stderr,"  %d/%d\r",i+1,TEST_N);
        }
        fprintf(stderr,"\n");free(votes_all);
        printf("  Precompute: %.2f sec\n",now_sec()-tp);

        preds_mt=calloc(TEST_N,1);
        cMT=run_static(pre_mt,nc_mt,qg,tg,nreg,w_c,w_p,w_d,w_g,sc_val,preds_mt);
        report(preds_mt,"Test 2: Multi-threshold S=5 + topo9",cMT);
        error_mode_analysis(preds_mt,"S=5 multi-threshold");
    }

    /* ================================================================
     *  Test 3: Ablation — S=1 through S=5 incrementally
     * ================================================================ */
    printf("=== Test 3: Ablation (incremental schemes) ===\n");
    {
        uint32_t (*votes_all)[TRAIN_N] = malloc((size_t)num_schemes * TRAIN_N * sizeof(uint32_t));
        uint8_t*preds_abl=calloc(TEST_N,1);

        for(int S=1;S<=num_schemes;S++){
            double tp=now_sec();
            cand_t*pre=malloc((size_t)TEST_N*TOP_K*sizeof(cand_t));
            int*nc_arr=malloc((size_t)TEST_N*sizeof(int));

            for(int i=0;i<TEST_N;i++){
                for(int s=0;s<S;s++){
                    vote_scheme(votes_all[s],i,&schemes[s]);
                }
                cand_t*ci=pre+(size_t)i*TOP_K;
                int nc;
                if(S==1){
                    nc=select_top_k(votes_all[0],TRAIN_N,ci,TOP_K);
                } else {
                    nc=merge_scheme_candidates(votes_all,S,ci);
                }
                compute_base(ci,nc,i,cent_test[i],hprof_test+(size_t)i*HP_PAD,divneg_test[i],divneg_cy_test[i]);
                nc_arr[i]=nc;
            }
            int c=run_static(pre,nc_arr,qg,tg,nreg,w_c,w_p,w_d,w_g,sc_val,preds_abl);
            printf("  S=%d: %.2f%% (%d errors, %.2f sec)\n",S,100.0*c/TEST_N,TEST_N-c,now_sec()-tp);
            free(pre);free(nc_arr);
        }
        printf("\n");
        free(votes_all);free(preds_abl);
    }

    /* ================================================================
     *  Test 4: Error mode analysis (already done above for S=1 and S=5)
     *  Print summary comparison
     * ================================================================ */
    printf("=== Test 4: Error mode comparison (see above) ===\n\n");

    /* ================================================================
     *  Test 5: Per-pair confusion deltas — S=5 vs baseline
     * ================================================================ */
    printf("=== Test 5: Per-pair confusion deltas (S=5 vs S=1 baseline) ===\n");
    {
        /* Recompute baseline preds */
        cand_t*pre_base=malloc((size_t)TEST_N*TOP_K*sizeof(cand_t));
        int*nc_base=malloc((size_t)TEST_N*sizeof(int));
        uint32_t*votes=calloc(TRAIN_N,sizeof(uint32_t));
        for(int i=0;i<TEST_N;i++){
            vote_scheme(votes,i,&schemes[0]);
            cand_t*ci=pre_base+(size_t)i*TOP_K;
            int nc=select_top_k(votes,TRAIN_N,ci,TOP_K);
            compute_base(ci,nc,i,cent_test[i],hprof_test+(size_t)i*HP_PAD,divneg_test[i],divneg_cy_test[i]);
            nc_base[i]=nc;
        }
        free(votes);
        uint8_t*preds_base=calloc(TEST_N,1);
        int cBase=run_static(pre_base,nc_base,qg,tg,nreg,w_c,w_p,w_d,w_g,sc_val,preds_base);

        printf("  %-8s %8s %8s %8s\n","Pair","Base","S=5","Delta");
        int tpairs[][2]={{3,5},{4,9},{1,7},{3,8},{7,9},{0,6},{2,4},{2,6},{2,7},{5,8},{8,9}};
        for(int pi=0;pi<11;pi++){
            int a2=tpairs[pi][0],b2=tpairs[pi][1];int eb=0,em=0;
            for(int i=0;i<TEST_N;i++){
                int tl=test_labels[i];if(tl!=a2&&tl!=b2)continue;
                if(preds_base[i]!=tl&&(preds_base[i]==a2||preds_base[i]==b2))eb++;
                if(preds_mt[i]!=tl&&(preds_mt[i]==a2||preds_mt[i]==b2))em++;
            }
            printf("  %d<->%d: %8d %8d %+8d\n",a2,b2,eb,em,em-eb);
        }
        printf("\n  Overall: Base=%.2f%% (%d err), S=5=%.2f%% (%d err), delta=%+d\n\n",
               100.0*cBase/TEST_N,TEST_N-cBase,100.0*cMT/TEST_N,TEST_N-cMT,cMT-cBase);

        free(pre_base);free(nc_base);free(preds_base);
    }

    /* Summary */
    printf("\n=== SUMMARY ===\n");
    printf("  Baseline (S=1 topo9): run above\n");
    printf("  Multi-threshold S=5:  %.2f%%  (%d errors)\n",100.0*cMT/TEST_N,TEST_N-cMT);
    printf("\nTotal: %.2f sec\n",now_sec()-t0);

    /* Cleanup */
    free(pre_mt);free(nc_mt);free(preds_mt);
    free(gdiv_2x4_train);free(gdiv_2x4_test);free(gdiv_3x3_train);free(gdiv_3x3_test);
    free(cent_train);free(cent_test);free(hprof_train);free(hprof_test);
    free(divneg_train);free(divneg_test);free(divneg_cy_train);free(divneg_cy_test);

    /* Null out scheme 1 pointers before freeing (they alias global topo ptrs) */
    /* Actually the globals just point into scheme[0], so don't double-free */
    tern_train=NULL;tern_test=NULL;hgrad_train=NULL;hgrad_test=NULL;vgrad_train=NULL;vgrad_test=NULL;
    for(int s=0;s<num_schemes;s++) free_scheme(&schemes[s]);
    free(raw_train_img);free(raw_test_img);free(train_labels);free(test_labels);
    free(g_hist);
    return 0;
}
