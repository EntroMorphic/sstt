/*
 * sstt_mtfp_dsp.c — MTFP + DSP Texture Features Experiment
 *
 * Extends MTFP with three DSP-based texture features targeting
 * Fashion-MNIST upper-body garment confusion (T-shirt/Pullover/Coat/Shirt):
 *   1. LBP (Local Binary Pattern) spatial histogram
 *   2. Haar wavelet energy per spatial region
 *   3. Gradient orientation histogram (GOH)
 *
 * Ablation on Fashion-MNIST val split, then MNIST verification.
 *
 * Build: gcc -O3 -mavx2 -mfma -march=native -Wall -Wextra -o build/sstt_mtfp_dsp src/core/sstt_mtfp_dsp.c -lm
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
#define N_BVALS    27
#define TRIT_NBR   6
#define PACKED_PAD 224
#define IG_SCALE   16
#define TOP_K      200
#define HP_PAD     32
#define FG_W       30
#define FG_H       30
#define FG_SZ      (FG_W*FG_H)
#define CENT_BOTH_OPEN  50
#define CENT_DISAGREE   80
#define MAX_REGIONS 16
#define VAL_N      5000
#define HOLDOUT_START 5000

/* DSP feature dimensions */
#define LBP_GRID   4
#define LBP_BINS   4
#define LBP_DIM    (LBP_GRID*LBP_GRID*LBP_BINS)  /* 64 */
#define HAAR_GRID  4
#define HAAR_ORI   2
#define HAAR_DIM   (HAAR_GRID*HAAR_GRID*HAAR_ORI)  /* 32 */
#define GOH_GRID   4
#define GOH_BINS   4
#define GOH_DIM    (GOH_GRID*GOH_GRID*GOH_BINS)    /* 64 */

/* Phase 2 DSP features */
#define CWHT_ROWS_START 2
#define CWHT_ROWS_END   9
#define CWHT_N_ROWS     (CWHT_ROWS_END-CWHT_ROWS_START) /* 7 collar rows */
#define CWHT_BANDS      4  /* energy in 4 frequency bands per row, summed */
#define CWHT_DIM        CWHT_BANDS
#define SYMM_DIM        1  /* single scalar */
#define GLCM_DIM        3  /* h-contrast, v-contrast, ratio */

static const char *data_dir="data/";
static double now_sec(void){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return ts.tv_sec+ts.tv_nsec*1e-9;}

/* Raw data */
static uint8_t *raw_train_img,*raw_test_img,*train_labels,*test_labels;

/* Packed ternary arrays */
static uint8_t *pk_tern_train,*pk_tern_test;
static uint8_t *pk_hgrad_train,*pk_hgrad_test;
static uint8_t *pk_vgrad_train,*pk_vgrad_test;

/* Block signatures (base-27, uint8) */
static uint8_t *px_tr,*px_te,*hg_tr,*hg_te,*vg_tr,*vg_te;

/* Per-channel inverted indices */
static uint32_t px_idx_off[N_BLOCKS][N_BVALS],hg_idx_off[N_BLOCKS][N_BVALS],vg_idx_off[N_BLOCKS][N_BVALS];
static uint16_t px_idx_sz[N_BLOCKS][N_BVALS],hg_idx_sz[N_BLOCKS][N_BVALS],vg_idx_sz[N_BLOCKS][N_BVALS];
static uint32_t *px_idx_pool,*hg_idx_pool,*vg_idx_pool;
static uint16_t px_ig_w[N_BLOCKS],hg_ig_w[N_BLOCKS],vg_ig_w[N_BLOCKS];
static uint8_t px_bg,hg_bg,vg_bg;

/* Trit neighbor table */
static uint8_t trit_nbr[N_BVALS][TRIT_NBR];

/* Existing features */
static int16_t *cent_train,*cent_test,*hprof_train,*hprof_test;
static int16_t *divneg_train,*divneg_test,*divneg_cy_train,*divneg_cy_test;
static int16_t *gdiv_3x3_train,*gdiv_3x3_test;

/* DSP features (phase 1) */
static uint16_t *lbp_train,*lbp_test;   /* [N][LBP_DIM] */
static int16_t  *haar_train,*haar_test;  /* [N][HAAR_DIM] */
static uint16_t *goh_train,*goh_test;    /* [N][GOH_DIM] */

/* DSP features (phase 2) */
static int16_t *cwht_train,*cwht_test;   /* [N][CWHT_DIM] */
static int32_t *symm_train,*symm_test;   /* [N] */
static int16_t *glcm_train,*glcm_test;   /* [N][GLCM_DIM] */

static int *g_hist=NULL;static size_t g_hist_cap=0;

/* === Data loading === */
static uint8_t *load_idx(const char*path,uint32_t*cnt,uint32_t*ro,uint32_t*co){FILE*f=fopen(path,"rb");if(!f){fprintf(stderr,"ERR:%s\n",path);exit(1);}uint32_t m,n;if(fread(&m,4,1,f)!=1||fread(&n,4,1,f)!=1){fclose(f);exit(1);}m=__builtin_bswap32(m);n=__builtin_bswap32(n);*cnt=n;size_t s=1;if((m&0xFF)>=3){uint32_t r,c;if(fread(&r,4,1,f)!=1||fread(&c,4,1,f)!=1){fclose(f);exit(1);}r=__builtin_bswap32(r);c=__builtin_bswap32(c);if(ro)*ro=r;if(co)*co=c;s=(size_t)r*c;}else{if(ro)*ro=0;if(co)*co=0;}size_t total=(size_t)n*s;uint8_t*d=malloc(total);if(!d||fread(d,1,total,f)!=total){fclose(f);exit(1);}fclose(f);return d;}
static void load_data(void){uint32_t n,r,c;char p[256];snprintf(p,sizeof(p),"%strain-images-idx3-ubyte",data_dir);raw_train_img=load_idx(p,&n,&r,&c);snprintf(p,sizeof(p),"%strain-labels-idx1-ubyte",data_dir);train_labels=load_idx(p,&n,NULL,NULL);snprintf(p,sizeof(p),"%st10k-images-idx3-ubyte",data_dir);raw_test_img=load_idx(p,&n,&r,&c);snprintf(p,sizeof(p),"%st10k-labels-idx1-ubyte",data_dir);test_labels=load_idx(p,&n,NULL,NULL);}

/* === Quantization and gradients === */
static inline int8_t clamp_trit(int v){return v>0?1:v<0?-1:0;}
static void quant_tern(const uint8_t*src,int8_t*dst,int n){const __m256i bias=_mm256_set1_epi8((char)0x80),thi=_mm256_set1_epi8((char)(170^0x80)),tlo=_mm256_set1_epi8((char)(85^0x80)),one=_mm256_set1_epi8(1);for(int i=0;i<n;i++){const uint8_t*s=src+(size_t)i*PIXELS;int8_t*d=dst+(size_t)i*PADDED;int k;for(k=0;k+32<=PIXELS;k+=32){__m256i px=_mm256_loadu_si256((const __m256i*)(s+k));__m256i sp=_mm256_xor_si256(px,bias);_mm256_storeu_si256((__m256i*)(d+k),_mm256_sub_epi8(_mm256_and_si256(_mm256_cmpgt_epi8(sp,thi),one),_mm256_and_si256(_mm256_cmpgt_epi8(tlo,sp),one)));}for(;k<PIXELS;k++)d[k]=s[k]>170?1:s[k]<85?-1:0;memset(d+PIXELS,0,PADDED-PIXELS);}}
static void gradients(const int8_t*t,int8_t*h,int8_t*v,int n){for(int i=0;i<n;i++){const int8_t*ti=t+(size_t)i*PADDED;int8_t*hi=h+(size_t)i*PADDED;int8_t*vi=v+(size_t)i*PADDED;for(int y=0;y<IMG_H;y++){for(int x=0;x<IMG_W-1;x++)hi[y*IMG_W+x]=clamp_trit(ti[y*IMG_W+x+1]-ti[y*IMG_W+x]);hi[y*IMG_W+IMG_W-1]=0;}memset(hi+PIXELS,0,PADDED-PIXELS);for(int y=0;y<IMG_H-1;y++)for(int x=0;x<IMG_W;x++)vi[y*IMG_W+x]=clamp_trit(ti[(y+1)*IMG_W+x]-ti[y*IMG_W+x]);memset(vi+(IMG_H-1)*IMG_W,0,IMG_W);memset(vi+PIXELS,0,PADDED-PIXELS);}}

/* === MTFP Trit Packing === */
static inline uint8_t pack4(int8_t t0,int8_t t1,int8_t t2,int8_t t3){
    uint8_t e0=(uint8_t)(t0==1?1:t0==0?2:3);
    uint8_t e1=(uint8_t)(t1==1?1:t1==0?2:3);
    uint8_t e2=(uint8_t)(t2==1?1:t2==0?2:3);
    uint8_t e3=(uint8_t)(t3==1?1:t3==0?2:3);
    return e0|(e1<<2)|(e2<<4)|(e3<<6);
}
static void pack_trits(const int8_t*src,uint8_t*dst,int n,int src_stride,int dst_stride){
    for(int i=0;i<n;i++){
        const int8_t*s=src+(size_t)i*src_stride;
        uint8_t*d=dst+(size_t)i*dst_stride;
        int j;
        for(j=0;j+3<PIXELS;j+=4)
            d[j/4]=pack4(s[j],s[j+1],s[j+2],s[j+3]);
        memset(d+PIXELS/4,0,dst_stride-PIXELS/4);
    }
}
static void unpack_trits(const uint8_t*packed,int8_t*out){
    static const int8_t lut[4]={0,1,0,-1};
    int n_bytes=PIXELS/4;
    for(int i=0;i<n_bytes;i++){
        uint8_t b=packed[i];
        out[i*4+0]=lut[b&3];
        out[i*4+1]=lut[(b>>2)&3];
        out[i*4+2]=lut[(b>>4)&3];
        out[i*4+3]=lut[(b>>6)&3];
    }
    memset(out+PIXELS,0,PADDED-PIXELS);
}

/* === Trit-flip neighbor table === */
static void init_trit_nbr(void){
    for(int v=0;v<N_BVALS;v++){
        int t[3]={v/9,(v/3)%3,v%3};
        int ni=0;
        for(int pos=0;pos<3;pos++){
            int mult=(pos==0)?9:(pos==1)?3:1;
            if(t[pos]>0)trit_nbr[v][ni++]=(uint8_t)(v-mult);
            if(t[pos]<2)trit_nbr[v][ni++]=(uint8_t)(v+mult);
        }
        while(ni<TRIT_NBR)trit_nbr[v][ni++]=0xFF;
    }
}

/* === Block signatures (base-27) === */
static inline uint8_t benc(int8_t a,int8_t b,int8_t c){return(uint8_t)((a+1)*9+(b+1)*3+(c+1));}
static void block_sigs(const int8_t*data,uint8_t*sigs,int n){for(int i=0;i<n;i++){const int8_t*img=data+(size_t)i*PADDED;uint8_t*sig=sigs+(size_t)i*SIG_PAD;for(int y=0;y<IMG_H;y++)for(int s=0;s<BLKS_PER_ROW;s++){int b=y*IMG_W+s*3;sig[y*BLKS_PER_ROW+s]=benc(img[b],img[b+1],img[b+2]);}memset(sig+N_BLOCKS,0xFF,SIG_PAD-N_BLOCKS);}}

/* === Information Gain === */
static void compute_ig(const uint8_t*sigs,int nv,uint8_t bgv,uint16_t*ig_out){int cc[N_CLASSES]={0};for(int i=0;i<TRAIN_N;i++)cc[train_labels[i]]++;double hc=0;for(int c=0;c<N_CLASSES;c++){double p=(double)cc[c]/TRAIN_N;if(p>0)hc-=p*log2(p);}double raw[N_BLOCKS],mx=0;for(int k=0;k<N_BLOCKS;k++){int*cnt=calloc((size_t)nv*N_CLASSES,sizeof(int));int*vt=calloc(nv,sizeof(int));for(int i=0;i<TRAIN_N;i++){int v=sigs[(size_t)i*SIG_PAD+k];if(v<nv){cnt[v*N_CLASSES+train_labels[i]]++;vt[v]++;}}double hcond=0;for(int v=0;v<nv;v++){if(!vt[v]||v==(int)bgv)continue;double pv=(double)vt[v]/TRAIN_N,hv=0;for(int c=0;c<N_CLASSES;c++){double pc=(double)cnt[v*N_CLASSES+c]/vt[v];if(pc>0)hv-=pc*log2(pc);}hcond+=pv*hv;}raw[k]=hc-hcond;if(raw[k]>mx)mx=raw[k];free(cnt);free(vt);}for(int k=0;k<N_BLOCKS;k++){ig_out[k]=mx>0?(uint16_t)(raw[k]/mx*IG_SCALE+0.5):1;if(!ig_out[k])ig_out[k]=1;}}

/* === Per-channel inverted index === */
static void build_channel_index(
    const uint8_t*sigs,
    uint8_t*bg_out,
    uint16_t*ig_out,
    uint32_t idx_off_out[N_BLOCKS][N_BVALS],
    uint16_t idx_sz_out[N_BLOCKS][N_BVALS],
    uint32_t**pool_out)
{
    long vc[N_BVALS]={0};
    for(int i=0;i<TRAIN_N;i++){const uint8_t*s=sigs+(size_t)i*SIG_PAD;for(int k=0;k<N_BLOCKS;k++)if(s[k]<N_BVALS)vc[s[k]]++;}
    uint8_t bg=0;long mc=0;
    for(int v=0;v<N_BVALS;v++)if(vc[v]>mc){mc=vc[v];bg=(uint8_t)v;}
    *bg_out=bg;
    compute_ig(sigs,N_BVALS,bg,ig_out);
    memset(idx_sz_out,0,sizeof(uint16_t)*N_BLOCKS*N_BVALS);
    for(int i=0;i<TRAIN_N;i++){const uint8_t*s=sigs+(size_t)i*SIG_PAD;for(int k=0;k<N_BLOCKS;k++)if(s[k]!=bg&&s[k]<N_BVALS)idx_sz_out[k][s[k]]++;}
    uint32_t tot=0;
    for(int k=0;k<N_BLOCKS;k++)for(int v=0;v<N_BVALS;v++){idx_off_out[k][v]=tot;tot+=idx_sz_out[k][v];}
    *pool_out=malloc((size_t)tot*sizeof(uint32_t));
    uint32_t(*wp)[N_BVALS]=malloc((size_t)N_BLOCKS*N_BVALS*4);
    memcpy(wp,idx_off_out,(size_t)N_BLOCKS*N_BVALS*4);
    for(int i=0;i<TRAIN_N;i++){const uint8_t*s=sigs+(size_t)i*SIG_PAD;for(int k=0;k<N_BLOCKS;k++)if(s[k]!=bg&&s[k]<N_BVALS)(*pool_out)[wp[k][s[k]]++]=(uint32_t)i;}
    free(wp);
    printf("  bg=%d, %u entries (%.1f MB)\n",(int)bg,tot,(double)tot*4/1048576);
}

/* === Ternary dot product (AVX2) === */
static inline int32_t tdot(const int8_t*a,const int8_t*b){__m256i acc=_mm256_setzero_si256();for(int i=0;i<PADDED;i+=32)acc=_mm256_add_epi8(acc,_mm256_sign_epi8(_mm256_load_si256((const __m256i*)(a+i)),_mm256_load_si256((const __m256i*)(b+i))));__m256i lo=_mm256_cvtepi8_epi16(_mm256_castsi256_si128(acc)),hi=_mm256_cvtepi8_epi16(_mm256_extracti128_si256(acc,1));__m256i s32=_mm256_madd_epi16(_mm256_add_epi16(lo,hi),_mm256_set1_epi16(1));__m128i s=_mm_add_epi32(_mm256_castsi256_si128(s32),_mm256_extracti128_si256(s32,1));s=_mm_hadd_epi32(s,s);s=_mm_hadd_epi32(s,s);return _mm_cvtsi128_si32(s);}

/* === Existing features === */
static int16_t enclosed_centroid(const int8_t*tern){uint8_t grid[FG_SZ];memset(grid,0,sizeof(grid));for(int y=0;y<IMG_H;y++)for(int x=0;x<IMG_W;x++)grid[(y+1)*FG_W+(x+1)]=(tern[y*IMG_W+x]>0)?1:0;uint8_t visited[FG_SZ];memset(visited,0,sizeof(visited));int stack[FG_SZ];int sp=0;for(int y=0;y<FG_H;y++)for(int x=0;x<FG_W;x++){if(y==0||y==FG_H-1||x==0||x==FG_W-1){int pos=y*FG_W+x;if(!grid[pos]&&!visited[pos]){visited[pos]=1;stack[sp++]=pos;}}}while(sp>0){int pos=stack[--sp];int py=pos/FG_W,px2=pos%FG_W;const int dx[]={0,0,1,-1},dy[]={1,-1,0,0};for(int d=0;d<4;d++){int ny=py+dy[d],nx=px2+dx[d];if(ny<0||ny>=FG_H||nx<0||nx>=FG_W)continue;int npos=ny*FG_W+nx;if(!visited[npos]&&!grid[npos]){visited[npos]=1;stack[sp++]=npos;}}}int sum_y=0,count=0;for(int y=1;y<FG_H-1;y++)for(int x=1;x<FG_W-1;x++){int pos=y*FG_W+x;if(!grid[pos]&&!visited[pos]){sum_y+=(y-1);count++;}}return count>0?(int16_t)(sum_y/count):-1;}

static void hprofile(const int8_t*tern,int16_t*prof){for(int y=0;y<IMG_H;y++){int c=0;for(int x=0;x<IMG_W;x++)c+=(tern[y*IMG_W+x]>0);prof[y]=(int16_t)c;}for(int y=IMG_H;y<HP_PAD;y++)prof[y]=0;}

static void div_features(const int8_t*hg,const int8_t*vg,int16_t*ns,int16_t*cy){int neg_sum=0,neg_ysum=0,neg_cnt=0;for(int y=0;y<IMG_H;y++)for(int x=0;x<IMG_W;x++){int dh=(int)hg[y*IMG_W+x]-(x>0?(int)hg[y*IMG_W+x-1]:0);int dv=(int)vg[y*IMG_W+x]-(y>0?(int)vg[(y-1)*IMG_W+x]:0);int d=dh+dv;if(d<0){neg_sum+=d;neg_ysum+=y;neg_cnt++;}}*ns=(int16_t)(neg_sum<-32767?-32767:neg_sum);*cy=neg_cnt>0?(int16_t)(neg_ysum/neg_cnt):-1;}

static void grid_div(const int8_t*hg,const int8_t*vg,int grow,int gcol,int16_t*out){
    int nr=grow*gcol;int regions[MAX_REGIONS];memset(regions,0,sizeof(int)*nr);
    for(int y=0;y<IMG_H;y++)for(int x=0;x<IMG_W;x++){int dh=(int)hg[y*IMG_W+x]-(x>0?(int)hg[y*IMG_W+x-1]:0);int dv=(int)vg[y*IMG_W+x]-(y>0?(int)vg[(y-1)*IMG_W+x]:0);int d=dh+dv;if(d<0){int ry=y*grow/IMG_H;int rx=x*gcol/IMG_W;if(ry>=grow)ry=grow-1;if(rx>=gcol)rx=gcol-1;regions[ry*gcol+rx]+=d;}}
    for(int i=0;i<nr;i++)out[i]=(int16_t)(regions[i]<-32767?-32767:regions[i]);
}

/* Feature computation from packed arrays */
static void compute_centroid_all_pk(const uint8_t*packed,int16_t*out,int n,int stride){
    int8_t buf[PADDED] __attribute__((aligned(32)));
    for(int i=0;i<n;i++){unpack_trits(packed+(size_t)i*stride,buf);out[i]=enclosed_centroid(buf);}
}
static void compute_hprof_all_pk(const uint8_t*packed,int16_t*out,int n,int stride){
    int8_t buf[PADDED] __attribute__((aligned(32)));
    for(int i=0;i<n;i++){unpack_trits(packed+(size_t)i*stride,buf);hprofile(buf,out+(size_t)i*HP_PAD);}
}
static void compute_div_all_pk(const uint8_t*pk_hg,const uint8_t*pk_vg,int16_t*ns,int16_t*cy,int n,int stride){
    int8_t hbuf[PADDED] __attribute__((aligned(32))),vbuf[PADDED] __attribute__((aligned(32)));
    for(int i=0;i<n;i++){unpack_trits(pk_hg+(size_t)i*stride,hbuf);unpack_trits(pk_vg+(size_t)i*stride,vbuf);div_features(hbuf,vbuf,&ns[i],&cy[i]);}
}
static void compute_grid_div_pk(const uint8_t*pk_hg,const uint8_t*pk_vg,int grow,int gcol,int16_t*out,int n,int stride){
    int8_t hbuf[PADDED] __attribute__((aligned(32))),vbuf[PADDED] __attribute__((aligned(32)));
    for(int i=0;i<n;i++){unpack_trits(pk_hg+(size_t)i*stride,hbuf);unpack_trits(pk_vg+(size_t)i*stride,vbuf);grid_div(hbuf,vbuf,grow,gcol,out+(size_t)i*MAX_REGIONS);}
}

/* ================================================================
 *  DSP Feature 1: LBP (Local Binary Pattern) spatial histogram
 *
 *  Computed on raw uint8 pixels. 4x4 spatial grid, 4 coarse bins.
 *  Bin 0: flat (LBP==0x00 or 0xFF)
 *  Bin 1: popcount(LBP) <= 2
 *  Bin 2: popcount(LBP) >= 6
 *  Bin 3: everything else (edges/textures)
 * ================================================================ */
static inline int popcount8(uint8_t v){return __builtin_popcount(v);}

static void compute_lbp_single(const uint8_t*raw,uint16_t*hist){
    memset(hist,0,LBP_DIM*sizeof(uint16_t));
    for(int y=1;y<IMG_H-1;y++){
        int gy=y*LBP_GRID/IMG_H; if(gy>=LBP_GRID)gy=LBP_GRID-1;
        for(int x=1;x<IMG_W-1;x++){
            int gx=x*LBP_GRID/IMG_W; if(gx>=LBP_GRID)gx=LBP_GRID-1;
            int cx=(int)raw[y*IMG_W+x];
            uint8_t lbp=0;
            if(raw[(y-1)*IMG_W+x-1]>=cx) lbp|=0x01; /* TL */
            if(raw[(y-1)*IMG_W+x  ]>=cx) lbp|=0x02; /* T  */
            if(raw[(y-1)*IMG_W+x+1]>=cx) lbp|=0x04; /* TR */
            if(raw[ y   *IMG_W+x+1]>=cx) lbp|=0x08; /* R  */
            if(raw[(y+1)*IMG_W+x+1]>=cx) lbp|=0x10; /* BR */
            if(raw[(y+1)*IMG_W+x  ]>=cx) lbp|=0x20; /* B  */
            if(raw[(y+1)*IMG_W+x-1]>=cx) lbp|=0x40; /* BL */
            if(raw[ y   *IMG_W+x-1]>=cx) lbp|=0x80; /* L  */

            int bin;
            if(lbp==0x00||lbp==0xFF) bin=0;
            else{int pc=popcount8(lbp);
                if(pc<=2)bin=1;
                else if(pc>=6)bin=2;
                else bin=3;}
            hist[(gy*LBP_GRID+gx)*LBP_BINS+bin]++;
        }
    }
}

static void compute_lbp_all(const uint8_t*imgs,uint16_t*out,int n){
    for(int i=0;i<n;i++)
        compute_lbp_single(imgs+(size_t)i*PIXELS,out+(size_t)i*LBP_DIM);
}

/* ================================================================
 *  DSP Feature 2: Haar wavelet energy per spatial region
 *
 *  Horizontal: sum |raw[y*28+x] - raw[y*28+x+1]| per cell
 *  Vertical:   sum |raw[y*28+x] - raw[(y+1)*28+x]| per cell
 *  4x4 grid x 2 orientations = 32-element int16 vector
 * ================================================================ */
static void compute_haar_single(const uint8_t*raw,int16_t*out){
    memset(out,0,HAAR_DIM*sizeof(int16_t));
    /* horizontal differences */
    for(int y=0;y<IMG_H;y++){
        int gy=y*HAAR_GRID/IMG_H; if(gy>=HAAR_GRID)gy=HAAR_GRID-1;
        for(int x=0;x<IMG_W-1;x++){
            int gx=x*HAAR_GRID/IMG_W; if(gx>=HAAR_GRID)gx=HAAR_GRID-1;
            int d=abs((int)raw[y*IMG_W+x]-(int)raw[y*IMG_W+x+1]);
            int idx=(gy*HAAR_GRID+gx)*HAAR_ORI+0;
            int32_t v=(int32_t)out[idx]+d;
            out[idx]=(int16_t)(v>32767?32767:v);
        }
    }
    /* vertical differences */
    for(int y=0;y<IMG_H-1;y++){
        int gy=y*HAAR_GRID/IMG_H; if(gy>=HAAR_GRID)gy=HAAR_GRID-1;
        for(int x=0;x<IMG_W;x++){
            int gx=x*HAAR_GRID/IMG_W; if(gx>=HAAR_GRID)gx=HAAR_GRID-1;
            int d=abs((int)raw[y*IMG_W+x]-(int)raw[(y+1)*IMG_W+x]);
            int idx=(gy*HAAR_GRID+gx)*HAAR_ORI+1;
            int32_t v=(int32_t)out[idx]+d;
            out[idx]=(int16_t)(v>32767?32767:v);
        }
    }
}

static void compute_haar_all(const uint8_t*imgs,int16_t*out,int n){
    for(int i=0;i<n;i++)
        compute_haar_single(imgs+(size_t)i*PIXELS,out+(size_t)i*HAAR_DIM);
}

/* ================================================================
 *  DSP Feature 3: Gradient Orientation Histogram (GOH)
 *
 *  From packed ternary hgrad/vgrad. 4x4 grid, 4 orientation bins.
 *  Bin 0: horizontal (h!=0, v==0)
 *  Bin 1: vertical   (h==0, v!=0)
 *  Bin 2: diagonal   (h==v, both nonzero)
 *  Bin 3: anti-diag  (h!=v, both nonzero)
 * ================================================================ */
static void compute_goh_single(const int8_t*hg,const int8_t*vg,uint16_t*hist){
    memset(hist,0,GOH_DIM*sizeof(uint16_t));
    for(int y=0;y<IMG_H;y++){
        int gy=y*GOH_GRID/IMG_H; if(gy>=GOH_GRID)gy=GOH_GRID-1;
        for(int x=0;x<IMG_W;x++){
            int gx=x*GOH_GRID/IMG_W; if(gx>=GOH_GRID)gx=GOH_GRID-1;
            int8_t h=hg[y*IMG_W+x];
            int8_t v=vg[y*IMG_W+x];
            if(h==0&&v==0)continue;
            int bin;
            if(h!=0&&v==0)bin=0;
            else if(h==0&&v!=0)bin=1;
            else if(h==v)bin=2;
            else bin=3;
            hist[(gy*GOH_GRID+gx)*GOH_BINS+bin]++;
        }
    }
}

static void compute_goh_all_pk(const uint8_t*pk_hg,const uint8_t*pk_vg,uint16_t*out,int n,int stride){
    int8_t hbuf[PADDED] __attribute__((aligned(32))),vbuf[PADDED] __attribute__((aligned(32)));
    for(int i=0;i<n;i++){
        unpack_trits(pk_hg+(size_t)i*stride,hbuf);
        unpack_trits(pk_vg+(size_t)i*stride,vbuf);
        compute_goh_single(hbuf,vbuf,out+(size_t)i*GOH_DIM);
    }
}

/* ================================================================
 *  DSP Similarity functions
 * ================================================================ */
static inline int32_t lbp_similarity(const uint16_t*a,const uint16_t*b){
    int32_t sim=0;
    for(int i=0;i<LBP_DIM;i++)sim-=abs((int)a[i]-(int)b[i]);
    return sim;
}

static inline int32_t haar_similarity(const int16_t*a,const int16_t*b){
    int32_t sim=0;
    for(int i=0;i<HAAR_DIM;i++)sim-=abs((int)a[i]-(int)b[i]);
    return sim;
}

static inline int32_t goh_similarity(const uint16_t*a,const uint16_t*b){
    int32_t sim=0;
    for(int i=0;i<GOH_DIM;i++)sim-=abs((int)a[i]-(int)b[i]);
    return sim;
}

/* === Phase 2 DSP: Collar WHT, Symmetry, GLCM === */

/* Collar-zone WHT energy: 1D Haar wavelet on rows CWHT_ROWS_START..CWHT_ROWS_END
 * For each collar row, compute 3 levels of Haar detail energy on 28 pixels (pad to 32).
 * Sum absolute detail coefficients per level across all collar rows.
 * Output: 4 energy values (level 0=DC, 1=fine, 2=mid, 3=coarse) */
static void compute_cwht(const uint8_t *raw, int16_t *out) {
    int32_t band[CWHT_BANDS]; memset(band, 0, sizeof(band));
    for (int y = CWHT_ROWS_START; y < CWHT_ROWS_END; y++) {
        int16_t buf[32]; memset(buf, 0, sizeof(buf));
        for (int x = 0; x < IMG_W; x++) buf[x] = (int16_t)raw[y * IMG_W + x];
        /* 3-level Haar: 32 -> 16+16 -> 8+8+16 -> 4+4+8+16 */
        int16_t tmp[32];
        /* Level 1: pairs of adjacent */
        for (int i = 0; i < 16; i++) { tmp[i] = (buf[2*i] + buf[2*i+1]) / 2; tmp[16+i] = buf[2*i] - buf[2*i+1]; }
        int32_t e1 = 0; for (int i = 16; i < 32; i++) e1 += abs(tmp[i]);
        /* Level 2 */
        int16_t t2[16];
        for (int i = 0; i < 8; i++) { t2[i] = (tmp[2*i] + tmp[2*i+1]) / 2; t2[8+i] = tmp[2*i] - tmp[2*i+1]; }
        int32_t e2 = 0; for (int i = 8; i < 16; i++) e2 += abs(t2[i]);
        /* Level 3 */
        int16_t t3[8];
        for (int i = 0; i < 4; i++) { t3[i] = (t2[2*i] + t2[2*i+1]) / 2; t3[4+i] = t2[2*i] - t2[2*i+1]; }
        int32_t e3 = 0; for (int i = 4; i < 8; i++) e3 += abs(t3[i]);
        int32_t dc = 0; for (int i = 0; i < 4; i++) dc += abs(t3[i]);
        band[0] += dc; band[1] += e1; band[2] += e2; band[3] += e3;
    }
    for (int i = 0; i < CWHT_BANDS; i++)
        out[i] = (int16_t)(band[i] > 32767 ? 32767 : band[i]);
}

static void compute_cwht_all(const uint8_t *raw, int16_t *out, int n) {
    for (int i = 0; i < n; i++)
        compute_cwht(raw + (size_t)i * PIXELS, out + (size_t)i * CWHT_DIM);
}

/* Vertical symmetry: L1 distance between left and right halves.
 * Low = symmetric (T-shirt, pullover). High = asymmetric (shirt with buttons). */
static int32_t compute_symmetry(const uint8_t *raw) {
    int32_t asym = 0;
    for (int y = 0; y < IMG_H; y++)
        for (int x = 0; x < IMG_W / 2; x++)
            asym += abs((int)raw[y * IMG_W + x] - (int)raw[y * IMG_W + (IMG_W - 1 - x)]);
    return asym;
}

static void compute_symmetry_all(const uint8_t *raw, int32_t *out, int n) {
    for (int i = 0; i < n; i++)
        out[i] = compute_symmetry(raw + (size_t)i * PIXELS);
}

/* GLCM-inspired contrast: count of adjacent pixel differences above threshold.
 * Captures texture regularity. Three features: h-contrast, v-contrast, diagonal. */
static void compute_glcm(const uint8_t *raw, int16_t *out) {
    int h_ct = 0, v_ct = 0, d_ct = 0;
    int thresh = 30; /* empirical: ~12% of max range */
    for (int y = 0; y < IMG_H; y++)
        for (int x = 0; x < IMG_W; x++) {
            int p = raw[y * IMG_W + x];
            if (x < IMG_W - 1 && abs(p - (int)raw[y * IMG_W + x + 1]) > thresh) h_ct++;
            if (y < IMG_H - 1 && abs(p - (int)raw[(y + 1) * IMG_W + x]) > thresh) v_ct++;
            if (x < IMG_W - 1 && y < IMG_H - 1 && abs(p - (int)raw[(y + 1) * IMG_W + x + 1]) > thresh) d_ct++;
        }
    out[0] = (int16_t)(h_ct > 32767 ? 32767 : h_ct);
    out[1] = (int16_t)(v_ct > 32767 ? 32767 : v_ct);
    out[2] = (int16_t)(d_ct > 32767 ? 32767 : d_ct);
}

static void compute_glcm_all(const uint8_t *raw, int16_t *out, int n) {
    for (int i = 0; i < n; i++)
        compute_glcm(raw + (size_t)i * PIXELS, out + (size_t)i * GLCM_DIM);
}

static inline int32_t cwht_similarity(const int16_t *a, const int16_t *b) {
    int32_t sim = 0;
    for (int i = 0; i < CWHT_DIM; i++) sim -= abs((int)a[i] - (int)b[i]);
    return sim;
}
static inline int32_t symm_similarity(int32_t a, int32_t b) {
    return -(int32_t)abs(a - b);
}
static inline int32_t glcm_similarity(const int16_t *a, const int16_t *b) {
    int32_t sim = 0;
    for (int i = 0; i < GLCM_DIM; i++) sim -= abs((int)a[i] - (int)b[i]);
    return sim;
}

/* === Vote (MTFP: per-channel with trit-flip neighbors) === */
static void vote_mtfp(uint32_t*votes,int img){
    memset(votes,0,TRAIN_N*sizeof(uint32_t));

    const uint8_t*ps=px_te+(size_t)img*SIG_PAD;
    for(int k=0;k<N_BLOCKS;k++){
        uint8_t bv=ps[k];if(bv==px_bg||bv>=N_BVALS)continue;
        uint16_t w=px_ig_w[k],wh=w>1?w/2:1;
        {uint32_t off=px_idx_off[k][bv];uint16_t sz=px_idx_sz[k][bv];const uint32_t*ids=px_idx_pool+off;for(uint16_t j=0;j<sz;j++)votes[ids[j]]+=w;}
        for(int nb=0;nb<TRIT_NBR;nb++){uint8_t nv=trit_nbr[bv][nb];if(nv==0xFF||nv==px_bg)continue;uint32_t noff=px_idx_off[k][nv];uint16_t nsz=px_idx_sz[k][nv];const uint32_t*nids=px_idx_pool+noff;for(uint16_t j=0;j<nsz;j++)votes[nids[j]]+=wh;}
    }

    const uint8_t*hs=hg_te+(size_t)img*SIG_PAD;
    for(int k=0;k<N_BLOCKS;k++){
        uint8_t bv=hs[k];if(bv==hg_bg||bv>=N_BVALS)continue;
        uint16_t w=hg_ig_w[k],wh=w>1?w/2:1;
        {uint32_t off=hg_idx_off[k][bv];uint16_t sz=hg_idx_sz[k][bv];const uint32_t*ids=hg_idx_pool+off;for(uint16_t j=0;j<sz;j++)votes[ids[j]]+=w;}
        for(int nb=0;nb<TRIT_NBR;nb++){uint8_t nv=trit_nbr[bv][nb];if(nv==0xFF||nv==hg_bg)continue;uint32_t noff=hg_idx_off[k][nv];uint16_t nsz=hg_idx_sz[k][nv];const uint32_t*nids=hg_idx_pool+noff;for(uint16_t j=0;j<nsz;j++)votes[nids[j]]+=wh;}
    }

    const uint8_t*vs=vg_te+(size_t)img*SIG_PAD;
    for(int k=0;k<N_BLOCKS;k++){
        uint8_t bv=vs[k];if(bv==vg_bg||bv>=N_BVALS)continue;
        uint16_t w=vg_ig_w[k],wh=w>1?w/2:1;
        {uint32_t off=vg_idx_off[k][bv];uint16_t sz=vg_idx_sz[k][bv];const uint32_t*ids=vg_idx_pool+off;for(uint16_t j=0;j<sz;j++)votes[ids[j]]+=w;}
        for(int nb=0;nb<TRIT_NBR;nb++){uint8_t nv=trit_nbr[bv][nb];if(nv==0xFF||nv==vg_bg)continue;uint32_t noff=vg_idx_off[k][nv];uint16_t nsz=vg_idx_sz[k][nv];const uint32_t*nids=vg_idx_pool+noff;for(uint16_t j=0;j<nsz;j++)votes[nids[j]]+=wh;}
    }
}

/* === Candidate selection and scoring === */
typedef struct {
    uint32_t id,votes;
    int32_t dot_px,dot_vg;
    int32_t div_sim,cent_sim,prof_sim;
    int32_t gdiv_sim;
    int32_t lbp_sim,haar_sim,goh_sim;
    int32_t cwht_sim,symm_sim,glcm_sim;
    int64_t combined;
} cand_t;

static int cmp_votes_d(const void*a,const void*b){return(int)((const cand_t*)b)->votes-(int)((const cand_t*)a)->votes;}
static int cmp_comb_d(const void*a,const void*b){int64_t da=((const cand_t*)a)->combined,db=((const cand_t*)b)->combined;return(db>da)-(db<da);}

static int select_top_k(const uint32_t*votes,int n,cand_t*out,int k){uint32_t mx=0;for(int i=0;i<n;i++)if(votes[i]>mx)mx=votes[i];if(!mx)return 0;if((size_t)(mx+1)>g_hist_cap){g_hist_cap=(size_t)(mx+1)+4096;free(g_hist);g_hist=malloc(g_hist_cap*sizeof(int));}memset(g_hist,0,(mx+1)*sizeof(int));for(int i=0;i<n;i++)if(votes[i])g_hist[votes[i]]++;int cum=0,thr;for(thr=(int)mx;thr>=1;thr--){cum+=g_hist[thr];if(cum>=k)break;}if(thr<1)thr=1;int nc=0;for(int i=0;i<n&&nc<k;i++)if(votes[i]>=(uint32_t)thr){out[nc]=(cand_t){0};out[nc].id=(uint32_t)i;out[nc].votes=votes[i];nc++;}qsort(out,(size_t)nc,sizeof(cand_t),cmp_votes_d);return nc;}

static void compute_base(cand_t*cands,int nc,int ti,int16_t q_cent,const int16_t*q_prof,int16_t q_dn,int16_t q_dc){
    int8_t tp[PADDED] __attribute__((aligned(32)));
    int8_t tv[PADDED] __attribute__((aligned(32)));
    unpack_trits(pk_tern_test+(size_t)ti*PACKED_PAD,tp);
    unpack_trits(pk_vgrad_test+(size_t)ti*PACKED_PAD,tv);

    const uint16_t*q_lbp=lbp_test+(size_t)ti*LBP_DIM;
    const int16_t *q_haar=haar_test+(size_t)ti*HAAR_DIM;
    const uint16_t*q_goh=goh_test+(size_t)ti*GOH_DIM;

    for(int j=0;j<nc;j++){uint32_t id=cands[j].id;
        int8_t tr[PADDED] __attribute__((aligned(32)));
        int8_t vr[PADDED] __attribute__((aligned(32)));
        unpack_trits(pk_tern_train+(size_t)id*PACKED_PAD,tr);
        unpack_trits(pk_vgrad_train+(size_t)id*PACKED_PAD,vr);
        cands[j].dot_px=tdot(tp,tr);
        cands[j].dot_vg=tdot(tv,vr);
        int16_t cc=cent_train[id];if(q_cent<0&&cc<0)cands[j].cent_sim=CENT_BOTH_OPEN;else if(q_cent<0||cc<0)cands[j].cent_sim=-CENT_DISAGREE;else cands[j].cent_sim=-(int32_t)abs(q_cent-cc);
        const int16_t*cp=hprof_train+(size_t)id*HP_PAD;int32_t pd=0;for(int y=0;y<IMG_H;y++)pd+=(int32_t)q_prof[y]*cp[y];cands[j].prof_sim=pd;
        int16_t cdn=divneg_train[id],cdc=divneg_cy_train[id];int32_t ds=-(int32_t)abs(q_dn-cdn);if(q_dc>=0&&cdc>=0)ds-=(int32_t)abs(q_dc-cdc)*2;else if((q_dc<0)!=(cdc<0))ds-=10;cands[j].div_sim=ds;

        /* DSP similarities */
        cands[j].lbp_sim=lbp_similarity(q_lbp,lbp_train+(size_t)id*LBP_DIM);
        cands[j].haar_sim=haar_similarity(q_haar,haar_train+(size_t)id*HAAR_DIM);
        cands[j].goh_sim=goh_similarity(q_goh,goh_train+(size_t)id*GOH_DIM);
        /* Phase 2 DSP */
        cands[j].cwht_sim=cwht_similarity(cwht_test+(size_t)ti*CWHT_DIM,cwht_train+(size_t)id*CWHT_DIM);
        cands[j].symm_sim=symm_similarity(symm_test[ti],symm_train[id]);
        cands[j].glcm_sim=glcm_similarity(glcm_test+(size_t)ti*GLCM_DIM,glcm_train+(size_t)id*GLCM_DIM);
    }
}

static int knn_vote(const cand_t*c,int nc,int k){int v[N_CLASSES]={0};if(k>nc)k=nc;for(int i=0;i<k;i++)v[train_labels[c[i].id]]++;int best=0;for(int c2=1;c2<N_CLASSES;c2++)if(v[c2]>v[best])best=c2;return best;}
static int cmp_i16(const void*a,const void*b){return *(const int16_t*)a-*(const int16_t*)b;}
static int compute_mad(const cand_t*cands,int nc){if(nc<3)return 1000;int16_t vals[TOP_K];for(int j=0;j<nc;j++)vals[j]=divneg_train[cands[j].id];qsort(vals,(size_t)nc,sizeof(int16_t),cmp_i16);int16_t med=vals[nc/2];int16_t devs[TOP_K];for(int j=0;j<nc;j++)devs[j]=(int16_t)abs(vals[j]-med);qsort(devs,(size_t)nc,sizeof(int16_t),cmp_i16);return devs[nc/2]>0?devs[nc/2]:1;}

/* === Static sort-then-vote with DSP features === */
static int run_static_dsp(const cand_t*pre,const int*nc_arr,
                           const int16_t*qg,const int16_t*tg,int nreg,
                           int w_c,int w_p,int w_d,int w_g,int sc,
                           int w_lbp,int w_haar,int w_goh,
                           int w_cwht,int w_symm,int w_glcm,
                           int start,int end,uint8_t*preds){
    int correct=0;
    for(int i=start;i<end;i++){
        cand_t cands[TOP_K];int nc=nc_arr[i];
        memcpy(cands,pre+(size_t)i*TOP_K,(size_t)nc*sizeof(cand_t));
        const int16_t*qi=qg+(size_t)i*MAX_REGIONS;
        for(int j=0;j<nc;j++){const int16_t*ci=tg+(size_t)cands[j].id*MAX_REGIONS;int32_t l=0;for(int r=0;r<nreg;r++)l+=abs(qi[r]-ci[r]);cands[j].gdiv_sim=-l;}
        int mad=compute_mad(cands,nc);int64_t s=(int64_t)sc;
        int wc=(int)(s*w_c/(s+mad)),wp=(int)(s*w_p/(s+mad)),wd=(int)(s*w_d/(s+mad)),wg=(int)(s*w_g/(s+mad));
        int wl=(int)(s*w_lbp/(s+mad)),wh=(int)(s*w_haar/(s+mad)),wo=(int)(s*w_goh/(s+mad));
        int wcw=(int)(s*w_cwht/(s+mad)),wsy=(int)(s*w_symm/(s+mad)),wgl=(int)(s*w_glcm/(s+mad));
        for(int j=0;j<nc;j++)
            cands[j].combined=(int64_t)256*cands[j].dot_px+(int64_t)192*cands[j].dot_vg
                +(int64_t)wc*cands[j].cent_sim+(int64_t)wp*cands[j].prof_sim
                +(int64_t)wd*cands[j].div_sim+(int64_t)wg*cands[j].gdiv_sim
                +(int64_t)wl*cands[j].lbp_sim+(int64_t)wh*cands[j].haar_sim
                +(int64_t)wo*cands[j].goh_sim
                +(int64_t)wcw*cands[j].cwht_sim+(int64_t)wsy*cands[j].symm_sim
                +(int64_t)wgl*cands[j].glcm_sim;
        qsort(cands,(size_t)nc,sizeof(cand_t),cmp_comb_d);
        int pred=knn_vote(cands,nc,3);
        if(preds)preds[i]=(uint8_t)pred;
        if(pred==test_labels[i])correct++;
    }
    return correct;
}

/* === Bayesian Sequential with DSP === */
static int run_bayesian_dsp(const cand_t*pre,const int*nc_arr,
                             const int16_t*qg,const int16_t*tg,int nreg,
                             int w_c,int w_p,int w_d,int w_g,int sc,
                             int w_lbp,int w_haar,int w_goh,
                             int w_cwht,int w_symm,int w_glcm,
                             int decay_S,int K_seq,int topo_w,
                             int start,int end,uint8_t*preds){
    int correct=0;
    for(int i=start;i<end;i++){
        cand_t cands[TOP_K];int nc=nc_arr[i];
        memcpy(cands,pre+(size_t)i*TOP_K,(size_t)nc*sizeof(cand_t));
        const int16_t*qi=qg+(size_t)i*MAX_REGIONS;
        for(int j=0;j<nc;j++){const int16_t*ci=tg+(size_t)cands[j].id*MAX_REGIONS;int32_t l=0;for(int r=0;r<nreg;r++)l+=abs(qi[r]-ci[r]);cands[j].gdiv_sim=-l;}
        int mad=compute_mad(cands,nc);int64_t s=(int64_t)sc;
        int wc=(int)(s*w_c/(s+mad)),wp=(int)(s*w_p/(s+mad)),wd=(int)(s*w_d/(s+mad)),wg=(int)(s*w_g/(s+mad));
        int wl=(int)(s*w_lbp/(s+mad)),wh=(int)(s*w_haar/(s+mad)),wo=(int)(s*w_goh/(s+mad));
        int wcw=(int)(s*w_cwht/(s+mad)),wsy=(int)(s*w_symm/(s+mad)),wgl=(int)(s*w_glcm/(s+mad));
        for(int j=0;j<nc;j++)
            cands[j].combined=(int64_t)256*cands[j].dot_px+(int64_t)192*cands[j].dot_vg
                +(int64_t)wc*cands[j].cent_sim+(int64_t)wp*cands[j].prof_sim
                +(int64_t)wd*cands[j].div_sim+(int64_t)wg*cands[j].gdiv_sim
                +(int64_t)wl*cands[j].lbp_sim+(int64_t)wh*cands[j].haar_sim
                +(int64_t)wo*cands[j].goh_sim
                +(int64_t)wcw*cands[j].cwht_sim+(int64_t)wsy*cands[j].symm_sim
                +(int64_t)wgl*cands[j].glcm_sim;
        qsort(cands,(size_t)nc,sizeof(cand_t),cmp_comb_d);
        int64_t state[N_CLASSES];memset(state,0,sizeof(state));
        int ks=K_seq<nc?K_seq:nc;
        for(int j=0;j<ks;j++){uint8_t lbl=train_labels[cands[j].id];int64_t evidence=cands[j].combined+(int64_t)topo_w*(cands[j].div_sim+cands[j].gdiv_sim);int64_t weight=evidence*decay_S/(decay_S+j);state[lbl]+=weight;}
        int pred=0;for(int c=1;c<N_CLASSES;c++)if(state[c]>state[pred])pred=c;
        if(preds)preds[i]=(uint8_t)pred;
        if(pred==test_labels[i])correct++;
    }
    return correct;
}

/* === Pipeline A->B with DSP === */
static int run_pipeline_dsp(const cand_t*pre,const int*nc_arr,
                             const int16_t*qg,const int16_t*tg,int nreg,
                             int w_c,int w_p,int w_d,int w_g,int sc,
                             int w_lbp,int w_haar,int w_goh,
                             int w_cwht,int w_symm,int w_glcm,
                             int decay_S,int K_a,int topo_w,
                             int gain,int exp_S,int penalty_val,int K_b,
                             int start,int end,uint8_t*preds){
    int correct=0;
    for(int i=start;i<end;i++){
        cand_t cands[TOP_K];int nc=nc_arr[i];
        memcpy(cands,pre+(size_t)i*TOP_K,(size_t)nc*sizeof(cand_t));
        const int16_t*qi=qg+(size_t)i*MAX_REGIONS;
        for(int j=0;j<nc;j++){const int16_t*ci=tg+(size_t)cands[j].id*MAX_REGIONS;int32_t l=0;for(int r=0;r<nreg;r++)l+=abs(qi[r]-ci[r]);cands[j].gdiv_sim=-l;}
        int mad=compute_mad(cands,nc);int64_t s=(int64_t)sc;
        int wc=(int)(s*w_c/(s+mad)),wp=(int)(s*w_p/(s+mad)),wd=(int)(s*w_d/(s+mad)),wg=(int)(s*w_g/(s+mad));
        int wl=(int)(s*w_lbp/(s+mad)),wh=(int)(s*w_haar/(s+mad)),wo=(int)(s*w_goh/(s+mad));
        int wcw=(int)(s*w_cwht/(s+mad)),wsy=(int)(s*w_symm/(s+mad)),wgl=(int)(s*w_glcm/(s+mad));
        for(int j=0;j<nc;j++)
            cands[j].combined=(int64_t)256*cands[j].dot_px+(int64_t)192*cands[j].dot_vg
                +(int64_t)wc*cands[j].cent_sim+(int64_t)wp*cands[j].prof_sim
                +(int64_t)wd*cands[j].div_sim+(int64_t)wg*cands[j].gdiv_sim
                +(int64_t)wl*cands[j].lbp_sim+(int64_t)wh*cands[j].haar_sim
                +(int64_t)wo*cands[j].goh_sim
                +(int64_t)wcw*cands[j].cwht_sim+(int64_t)wsy*cands[j].symm_sim
                +(int64_t)wgl*cands[j].glcm_sim;
        qsort(cands,(size_t)nc,sizeof(cand_t),cmp_comb_d);
        int64_t h[N_CLASSES];memset(h,0,sizeof(h));
        int ka=K_a<nc?K_a:nc;
        for(int j=0;j<ka;j++){uint8_t lbl=train_labels[cands[j].id];int64_t evidence=cands[j].combined+(int64_t)topo_w*(cands[j].div_sim+cands[j].gdiv_sim);int64_t weight=evidence*decay_S/(decay_S+j);h[lbl]+=weight;}
        int kb_end=ka+K_b;if(kb_end>nc)kb_end=nc;
        for(int j=ka;j<kb_end;j++){uint8_t lbl=train_labels[cands[j].id];int64_t evidence=cands[j].combined/256;int64_t decay_num=(int64_t)exp_S;int64_t decay_den=(int64_t)exp_S+1;for(int c=0;c<N_CLASSES;c++){int64_t input=(c==lbl)?(int64_t)gain*evidence:-(int64_t)penalty_val;h[c]=(decay_num*h[c]+input)/decay_den;}}
        int pred=0;for(int c=1;c<N_CLASSES;c++)if(h[c]>h[pred])pred=c;
        if(preds)preds[i]=(uint8_t)pred;
        if(pred==test_labels[i])correct++;
    }
    return correct;
}

static void report(const uint8_t*preds,const char*label,int correct,int start,int end){
    int n=end-start;
    printf("--- %s ---\n",label);printf("  Accuracy: %.2f%%  (%d/%d correct, %d errors)\n",100.0*correct/n,correct,n,n-correct);
    int conf[N_CLASSES][N_CLASSES];memset(conf,0,sizeof(conf));for(int i=start;i<end;i++)conf[test_labels[i]][preds[i]]++;
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
    int is_fashion=strstr(data_dir,"fashion")!=NULL;
    const char*ds=is_fashion?"Fashion-MNIST":"MNIST";
    printf("=== SSTT MTFP + DSP Features (%s) ===\n\n",ds);
    printf("  Val: images 0-%d  |  Holdout: images %d-%d\n\n",VAL_N-1,HOLDOUT_START,TEST_N-1);

    /* 1. Load raw data */
    load_data();

    /* 2. Allocate temp int8 arrays, quantize, compute gradients */
    int8_t *tern_train=aligned_alloc(32,(size_t)TRAIN_N*PADDED);
    int8_t *tern_test=aligned_alloc(32,(size_t)TEST_N*PADDED);
    int8_t *hgrad_train=aligned_alloc(32,(size_t)TRAIN_N*PADDED);
    int8_t *hgrad_test=aligned_alloc(32,(size_t)TEST_N*PADDED);
    int8_t *vgrad_train=aligned_alloc(32,(size_t)TRAIN_N*PADDED);
    int8_t *vgrad_test=aligned_alloc(32,(size_t)TEST_N*PADDED);
    quant_tern(raw_train_img,tern_train,TRAIN_N);
    quant_tern(raw_test_img,tern_test,TEST_N);
    gradients(tern_train,hgrad_train,vgrad_train,TRAIN_N);
    gradients(tern_test,hgrad_test,vgrad_test,TEST_N);

    /* 3. Block signatures on int8 arrays (base-27) */
    px_tr=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD);px_te=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    hg_tr=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD);hg_te=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    vg_tr=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD);vg_te=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    block_sigs(tern_train,px_tr,TRAIN_N);block_sigs(tern_test,px_te,TEST_N);
    block_sigs(hgrad_train,hg_tr,TRAIN_N);block_sigs(hgrad_test,hg_te,TEST_N);
    block_sigs(vgrad_train,vg_tr,TRAIN_N);block_sigs(vgrad_test,vg_te,TEST_N);

    /* 4. Pack int8 trits into MTFP format */
    size_t pk_train_sz=(size_t)TRAIN_N*PACKED_PAD;
    size_t pk_test_sz=(size_t)TEST_N*PACKED_PAD;
    pk_tern_train=aligned_alloc(32,pk_train_sz);pk_tern_test=aligned_alloc(32,pk_test_sz);
    pk_hgrad_train=aligned_alloc(32,pk_train_sz);pk_hgrad_test=aligned_alloc(32,pk_test_sz);
    pk_vgrad_train=aligned_alloc(32,pk_train_sz);pk_vgrad_test=aligned_alloc(32,pk_test_sz);
    pack_trits(tern_train,pk_tern_train,TRAIN_N,PADDED,PACKED_PAD);
    pack_trits(tern_test,pk_tern_test,TEST_N,PADDED,PACKED_PAD);
    pack_trits(hgrad_train,pk_hgrad_train,TRAIN_N,PADDED,PACKED_PAD);
    pack_trits(hgrad_test,pk_hgrad_test,TEST_N,PADDED,PACKED_PAD);
    pack_trits(vgrad_train,pk_vgrad_train,TRAIN_N,PADDED,PACKED_PAD);
    pack_trits(vgrad_test,pk_vgrad_test,TEST_N,PADDED,PACKED_PAD);

    /* 5. Compute existing features */
    printf("Computing features...\n");double tf=now_sec();
    cent_train=malloc((size_t)TRAIN_N*2);cent_test=malloc((size_t)TEST_N*2);
    hprof_train=aligned_alloc(32,(size_t)TRAIN_N*HP_PAD*2);hprof_test=aligned_alloc(32,(size_t)TEST_N*HP_PAD*2);
    divneg_train=malloc((size_t)TRAIN_N*2);divneg_test=malloc((size_t)TEST_N*2);
    divneg_cy_train=malloc((size_t)TRAIN_N*2);divneg_cy_test=malloc((size_t)TEST_N*2);
    gdiv_3x3_train=malloc((size_t)TRAIN_N*MAX_REGIONS*2);gdiv_3x3_test=malloc((size_t)TEST_N*MAX_REGIONS*2);

    compute_centroid_all_pk(pk_tern_train,cent_train,TRAIN_N,PACKED_PAD);
    compute_centroid_all_pk(pk_tern_test,cent_test,TEST_N,PACKED_PAD);
    compute_hprof_all_pk(pk_tern_train,hprof_train,TRAIN_N,PACKED_PAD);
    compute_hprof_all_pk(pk_tern_test,hprof_test,TEST_N,PACKED_PAD);
    compute_div_all_pk(pk_hgrad_train,pk_vgrad_train,divneg_train,divneg_cy_train,TRAIN_N,PACKED_PAD);
    compute_div_all_pk(pk_hgrad_test,pk_vgrad_test,divneg_test,divneg_cy_test,TEST_N,PACKED_PAD);
    compute_grid_div_pk(pk_hgrad_train,pk_vgrad_train,3,3,gdiv_3x3_train,TRAIN_N,PACKED_PAD);
    compute_grid_div_pk(pk_hgrad_test,pk_vgrad_test,3,3,gdiv_3x3_test,TEST_N,PACKED_PAD);
    printf("  Existing features: %.2fs\n",now_sec()-tf);

    /* 6. Compute DSP features */
    printf("\nComputing DSP features...\n");

    /* LBP */
    double t_lbp=now_sec();
    lbp_train=malloc((size_t)TRAIN_N*LBP_DIM*sizeof(uint16_t));
    lbp_test=malloc((size_t)TEST_N*LBP_DIM*sizeof(uint16_t));
    compute_lbp_all(raw_train_img,lbp_train,TRAIN_N);
    compute_lbp_all(raw_test_img,lbp_test,TEST_N);
    printf("  LBP histograms: %.2fs\n",now_sec()-t_lbp);

    /* Haar */
    double t_haar=now_sec();
    haar_train=malloc((size_t)TRAIN_N*HAAR_DIM*sizeof(int16_t));
    haar_test=malloc((size_t)TEST_N*HAAR_DIM*sizeof(int16_t));
    compute_haar_all(raw_train_img,haar_train,TRAIN_N);
    compute_haar_all(raw_test_img,haar_test,TEST_N);
    printf("  Haar energy: %.2fs\n",now_sec()-t_haar);

    /* GOH */
    double t_goh=now_sec();
    goh_train=malloc((size_t)TRAIN_N*GOH_DIM*sizeof(uint16_t));
    goh_test=malloc((size_t)TEST_N*GOH_DIM*sizeof(uint16_t));
    compute_goh_all_pk(pk_hgrad_train,pk_vgrad_train,goh_train,TRAIN_N,PACKED_PAD);
    compute_goh_all_pk(pk_hgrad_test,pk_vgrad_test,goh_test,TEST_N,PACKED_PAD);
    printf("  Gradient orientation: %.2fs\n",now_sec()-t_goh);

    /* Collar WHT */
    printf("  Collar WHT: "); double tcw=now_sec();
    cwht_train=malloc((size_t)TRAIN_N*CWHT_DIM*sizeof(int16_t));
    cwht_test=malloc((size_t)TEST_N*CWHT_DIM*sizeof(int16_t));
    compute_cwht_all(raw_train_img,cwht_train,TRAIN_N);
    compute_cwht_all(raw_test_img,cwht_test,TEST_N);
    printf("%.2fs\n",now_sec()-tcw);

    /* Symmetry */
    printf("  Symmetry: "); double tsy=now_sec();
    symm_train=malloc((size_t)TRAIN_N*sizeof(int32_t));
    symm_test=malloc((size_t)TEST_N*sizeof(int32_t));
    compute_symmetry_all(raw_train_img,symm_train,TRAIN_N);
    compute_symmetry_all(raw_test_img,symm_test,TEST_N);
    printf("%.2fs\n",now_sec()-tsy);

    /* GLCM contrast */
    printf("  GLCM contrast: "); double tgl=now_sec();
    glcm_train=malloc((size_t)TRAIN_N*GLCM_DIM*sizeof(int16_t));
    glcm_test=malloc((size_t)TEST_N*GLCM_DIM*sizeof(int16_t));
    compute_glcm_all(raw_train_img,glcm_train,TRAIN_N);
    compute_glcm_all(raw_test_img,glcm_test,TEST_N);
    printf("%.2fs\n",now_sec()-tgl);

    printf("  DSP total: %.2fs\n\n",now_sec()-t_lbp);

    /* 7. Free temp int8 arrays */
    free(tern_train);free(tern_test);
    free(hgrad_train);free(hgrad_test);
    free(vgrad_train);free(vgrad_test);

    /* 8. Init trit neighbor table */
    init_trit_nbr();

    /* 9. Build per-channel indices */
    printf("Building per-channel indices...\n");
    printf("  PX: ");build_channel_index(px_tr,&px_bg,px_ig_w,px_idx_off,px_idx_sz,&px_idx_pool);
    printf("  HG: ");build_channel_index(hg_tr,&hg_bg,hg_ig_w,hg_idx_off,hg_idx_sz,&hg_idx_pool);
    printf("  VG: ");build_channel_index(vg_tr,&vg_bg,vg_ig_w,vg_idx_off,vg_idx_sz,&vg_idx_pool);
    printf("\n");

    /* 10. Precompute candidates (with DSP sims computed in compute_base) */
    printf("Precomputing candidates...\n");double tp=now_sec();
    cand_t*pre=malloc((size_t)TEST_N*TOP_K*sizeof(cand_t));
    int*nc_arr=malloc((size_t)TEST_N*sizeof(int));
    uint32_t*votes=calloc(TRAIN_N,sizeof(uint32_t));
    for(int i=0;i<TEST_N;i++){
        vote_mtfp(votes,i);
        cand_t*ci=pre+(size_t)i*TOP_K;
        int nc=select_top_k(votes,TRAIN_N,ci,TOP_K);
        compute_base(ci,nc,i,cent_test[i],hprof_test+(size_t)i*HP_PAD,divneg_test[i],divneg_cy_test[i]);
        nc_arr[i]=nc;
        if((i+1)%2000==0)fprintf(stderr,"  %d/%d\r",i+1,TEST_N);
    }
    fprintf(stderr,"\n");free(votes);
    printf("  %.2f sec\n\n",now_sec()-tp);

    uint8_t*preds=calloc(TEST_N,1);

    /* Base weights (Fashion val-derived from MTFP run) */
    int16_t *qg=gdiv_3x3_test,*tg=gdiv_3x3_train;
    int nreg=9;
    int w_c=25,w_p=0,w_d=50,w_g=100,sc=20;

    /* ================================================================
     * PHASE 1: Ablation on Fashion val (images 0-4999)
     * ================================================================ */
    printf("=== Ablation (%s val, images 0-%d) ===\n\n",ds,VAL_N-1);

    /* Baseline: no DSP features */
    int baseline=run_static_dsp(pre,nc_arr,qg,tg,nreg,w_c,w_p,w_d,w_g,sc,0,0,0,0,0,0,0,VAL_N,NULL);
    printf("%-40s %-10s %s\n","Method","Val Acc","Delta");
    printf("%-40s %5.2f%%     %s\n","MTFP baseline (no DSP)",100.0*baseline/VAL_N,"---");

    /* Grid search for each feature independently */
    int wdsp_vals[]={0,50,100,200};
    int n_wdsp=4;

    /* LBP only */
    int best_lbp_c=baseline,best_lbp_w=0;
    for(int wi=1;wi<n_wdsp;wi++){
        int c=run_static_dsp(pre,nc_arr,qg,tg,nreg,w_c,w_p,w_d,w_g,sc,
                              wdsp_vals[wi],0,0,0,0,0,0,VAL_N,NULL);
        if(c>best_lbp_c){best_lbp_c=c;best_lbp_w=wdsp_vals[wi];}
    }
    printf("+ LBP (w=%-3d)                            %5.2f%%     %+.2f\n",
           best_lbp_w,100.0*best_lbp_c/VAL_N,100.0*(best_lbp_c-baseline)/VAL_N);

    /* Haar only */
    int best_haar_c=baseline,best_haar_w=0;
    for(int wi=1;wi<n_wdsp;wi++){
        int c=run_static_dsp(pre,nc_arr,qg,tg,nreg,w_c,w_p,w_d,w_g,sc,
                              0,wdsp_vals[wi],0,0,0,0,0,VAL_N,NULL);
        if(c>best_haar_c){best_haar_c=c;best_haar_w=wdsp_vals[wi];}
    }
    printf("+ Haar (w=%-3d)                           %5.2f%%     %+.2f\n",
           best_haar_w,100.0*best_haar_c/VAL_N,100.0*(best_haar_c-baseline)/VAL_N);

    /* GOH only */
    int best_goh_c=baseline,best_goh_w=0;
    for(int wi=1;wi<n_wdsp;wi++){
        int c=run_static_dsp(pre,nc_arr,qg,tg,nreg,w_c,w_p,w_d,w_g,sc,
                              0,0,wdsp_vals[wi],0,0,0,0,VAL_N,NULL);
        if(c>best_goh_c){best_goh_c=c;best_goh_w=wdsp_vals[wi];}
    }
    printf("+ GOH (w=%-3d)                            %5.2f%%     %+.2f\n",
           best_goh_w,100.0*best_goh_c/VAL_N,100.0*(best_goh_c-baseline)/VAL_N);

    /* All three combined: grid search */
    int best_all_c=baseline,best_all_lbp=0,best_all_haar=0,best_all_goh=0;
    for(int li=0;li<n_wdsp;li++)
    for(int hi=0;hi<n_wdsp;hi++)
    for(int gi=0;gi<n_wdsp;gi++){
        if(wdsp_vals[li]==0&&wdsp_vals[hi]==0&&wdsp_vals[gi]==0)continue;
        int c=run_static_dsp(pre,nc_arr,qg,tg,nreg,w_c,w_p,w_d,w_g,sc,
                              wdsp_vals[li],wdsp_vals[hi],wdsp_vals[gi],0,0,0,0,VAL_N,NULL);
        if(c>best_all_c){best_all_c=c;best_all_lbp=wdsp_vals[li];
                         best_all_haar=wdsp_vals[hi];best_all_goh=wdsp_vals[gi];}
    }
    printf("+ All DSP (lbp=%d,haar=%d,goh=%d)     %5.2f%%     %+.2f\n",
           best_all_lbp,best_all_haar,best_all_goh,
           100.0*best_all_c/VAL_N,100.0*(best_all_c-baseline)/VAL_N);

    /* Phase 2 DSP: individual ablation */
    printf("\n--- Phase 2 DSP features (collar WHT, symmetry, GLCM) ---\n");
    int best_cwht_c=best_all_c,best_cwht_w=0;
    for(int wi=0;wi<n_wdsp;wi++){
        if(wdsp_vals[wi]==0)continue;
        int c=run_static_dsp(pre,nc_arr,qg,tg,nreg,w_c,w_p,w_d,w_g,sc,
                              best_all_lbp,best_all_haar,best_all_goh,
                              wdsp_vals[wi],0,0,0,VAL_N,NULL);
        if(c>best_cwht_c){best_cwht_c=c;best_cwht_w=wdsp_vals[wi];}
    }
    printf("+ Collar WHT (w=%d)                      %5.2f%%     %+.2f\n",
           best_cwht_w,100.0*best_cwht_c/VAL_N,100.0*(best_cwht_c-best_all_c)/VAL_N);

    int best_symm_c=best_all_c,best_symm_w=0;
    for(int wi=0;wi<n_wdsp;wi++){
        if(wdsp_vals[wi]==0)continue;
        int c=run_static_dsp(pre,nc_arr,qg,tg,nreg,w_c,w_p,w_d,w_g,sc,
                              best_all_lbp,best_all_haar,best_all_goh,
                              0,wdsp_vals[wi],0,0,VAL_N,NULL);
        if(c>best_symm_c){best_symm_c=c;best_symm_w=wdsp_vals[wi];}
    }
    printf("+ Symmetry (w=%d)                         %5.2f%%     %+.2f\n",
           best_symm_w,100.0*best_symm_c/VAL_N,100.0*(best_symm_c-best_all_c)/VAL_N);

    int best_glcm_c=best_all_c,best_glcm_w=0;
    for(int wi=0;wi<n_wdsp;wi++){
        if(wdsp_vals[wi]==0)continue;
        int c=run_static_dsp(pre,nc_arr,qg,tg,nreg,w_c,w_p,w_d,w_g,sc,
                              best_all_lbp,best_all_haar,best_all_goh,
                              0,0,wdsp_vals[wi],0,VAL_N,NULL);
        if(c>best_glcm_c){best_glcm_c=c;best_glcm_w=wdsp_vals[wi];}
    }
    printf("+ GLCM (w=%d)                             %5.2f%%     %+.2f\n",
           best_glcm_w,100.0*best_glcm_c/VAL_N,100.0*(best_glcm_c-best_all_c)/VAL_N);

    /* Combined phase 1 + phase 2 */
    int best_p2_c=best_all_c,bp2_cwht=0,bp2_symm=0,bp2_glcm=0;
    for(int ci=0;ci<n_wdsp;ci++)for(int si=0;si<n_wdsp;si++)for(int gi=0;gi<n_wdsp;gi++){
        if(wdsp_vals[ci]==0&&wdsp_vals[si]==0&&wdsp_vals[gi]==0)continue;
        int c=run_static_dsp(pre,nc_arr,qg,tg,nreg,w_c,w_p,w_d,w_g,sc,
                              best_all_lbp,best_all_haar,best_all_goh,
                              wdsp_vals[ci],wdsp_vals[si],wdsp_vals[gi],0,VAL_N,NULL);
        if(c>best_p2_c){best_p2_c=c;bp2_cwht=wdsp_vals[ci];bp2_symm=wdsp_vals[si];bp2_glcm=wdsp_vals[gi];}
    }
    printf("+ All P2 (cwht=%d,symm=%d,glcm=%d)   %5.2f%%     %+.2f\n",
           bp2_cwht,bp2_symm,bp2_glcm,100.0*best_p2_c/VAL_N,100.0*(best_p2_c-best_all_c)/VAL_N);

    /* ================================================================
     * PHASE 2: Best config on holdout + MNIST check
     * ================================================================ */
    printf("\n=== Best config on holdout ===\n");

    /* Holdout with best DSP weights */
    int hold_dsp=run_static_dsp(pre,nc_arr,qg,tg,nreg,w_c,w_p,w_d,w_g,sc,
                                 best_all_lbp,best_all_haar,best_all_goh,
                                 bp2_cwht,bp2_symm,bp2_glcm,
                                 HOLDOUT_START,TEST_N,preds);
    int hold_n=TEST_N-HOLDOUT_START;
    printf("  %s holdout: %.2f%% (%d/%d)\n",ds,100.0*hold_dsp/hold_n,hold_dsp,hold_n);
    report(preds,"Best DSP (holdout)",hold_dsp,HOLDOUT_START,TEST_N);

    /* Holdout with no DSP for comparison */
    int hold_base=run_static_dsp(pre,nc_arr,qg,tg,nreg,w_c,w_p,w_d,w_g,sc,
                                  0,0,0,0,0,0,HOLDOUT_START,TEST_N,NULL);
    printf("  %s holdout baseline: %.2f%%\n",ds,100.0*hold_base/hold_n);
    printf("  Delta: %+.2f pp\n\n",100.0*(hold_dsp-hold_base)/hold_n);

    /* Full 10K with best DSP */
    int full_dsp=run_static_dsp(pre,nc_arr,qg,tg,nreg,w_c,w_p,w_d,w_g,sc,
                                 best_all_lbp,best_all_haar,best_all_goh,
                                 bp2_cwht,bp2_symm,bp2_glcm,
                                 0,TEST_N,preds);
    int full_base=run_static_dsp(pre,nc_arr,qg,tg,nreg,w_c,w_p,w_d,w_g,sc,
                                  0,0,0,0,0,0,0,TEST_N,NULL);
    printf("  %s full 10K:  %.2f%% (baseline %.2f%%, delta %+.2f pp)\n\n",
           ds,100.0*full_dsp/TEST_N,100.0*full_base/TEST_N,100.0*(full_dsp-full_base)/TEST_N);
    report(preds,"Best DSP (full 10K)",full_dsp,0,TEST_N);

    /* ================================================================
     * PHASE 3: Cross-dataset check
     *
     * If we ran on Fashion, now run MNIST with same DSP weights (and vice versa)
     * by printing the command to run.
     * ================================================================ */
    if(is_fashion){
        printf("=== To verify on MNIST ===\n");
        printf("  Run: ./build/sstt_mtfp_dsp data/\n");
        printf("  (best DSP weights: lbp=%d haar=%d goh=%d)\n\n",
               best_all_lbp,best_all_haar,best_all_goh);
    } else {
        printf("=== To verify on Fashion-MNIST ===\n");
        printf("  Run: ./build/sstt_mtfp_dsp data-fashion/\n");
        printf("  (best DSP weights: lbp=%d haar=%d goh=%d)\n\n",
               best_all_lbp,best_all_haar,best_all_goh);
    }

    /* ================================================================
     * PHASE 4: Sequential processing with DSP features (val search)
     * ================================================================ */
    printf("=== PHASE 4: Bayesian Sequential + DSP (val only) ===\n");
    {
        int best_c=baseline,best_dS=0,best_Ks=0,best_tw=0;
        int dS_vals[]={1,2,4};int Ks_vals[]={10,20,50};int tw_vals[]={50,100,200};
        for(int di=0;di<3;di++)for(int ki=0;ki<3;ki++)for(int ti=0;ti<3;ti++){
            int c=run_bayesian_dsp(pre,nc_arr,qg,tg,nreg,w_c,w_p,w_d,w_g,sc,
                                    best_all_lbp,best_all_haar,best_all_goh,
                                    bp2_cwht,bp2_symm,bp2_glcm,
                                    dS_vals[di],Ks_vals[ki],tw_vals[ti],0,VAL_N,NULL);
            if(c>best_c){best_c=c;best_dS=dS_vals[di];best_Ks=Ks_vals[ki];best_tw=tw_vals[ti];}
        }
        printf("  Best Bayesian+DSP (val): dS=%d K=%d tw=%d -> %.2f%% (%+d vs static+DSP)\n",
               best_dS,best_Ks,best_tw,100.0*best_c/VAL_N,best_c-best_all_c);

        if(best_c>best_all_c){
            int hold_seq=run_bayesian_dsp(pre,nc_arr,qg,tg,nreg,w_c,w_p,w_d,w_g,sc,
                                           best_all_lbp,best_all_haar,best_all_goh,
                                           bp2_cwht,bp2_symm,bp2_glcm,
                                           best_dS,best_Ks,best_tw,HOLDOUT_START,TEST_N,preds);
            printf("  Bayesian+DSP holdout: %.2f%% (%d/%d)\n",100.0*hold_seq/hold_n,hold_seq,hold_n);
            report(preds,"Bayesian+DSP (holdout)",hold_seq,HOLDOUT_START,TEST_N);
        } else printf("  No improvement over static+DSP on val.\n\n");
    }

    printf("=== PHASE 5: Pipeline A->B + DSP (val only) ===\n");
    {
        int best_c=baseline,best_dS=0,best_Ka=0,best_tw=0,best_g=0,best_eS=0,best_p=0,best_Kb=0;
        int dS_vals[]={1,2};int Ka_vals[]={3,10};int tw_vals[]={50,100};
        int g_vals[]={2,8};int eS_vals[]={5,20};int p_vals[]={1,5};int Kb_vals[]={10,20};
        for(int di=0;di<2;di++)for(int ki=0;ki<2;ki++)for(int ti=0;ti<2;ti++)
        for(int gi=0;gi<2;gi++)for(int ei=0;ei<2;ei++)for(int pi=0;pi<2;pi++)for(int bi=0;bi<2;bi++){
            int c=run_pipeline_dsp(pre,nc_arr,qg,tg,nreg,w_c,w_p,w_d,w_g,sc,
                                    best_all_lbp,best_all_haar,best_all_goh,
                                    bp2_cwht,bp2_symm,bp2_glcm,
                                    dS_vals[di],Ka_vals[ki],tw_vals[ti],
                                    g_vals[gi],eS_vals[ei],p_vals[pi],Kb_vals[bi],0,VAL_N,NULL);
            if(c>best_c){best_c=c;best_dS=dS_vals[di];best_Ka=Ka_vals[ki];best_tw=tw_vals[ti];
                         best_g=g_vals[gi];best_eS=eS_vals[ei];best_p=p_vals[pi];best_Kb=Kb_vals[bi];}
        }
        printf("  Best Pipeline+DSP (val): dS=%d Ka=%d tw=%d g=%d eS=%d p=%d Kb=%d -> %.2f%% (%+d vs static+DSP)\n",
               best_dS,best_Ka,best_tw,best_g,best_eS,best_p,best_Kb,100.0*best_c/VAL_N,best_c-best_all_c);

        if(best_c>best_all_c){
            int hold_seq=run_pipeline_dsp(pre,nc_arr,qg,tg,nreg,w_c,w_p,w_d,w_g,sc,
                                           best_all_lbp,best_all_haar,best_all_goh,
                                           bp2_cwht,bp2_symm,bp2_glcm,
                                           best_dS,best_Ka,best_tw,
                                           best_g,best_eS,best_p,best_Kb,HOLDOUT_START,TEST_N,preds);
            printf("  Pipeline+DSP holdout: %.2f%% (%d/%d)\n",100.0*hold_seq/hold_n,hold_seq,hold_n);
            report(preds,"Pipeline+DSP (holdout)",hold_seq,HOLDOUT_START,TEST_N);
        } else printf("  No improvement over static+DSP on val.\n\n");
    }

    printf("Total: %.2f sec\n",now_sec()-t0);

    /* Cleanup */
    free(pre);free(nc_arr);free(preds);
    free(gdiv_3x3_train);free(gdiv_3x3_test);
    free(cent_train);free(cent_test);free(hprof_train);free(hprof_test);
    free(divneg_train);free(divneg_test);free(divneg_cy_train);free(divneg_cy_test);
    free(lbp_train);free(lbp_test);free(haar_train);free(haar_test);free(goh_train);free(goh_test);
    free(cwht_train);free(cwht_test);free(symm_train);free(symm_test);free(glcm_train);free(glcm_test);
    free(px_idx_pool);free(hg_idx_pool);free(vg_idx_pool);
    free(px_tr);free(px_te);free(hg_tr);free(hg_te);free(vg_tr);free(vg_te);
    free(pk_tern_train);free(pk_tern_test);free(pk_hgrad_train);free(pk_hgrad_test);free(pk_vgrad_train);free(pk_vgrad_test);
    free(raw_train_img);free(raw_test_img);free(train_labels);free(test_labels);
    return 0;
}
