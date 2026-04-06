/*
 * sstt_mtfp_ensemble.c — Multi-Threshold MTFP Ensemble
 *
 * Multiple quantization thresholds ("eyes") feeding the MTFP per-channel
 * pipeline, with candidate pools merged before ranking.  Structural features
 * computed from the primary eye (85/170) only.
 *
 * Build: gcc -O3 -mavx2 -mfma -march=native -Wall -Wextra -o build/sstt_mtfp_ensemble src/core/sstt_mtfp_ensemble.c -lm
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
#define TOP_K      500
#define HP_PAD     32
#define FG_W       30
#define FG_H       30
#define FG_SZ      (FG_W*FG_H)
#define CENT_BOTH_OPEN  50
#define CENT_DISAGREE   80
#define MAX_REGIONS 16
#define N_EYES     5

static const int thresh_lo[] = { 85, 64, 96, 75, 105 };
static const int thresh_hi[] = { 170, 192, 160, 180, 155 };

static const char *data_dir="data/";
static double now_sec(void){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return ts.tv_sec+ts.tv_nsec*1e-9;}

/* Raw data */
static uint8_t *raw_train_img,*raw_test_img,*train_labels,*test_labels;

/* Primary eye (e=0) packed trit arrays — needed for dot products */
static uint8_t *pk_tern_train,*pk_tern_test;
static uint8_t *pk_hgrad_train,*pk_hgrad_test;
static uint8_t *pk_vgrad_train,*pk_vgrad_test;

/* Per-eye block signatures */
static uint8_t *eye_px_tr[N_EYES],*eye_px_te[N_EYES];
static uint8_t *eye_hg_tr[N_EYES],*eye_hg_te[N_EYES];
static uint8_t *eye_vg_tr[N_EYES],*eye_vg_te[N_EYES];

/* Per-eye inverted indices */
static uint32_t eye_px_idx_off[N_EYES][N_BLOCKS][N_BVALS];
static uint32_t eye_hg_idx_off[N_EYES][N_BLOCKS][N_BVALS];
static uint32_t eye_vg_idx_off[N_EYES][N_BLOCKS][N_BVALS];
static uint16_t eye_px_idx_sz[N_EYES][N_BLOCKS][N_BVALS];
static uint16_t eye_hg_idx_sz[N_EYES][N_BLOCKS][N_BVALS];
static uint16_t eye_vg_idx_sz[N_EYES][N_BLOCKS][N_BVALS];
static uint32_t *eye_px_idx_pool[N_EYES],*eye_hg_idx_pool[N_EYES],*eye_vg_idx_pool[N_EYES];
static uint16_t eye_px_ig_w[N_EYES][N_BLOCKS];
static uint16_t eye_hg_ig_w[N_EYES][N_BLOCKS];
static uint16_t eye_vg_ig_w[N_EYES][N_BLOCKS];
static uint8_t eye_px_bg[N_EYES],eye_hg_bg[N_EYES],eye_vg_bg[N_EYES];

/* Trit neighbor table */
static uint8_t trit_nbr[N_BVALS][TRIT_NBR];

/* Features (primary eye only) */
static int16_t *cent_train,*cent_test,*hprof_train,*hprof_test;
static int16_t *divneg_train,*divneg_test,*divneg_cy_train,*divneg_cy_test;
static int16_t *gdiv_2x4_train,*gdiv_2x4_test;

static int *g_hist=NULL;static size_t g_hist_cap=0;

/* === Data loading === */
static uint8_t *load_idx(const char*path,uint32_t*cnt,uint32_t*ro,uint32_t*co){FILE*f=fopen(path,"rb");if(!f){fprintf(stderr,"ERR:%s\n",path);exit(1);}uint32_t m,n;if(fread(&m,4,1,f)!=1||fread(&n,4,1,f)!=1){fclose(f);exit(1);}m=__builtin_bswap32(m);n=__builtin_bswap32(n);*cnt=n;size_t s=1;if((m&0xFF)>=3){uint32_t r,c;if(fread(&r,4,1,f)!=1||fread(&c,4,1,f)!=1){fclose(f);exit(1);}r=__builtin_bswap32(r);c=__builtin_bswap32(c);if(ro)*ro=r;if(co)*co=c;s=(size_t)r*c;}else{if(ro)*ro=0;if(co)*co=0;}size_t total=(size_t)n*s;uint8_t*d=malloc(total);if(!d||fread(d,1,total,f)!=total){fclose(f);exit(1);}fclose(f);return d;}
static void load_data(void){uint32_t n,r,c;char p[256];snprintf(p,sizeof(p),"%strain-images-idx3-ubyte",data_dir);raw_train_img=load_idx(p,&n,&r,&c);snprintf(p,sizeof(p),"%strain-labels-idx1-ubyte",data_dir);train_labels=load_idx(p,&n,NULL,NULL);snprintf(p,sizeof(p),"%st10k-images-idx3-ubyte",data_dir);raw_test_img=load_idx(p,&n,&r,&c);snprintf(p,sizeof(p),"%st10k-labels-idx1-ubyte",data_dir);test_labels=load_idx(p,&n,NULL,NULL);}

/* === Quantization === */
static inline int8_t clamp_trit(int v){return v>0?1:v<0?-1:0;}

/* AVX2 quantization for primary eye (85/170) */
static void quant_tern(const uint8_t*src,int8_t*dst,int n){
    const __m256i bias=_mm256_set1_epi8((char)0x80);
    const __m256i thi=_mm256_set1_epi8((char)(170^0x80));
    const __m256i tlo=_mm256_set1_epi8((char)(85^0x80));
    const __m256i one=_mm256_set1_epi8(1);
    for(int i=0;i<n;i++){
        const uint8_t*s=src+(size_t)i*PIXELS;
        int8_t*d=dst+(size_t)i*PADDED;
        int k;
        for(k=0;k+32<=PIXELS;k+=32){
            __m256i px=_mm256_loadu_si256((const __m256i*)(s+k));
            __m256i sp=_mm256_xor_si256(px,bias);
            _mm256_storeu_si256((__m256i*)(d+k),
                _mm256_sub_epi8(
                    _mm256_and_si256(_mm256_cmpgt_epi8(sp,thi),one),
                    _mm256_and_si256(_mm256_cmpgt_epi8(tlo,sp),one)));
        }
        for(;k<PIXELS;k++)d[k]=s[k]>170?1:s[k]<85?-1:0;
        memset(d+PIXELS,0,PADDED-PIXELS);
    }
}

/* Scalar quantization with variable thresholds */
static void quant_tern_thresh(const uint8_t*src,int8_t*dst,int n,int lo,int hi){
    for(int i=0;i<n;i++){
        const uint8_t*s=src+(size_t)i*PIXELS;
        int8_t*d=dst+(size_t)i*PADDED;
        for(int k=0;k<PIXELS;k++)
            d[k]=s[k]>(uint8_t)hi?1:s[k]<(uint8_t)lo?-1:0;
        memset(d+PIXELS,0,PADDED-PIXELS);
    }
}

static void gradients(const int8_t*t,int8_t*h,int8_t*v,int n){
    for(int i=0;i<n;i++){
        const int8_t*ti=t+(size_t)i*PADDED;
        int8_t*hi=h+(size_t)i*PADDED;
        int8_t*vi=v+(size_t)i*PADDED;
        for(int y=0;y<IMG_H;y++){
            for(int x=0;x<IMG_W-1;x++)hi[y*IMG_W+x]=clamp_trit(ti[y*IMG_W+x+1]-ti[y*IMG_W+x]);
            hi[y*IMG_W+IMG_W-1]=0;
        }
        memset(hi+PIXELS,0,PADDED-PIXELS);
        for(int y=0;y<IMG_H-1;y++)
            for(int x=0;x<IMG_W;x++)vi[y*IMG_W+x]=clamp_trit(ti[(y+1)*IMG_W+x]-ti[y*IMG_W+x]);
        memset(vi+(IMG_H-1)*IMG_W,0,IMG_W);
        memset(vi+PIXELS,0,PADDED-PIXELS);
    }
}

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
static void block_sigs(const int8_t*data,uint8_t*sigs,int n){
    for(int i=0;i<n;i++){
        const int8_t*img=data+(size_t)i*PADDED;
        uint8_t*sig=sigs+(size_t)i*SIG_PAD;
        for(int y=0;y<IMG_H;y++)
            for(int s=0;s<BLKS_PER_ROW;s++){
                int b=y*IMG_W+s*3;
                sig[y*BLKS_PER_ROW+s]=benc(img[b],img[b+1],img[b+2]);
            }
        memset(sig+N_BLOCKS,0xFF,SIG_PAD-N_BLOCKS);
    }
}

/* === Information Gain === */
static void compute_ig(const uint8_t*sigs,int nv,uint8_t bgv,uint16_t*ig_out){
    int cc[N_CLASSES]={0};
    for(int i=0;i<TRAIN_N;i++)cc[train_labels[i]]++;
    double hc=0;
    for(int c=0;c<N_CLASSES;c++){double p=(double)cc[c]/TRAIN_N;if(p>0)hc-=p*log2(p);}
    double raw[N_BLOCKS],mx=0;
    for(int k=0;k<N_BLOCKS;k++){
        int*cnt=calloc((size_t)nv*N_CLASSES,sizeof(int));
        int*vt=calloc(nv,sizeof(int));
        for(int i=0;i<TRAIN_N;i++){int v=sigs[(size_t)i*SIG_PAD+k];if(v<nv){cnt[v*N_CLASSES+train_labels[i]]++;vt[v]++;}}
        double hcond=0;
        for(int v=0;v<nv;v++){
            if(!vt[v]||v==(int)bgv)continue;
            double pv=(double)vt[v]/TRAIN_N,hv=0;
            for(int c=0;c<N_CLASSES;c++){double pc=(double)cnt[v*N_CLASSES+c]/vt[v];if(pc>0)hv-=pc*log2(pc);}
            hcond+=pv*hv;
        }
        raw[k]=hc-hcond;if(raw[k]>mx)mx=raw[k];
        free(cnt);free(vt);
    }
    for(int k=0;k<N_BLOCKS;k++){ig_out[k]=mx>0?(uint16_t)(raw[k]/mx*IG_SCALE+0.5):1;if(!ig_out[k])ig_out[k]=1;}
}

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
    printf("%.1f MB  ",(double)tot*4/1048576);
}

/* === Ternary dot product (AVX2) === */
static inline int32_t tdot(const int8_t*a,const int8_t*b){
    __m256i acc=_mm256_setzero_si256();
    for(int i=0;i<PADDED;i+=32)
        acc=_mm256_add_epi8(acc,_mm256_sign_epi8(
            _mm256_load_si256((const __m256i*)(a+i)),
            _mm256_load_si256((const __m256i*)(b+i))));
    __m256i lo=_mm256_cvtepi8_epi16(_mm256_castsi256_si128(acc));
    __m256i hi=_mm256_cvtepi8_epi16(_mm256_extracti128_si256(acc,1));
    __m256i s32=_mm256_madd_epi16(_mm256_add_epi16(lo,hi),_mm256_set1_epi16(1));
    __m128i s=_mm_add_epi32(_mm256_castsi256_si128(s32),_mm256_extracti128_si256(s32,1));
    s=_mm_hadd_epi32(s,s);s=_mm_hadd_epi32(s,s);
    return _mm_cvtsi128_si32(s);
}

/* === Features (primary eye only) === */
static int16_t enclosed_centroid(const int8_t*tern){
    uint8_t grid[FG_SZ];memset(grid,0,sizeof(grid));
    for(int y=0;y<IMG_H;y++)for(int x=0;x<IMG_W;x++)grid[(y+1)*FG_W+(x+1)]=(tern[y*IMG_W+x]>0)?1:0;
    uint8_t visited[FG_SZ];memset(visited,0,sizeof(visited));
    int stack[FG_SZ];int sp=0;
    for(int y=0;y<FG_H;y++)for(int x=0;x<FG_W;x++){
        if(y==0||y==FG_H-1||x==0||x==FG_W-1){int pos=y*FG_W+x;if(!grid[pos]&&!visited[pos]){visited[pos]=1;stack[sp++]=pos;}}
    }
    while(sp>0){int pos=stack[--sp];int py=pos/FG_W,px2=pos%FG_W;const int dx[]={0,0,1,-1},dy[]={1,-1,0,0};for(int d=0;d<4;d++){int ny=py+dy[d],nx=px2+dx[d];if(ny<0||ny>=FG_H||nx<0||nx>=FG_W)continue;int npos=ny*FG_W+nx;if(!visited[npos]&&!grid[npos]){visited[npos]=1;stack[sp++]=npos;}}}
    int sum_y=0,count=0;
    for(int y=1;y<FG_H-1;y++)for(int x=1;x<FG_W-1;x++){int pos=y*FG_W+x;if(!grid[pos]&&!visited[pos]){sum_y+=(y-1);count++;}}
    return count>0?(int16_t)(sum_y/count):-1;
}

static void hprofile(const int8_t*tern,int16_t*prof){
    for(int y=0;y<IMG_H;y++){int c=0;for(int x=0;x<IMG_W;x++)c+=(tern[y*IMG_W+x]>0);prof[y]=(int16_t)c;}
    for(int y=IMG_H;y<HP_PAD;y++)prof[y]=0;
}

static void div_features(const int8_t*hg,const int8_t*vg,int16_t*ns,int16_t*cy){
    int neg_sum=0,neg_ysum=0,neg_cnt=0;
    for(int y=0;y<IMG_H;y++)for(int x=0;x<IMG_W;x++){
        int dh=(int)hg[y*IMG_W+x]-(x>0?(int)hg[y*IMG_W+x-1]:0);
        int dv=(int)vg[y*IMG_W+x]-(y>0?(int)vg[(y-1)*IMG_W+x]:0);
        int d=dh+dv;if(d<0){neg_sum+=d;neg_ysum+=y;neg_cnt++;}
    }
    *ns=(int16_t)(neg_sum<-32767?-32767:neg_sum);
    *cy=neg_cnt>0?(int16_t)(neg_ysum/neg_cnt):-1;
}

static void grid_div(const int8_t*hg,const int8_t*vg,int grow,int gcol,int16_t*out){
    int nr=grow*gcol;int regions[MAX_REGIONS];memset(regions,0,sizeof(int)*nr);
    for(int y=0;y<IMG_H;y++)for(int x=0;x<IMG_W;x++){
        int dh=(int)hg[y*IMG_W+x]-(x>0?(int)hg[y*IMG_W+x-1]:0);
        int dv=(int)vg[y*IMG_W+x]-(y>0?(int)vg[(y-1)*IMG_W+x]:0);
        int d=dh+dv;if(d<0){int ry=y*grow/IMG_H;int rx=x*gcol/IMG_W;if(ry>=grow)ry=grow-1;if(rx>=gcol)rx=gcol-1;regions[ry*gcol+rx]+=d;}
    }
    for(int i=0;i<nr;i++)out[i]=(int16_t)(regions[i]<-32767?-32767:regions[i]);
}

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

/* === Single-eye vote (for baseline) === */
static void vote_single_eye(uint32_t*votes,int img,int eye){
    memset(votes,0,TRAIN_N*sizeof(uint32_t));
    const uint8_t*ps=eye_px_te[eye]+(size_t)img*SIG_PAD;
    for(int k=0;k<N_BLOCKS;k++){
        uint8_t bv=ps[k];if(bv==eye_px_bg[eye]||bv>=N_BVALS)continue;
        uint16_t w=eye_px_ig_w[eye][k],wh=w>1?w/2:1;
        {uint32_t off=eye_px_idx_off[eye][k][bv];uint16_t sz=eye_px_idx_sz[eye][k][bv];const uint32_t*ids=eye_px_idx_pool[eye]+off;for(uint16_t j=0;j<sz;j++)votes[ids[j]]+=w;}
        for(int nb=0;nb<TRIT_NBR;nb++){uint8_t nv=trit_nbr[bv][nb];if(nv==0xFF||nv==eye_px_bg[eye])continue;uint32_t noff=eye_px_idx_off[eye][k][nv];uint16_t nsz=eye_px_idx_sz[eye][k][nv];const uint32_t*nids=eye_px_idx_pool[eye]+noff;for(uint16_t j=0;j<nsz;j++)votes[nids[j]]+=wh;}
    }
    const uint8_t*hs=eye_hg_te[eye]+(size_t)img*SIG_PAD;
    for(int k=0;k<N_BLOCKS;k++){
        uint8_t bv=hs[k];if(bv==eye_hg_bg[eye]||bv>=N_BVALS)continue;
        uint16_t w=eye_hg_ig_w[eye][k],wh=w>1?w/2:1;
        {uint32_t off=eye_hg_idx_off[eye][k][bv];uint16_t sz=eye_hg_idx_sz[eye][k][bv];const uint32_t*ids=eye_hg_idx_pool[eye]+off;for(uint16_t j=0;j<sz;j++)votes[ids[j]]+=w;}
        for(int nb=0;nb<TRIT_NBR;nb++){uint8_t nv=trit_nbr[bv][nb];if(nv==0xFF||nv==eye_hg_bg[eye])continue;uint32_t noff=eye_hg_idx_off[eye][k][nv];uint16_t nsz=eye_hg_idx_sz[eye][k][nv];const uint32_t*nids=eye_hg_idx_pool[eye]+noff;for(uint16_t j=0;j<nsz;j++)votes[nids[j]]+=wh;}
    }
    const uint8_t*vs=eye_vg_te[eye]+(size_t)img*SIG_PAD;
    for(int k=0;k<N_BLOCKS;k++){
        uint8_t bv=vs[k];if(bv==eye_vg_bg[eye]||bv>=N_BVALS)continue;
        uint16_t w=eye_vg_ig_w[eye][k],wh=w>1?w/2:1;
        {uint32_t off=eye_vg_idx_off[eye][k][bv];uint16_t sz=eye_vg_idx_sz[eye][k][bv];const uint32_t*ids=eye_vg_idx_pool[eye]+off;for(uint16_t j=0;j<sz;j++)votes[ids[j]]+=w;}
        for(int nb=0;nb<TRIT_NBR;nb++){uint8_t nv=trit_nbr[bv][nb];if(nv==0xFF||nv==eye_vg_bg[eye])continue;uint32_t noff=eye_vg_idx_off[eye][k][nv];uint16_t nsz=eye_vg_idx_sz[eye][k][nv];const uint32_t*nids=eye_vg_idx_pool[eye]+noff;for(uint16_t j=0;j<nsz;j++)votes[nids[j]]+=wh;}
    }
}

/* === Multi-eye ensemble vote === */
static void vote_ensemble(uint32_t*votes,int img,int n_eyes){
    memset(votes,0,TRAIN_N*sizeof(uint32_t));
    for(int e=0;e<n_eyes;e++){
        /* PX channel */
        const uint8_t*ps=eye_px_te[e]+(size_t)img*SIG_PAD;
        for(int k=0;k<N_BLOCKS;k++){
            uint8_t bv=ps[k];if(bv==eye_px_bg[e]||bv>=N_BVALS)continue;
            uint16_t w=eye_px_ig_w[e][k],wh=w>1?w/2:1;
            {uint32_t off=eye_px_idx_off[e][k][bv];uint16_t sz=eye_px_idx_sz[e][k][bv];const uint32_t*ids=eye_px_idx_pool[e]+off;for(uint16_t j=0;j<sz;j++)votes[ids[j]]+=w;}
            for(int nb=0;nb<TRIT_NBR;nb++){uint8_t nv=trit_nbr[bv][nb];if(nv==0xFF||nv==eye_px_bg[e])continue;uint32_t noff=eye_px_idx_off[e][k][nv];uint16_t nsz=eye_px_idx_sz[e][k][nv];const uint32_t*nids=eye_px_idx_pool[e]+noff;for(uint16_t j=0;j<nsz;j++)votes[nids[j]]+=wh;}
        }
        /* HG channel */
        const uint8_t*hs=eye_hg_te[e]+(size_t)img*SIG_PAD;
        for(int k=0;k<N_BLOCKS;k++){
            uint8_t bv=hs[k];if(bv==eye_hg_bg[e]||bv>=N_BVALS)continue;
            uint16_t w=eye_hg_ig_w[e][k],wh=w>1?w/2:1;
            {uint32_t off=eye_hg_idx_off[e][k][bv];uint16_t sz=eye_hg_idx_sz[e][k][bv];const uint32_t*ids=eye_hg_idx_pool[e]+off;for(uint16_t j=0;j<sz;j++)votes[ids[j]]+=w;}
            for(int nb=0;nb<TRIT_NBR;nb++){uint8_t nv=trit_nbr[bv][nb];if(nv==0xFF||nv==eye_hg_bg[e])continue;uint32_t noff=eye_hg_idx_off[e][k][nv];uint16_t nsz=eye_hg_idx_sz[e][k][nv];const uint32_t*nids=eye_hg_idx_pool[e]+noff;for(uint16_t j=0;j<nsz;j++)votes[nids[j]]+=wh;}
        }
        /* VG channel */
        const uint8_t*vs=eye_vg_te[e]+(size_t)img*SIG_PAD;
        for(int k=0;k<N_BLOCKS;k++){
            uint8_t bv=vs[k];if(bv==eye_vg_bg[e]||bv>=N_BVALS)continue;
            uint16_t w=eye_vg_ig_w[e][k],wh=w>1?w/2:1;
            {uint32_t off=eye_vg_idx_off[e][k][bv];uint16_t sz=eye_vg_idx_sz[e][k][bv];const uint32_t*ids=eye_vg_idx_pool[e]+off;for(uint16_t j=0;j<sz;j++)votes[ids[j]]+=w;}
            for(int nb=0;nb<TRIT_NBR;nb++){uint8_t nv=trit_nbr[bv][nb];if(nv==0xFF||nv==eye_vg_bg[e])continue;uint32_t noff=eye_vg_idx_off[e][k][nv];uint16_t nsz=eye_vg_idx_sz[e][k][nv];const uint32_t*nids=eye_vg_idx_pool[e]+noff;for(uint16_t j=0;j<nsz;j++)votes[nids[j]]+=wh;}
        }
    }
}

/* === Candidate selection and scoring === */
typedef struct {
    uint32_t id,votes;
    int32_t dot_px,dot_vg;
    int32_t div_sim,cent_sim,prof_sim;
    int32_t gdiv_sim;
    int64_t combined;
} cand_t;

static int cmp_votes_d(const void*a,const void*b){return(int)((const cand_t*)b)->votes-(int)((const cand_t*)a)->votes;}
static int cmp_comb_d(const void*a,const void*b){int64_t da=((const cand_t*)a)->combined,db=((const cand_t*)b)->combined;return(db>da)-(db<da);}

static int select_top_k(const uint32_t*votes,int n,cand_t*out,int k){
    uint32_t mx=0;for(int i=0;i<n;i++)if(votes[i]>mx)mx=votes[i];
    if(!mx)return 0;
    if((size_t)(mx+1)>g_hist_cap){g_hist_cap=(size_t)(mx+1)+4096;free(g_hist);g_hist=malloc(g_hist_cap*sizeof(int));}
    memset(g_hist,0,(mx+1)*sizeof(int));
    for(int i=0;i<n;i++)if(votes[i])g_hist[votes[i]]++;
    int cum=0,thr;for(thr=(int)mx;thr>=1;thr--){cum+=g_hist[thr];if(cum>=k)break;}
    if(thr<1)thr=1;
    int nc=0;for(int i=0;i<n&&nc<k;i++)if(votes[i]>=(uint32_t)thr){out[nc]=(cand_t){0};out[nc].id=(uint32_t)i;out[nc].votes=votes[i];nc++;}
    qsort(out,(size_t)nc,sizeof(cand_t),cmp_votes_d);
    return nc;
}

/* Compute structural features using PRIMARY eye packed arrays */
static void compute_base(cand_t*cands,int nc,int ti,int16_t q_cent,const int16_t*q_prof,int16_t q_dn,int16_t q_dc){
    int8_t tp[PADDED] __attribute__((aligned(32)));
    int8_t tv[PADDED] __attribute__((aligned(32)));
    unpack_trits(pk_tern_test+(size_t)ti*PACKED_PAD,tp);
    unpack_trits(pk_vgrad_test+(size_t)ti*PACKED_PAD,tv);
    for(int j=0;j<nc;j++){uint32_t id=cands[j].id;
        int8_t tr[PADDED] __attribute__((aligned(32)));
        int8_t vr[PADDED] __attribute__((aligned(32)));
        unpack_trits(pk_tern_train+(size_t)id*PACKED_PAD,tr);
        unpack_trits(pk_vgrad_train+(size_t)id*PACKED_PAD,vr);
        cands[j].dot_px=tdot(tp,tr);
        cands[j].dot_vg=tdot(tv,vr);
        int16_t cc=cent_train[id];
        if(q_cent<0&&cc<0)cands[j].cent_sim=CENT_BOTH_OPEN;
        else if(q_cent<0||cc<0)cands[j].cent_sim=-CENT_DISAGREE;
        else cands[j].cent_sim=-(int32_t)abs(q_cent-cc);
        const int16_t*cp=hprof_train+(size_t)id*HP_PAD;
        int32_t pd=0;for(int y=0;y<IMG_H;y++)pd+=(int32_t)q_prof[y]*cp[y];
        cands[j].prof_sim=pd;
        int16_t cdn=divneg_train[id],cdc=divneg_cy_train[id];
        int32_t ds=-(int32_t)abs(q_dn-cdn);
        if(q_dc>=0&&cdc>=0)ds-=(int32_t)abs(q_dc-cdc)*2;
        else if((q_dc<0)!=(cdc<0))ds-=10;
        cands[j].div_sim=ds;
    }
}

static int knn_vote(const cand_t*c,int nc,int k){
    int v[N_CLASSES]={0};if(k>nc)k=nc;
    for(int i=0;i<k;i++)v[train_labels[c[i].id]]++;
    int best=0;for(int c2=1;c2<N_CLASSES;c2++)if(v[c2]>v[best])best=c2;
    return best;
}

static int cmp_i16(const void*a,const void*b){return *(const int16_t*)a-*(const int16_t*)b;}
static int compute_mad(const cand_t*cands,int nc){
    if(nc<3)return 1000;
    int16_t *vals=malloc((size_t)nc*sizeof(int16_t));
    for(int j=0;j<nc;j++)vals[j]=divneg_train[cands[j].id];
    qsort(vals,(size_t)nc,sizeof(int16_t),cmp_i16);
    int16_t med=vals[nc/2];
    int16_t *devs=malloc((size_t)nc*sizeof(int16_t));
    for(int j=0;j<nc;j++)devs[j]=(int16_t)abs(vals[j]-med);
    qsort(devs,(size_t)nc,sizeof(int16_t),cmp_i16);
    int result=devs[nc/2]>0?devs[nc/2]:1;
    free(vals);free(devs);
    return result;
}

/* === Classify with hardcoded weights === */
static int classify_cands(cand_t*cands,int nc,int ti){
    /* grid=2x4, w_c=50, w_p=16, w_d=200, w_g=100, sc=50 */
    const int w_c=50,w_p=16,w_d=200,w_g=100,sc=50;
    const int16_t*qi=gdiv_2x4_test+(size_t)ti*MAX_REGIONS;
    for(int j=0;j<nc;j++){
        const int16_t*ci=gdiv_2x4_train+(size_t)cands[j].id*MAX_REGIONS;
        int32_t l=0;for(int r=0;r<8;r++)l+=abs(qi[r]-ci[r]);
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

/* ================================================================
 *  Main
 * ================================================================ */
int main(int argc,char**argv){
    double t0=now_sec();
    if(argc>1){data_dir=argv[1];size_t l=strlen(data_dir);if(l&&data_dir[l-1]!='/'){char*buf=malloc(l+2);memcpy(buf,data_dir,l);buf[l]='/';buf[l+1]='\0';data_dir=buf;}}

    printf("=== SSTT MTFP Ensemble: Multi-Threshold Retrieval ===\n\n");

    /* 1. Load raw data */
    load_data();

    /* 2. Init trit neighbor table */
    init_trit_nbr();

    /* 3. Allocate reusable temp buffers for quantization/gradients */
    int8_t *tern_train=aligned_alloc(32,(size_t)TRAIN_N*PADDED);
    int8_t *tern_test =aligned_alloc(32,(size_t)TEST_N*PADDED);
    int8_t *hgrad_train=aligned_alloc(32,(size_t)TRAIN_N*PADDED);
    int8_t *hgrad_test =aligned_alloc(32,(size_t)TEST_N*PADDED);
    int8_t *vgrad_train=aligned_alloc(32,(size_t)TRAIN_N*PADDED);
    int8_t *vgrad_test =aligned_alloc(32,(size_t)TEST_N*PADDED);

    printf("Building eyes...\n");

    /* 4. Build each eye */
    for(int e=0;e<N_EYES;e++){
        double te=now_sec();

        /* Quantize */
        if(e==0)
            quant_tern(raw_train_img,tern_train,TRAIN_N);
        else
            quant_tern_thresh(raw_train_img,tern_train,TRAIN_N,thresh_lo[e],thresh_hi[e]);
        if(e==0)
            quant_tern(raw_test_img,tern_test,TEST_N);
        else
            quant_tern_thresh(raw_test_img,tern_test,TEST_N,thresh_lo[e],thresh_hi[e]);

        /* Gradients */
        gradients(tern_train,hgrad_train,vgrad_train,TRAIN_N);
        gradients(tern_test,hgrad_test,vgrad_test,TEST_N);

        /* Block signatures (persist) */
        eye_px_tr[e]=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD);
        eye_px_te[e]=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
        eye_hg_tr[e]=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD);
        eye_hg_te[e]=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
        eye_vg_tr[e]=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD);
        eye_vg_te[e]=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
        block_sigs(tern_train,eye_px_tr[e],TRAIN_N);block_sigs(tern_test,eye_px_te[e],TEST_N);
        block_sigs(hgrad_train,eye_hg_tr[e],TRAIN_N);block_sigs(hgrad_test,eye_hg_te[e],TEST_N);
        block_sigs(vgrad_train,eye_vg_tr[e],TRAIN_N);block_sigs(vgrad_test,eye_vg_te[e],TEST_N);

        /* Pack trits for primary eye only (needed for dot products) */
        if(e==0){
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
        }

        /* Build per-channel inverted indices */
        printf("  Eye %d (%d/%d): px=",e,thresh_lo[e],thresh_hi[e]);
        build_channel_index(eye_px_tr[e],&eye_px_bg[e],eye_px_ig_w[e],
            eye_px_idx_off[e],eye_px_idx_sz[e],&eye_px_idx_pool[e]);
        printf("hg=");
        build_channel_index(eye_hg_tr[e],&eye_hg_bg[e],eye_hg_ig_w[e],
            eye_hg_idx_off[e],eye_hg_idx_sz[e],&eye_hg_idx_pool[e]);
        printf("vg=");
        build_channel_index(eye_vg_tr[e],&eye_vg_bg[e],eye_vg_ig_w[e],
            eye_vg_idx_off[e],eye_vg_idx_sz[e],&eye_vg_idx_pool[e]);
        printf(" (%.1fs)\n",now_sec()-te);
    }

    /* 5. Compute features from primary eye */
    printf("\nComputing features (primary eye)...\n");double tf=now_sec();
    cent_train=malloc((size_t)TRAIN_N*2);cent_test=malloc((size_t)TEST_N*2);
    hprof_train=aligned_alloc(32,(size_t)TRAIN_N*HP_PAD*2);hprof_test=aligned_alloc(32,(size_t)TEST_N*HP_PAD*2);
    divneg_train=malloc((size_t)TRAIN_N*2);divneg_test=malloc((size_t)TEST_N*2);
    divneg_cy_train=malloc((size_t)TRAIN_N*2);divneg_cy_test=malloc((size_t)TEST_N*2);
    gdiv_2x4_train=malloc((size_t)TRAIN_N*MAX_REGIONS*2);gdiv_2x4_test=malloc((size_t)TEST_N*MAX_REGIONS*2);

    compute_centroid_all_pk(pk_tern_train,cent_train,TRAIN_N,PACKED_PAD);
    compute_centroid_all_pk(pk_tern_test,cent_test,TEST_N,PACKED_PAD);
    compute_hprof_all_pk(pk_tern_train,hprof_train,TRAIN_N,PACKED_PAD);
    compute_hprof_all_pk(pk_tern_test,hprof_test,TEST_N,PACKED_PAD);
    compute_div_all_pk(pk_hgrad_train,pk_vgrad_train,divneg_train,divneg_cy_train,TRAIN_N,PACKED_PAD);
    compute_div_all_pk(pk_hgrad_test,pk_vgrad_test,divneg_test,divneg_cy_test,TEST_N,PACKED_PAD);
    compute_grid_div_pk(pk_hgrad_train,pk_vgrad_train,2,4,gdiv_2x4_train,TRAIN_N,PACKED_PAD);
    compute_grid_div_pk(pk_hgrad_test,pk_vgrad_test,2,4,gdiv_2x4_test,TEST_N,PACKED_PAD);
    printf("  Features: %.2f sec\n",now_sec()-tf);

    /* 6. Free temp int8 arrays */
    free(tern_train);free(tern_test);
    free(hgrad_train);free(hgrad_test);
    free(vgrad_train);free(vgrad_test);

    /* === Run experiments === */
    printf("\n%-35s %8s  %7s  %6s\n","Method","Accuracy","Correct","Errors");
    printf("%-35s %8s  %7s  %6s\n","------","--------","-------","------");

    uint32_t*votes=calloc(TRAIN_N,sizeof(uint32_t));
    cand_t*cands=malloc((size_t)TOP_K*sizeof(cand_t));

    /* --- 1. Single-eye MTFP K=500 (baseline) --- */
    {
        double ts=now_sec();
        int correct=0;
        for(int i=0;i<TEST_N;i++){
            vote_single_eye(votes,i,0);
            int nc=select_top_k(votes,TRAIN_N,cands,TOP_K);
            compute_base(cands,nc,i,cent_test[i],hprof_test+(size_t)i*HP_PAD,divneg_test[i],divneg_cy_test[i]);
            int pred=classify_cands(cands,nc,i);
            if(pred==test_labels[i])correct++;
            if((i+1)%2000==0)fprintf(stderr,"  single-eye: %d/%d\r",i+1,TEST_N);
        }
        fprintf(stderr,"                              \r");
        printf("%-35s %7.2f%%  %7d  %6d  (%.1fs)\n","Single-eye K=500 (baseline)",
            100.0*correct/TEST_N,correct,TEST_N-correct,now_sec()-ts);
    }

    /* --- 2. 3-eye ensemble K=500 --- */
    {
        double ts=now_sec();
        int correct=0;
        for(int i=0;i<TEST_N;i++){
            vote_ensemble(votes,i,3);
            int nc=select_top_k(votes,TRAIN_N,cands,TOP_K);
            compute_base(cands,nc,i,cent_test[i],hprof_test+(size_t)i*HP_PAD,divneg_test[i],divneg_cy_test[i]);
            int pred=classify_cands(cands,nc,i);
            if(pred==test_labels[i])correct++;
            if((i+1)%2000==0)fprintf(stderr,"  3-eye: %d/%d\r",i+1,TEST_N);
        }
        fprintf(stderr,"                              \r");
        printf("%-35s %7.2f%%  %7d  %6d  (%.1fs)\n","3-eye ensemble K=500",
            100.0*correct/TEST_N,correct,TEST_N-correct,now_sec()-ts);
    }

    /* --- 3. 5-eye ensemble K=200 --- */
    {
        double ts=now_sec();
        int correct=0;
        int k=200;
        for(int i=0;i<TEST_N;i++){
            vote_ensemble(votes,i,5);
            int nc=select_top_k(votes,TRAIN_N,cands,k);
            compute_base(cands,nc,i,cent_test[i],hprof_test+(size_t)i*HP_PAD,divneg_test[i],divneg_cy_test[i]);
            int pred=classify_cands(cands,nc,i);
            if(pred==test_labels[i])correct++;
            if((i+1)%2000==0)fprintf(stderr,"  5-eye K=200: %d/%d\r",i+1,TEST_N);
        }
        fprintf(stderr,"                              \r");
        printf("%-35s %7.2f%%  %7d  %6d  (%.1fs)\n","5-eye ensemble K=200",
            100.0*correct/TEST_N,correct,TEST_N-correct,now_sec()-ts);
    }

    /* --- 4. 5-eye ensemble K=500 --- */
    {
        double ts=now_sec();
        int correct=0;
        for(int i=0;i<TEST_N;i++){
            vote_ensemble(votes,i,5);
            int nc=select_top_k(votes,TRAIN_N,cands,TOP_K);
            compute_base(cands,nc,i,cent_test[i],hprof_test+(size_t)i*HP_PAD,divneg_test[i],divneg_cy_test[i]);
            int pred=classify_cands(cands,nc,i);
            if(pred==test_labels[i])correct++;
            if((i+1)%2000==0)fprintf(stderr,"  5-eye K=500: %d/%d\r",i+1,TEST_N);
        }
        fprintf(stderr,"                              \r");
        printf("%-35s %7.2f%%  %7d  %6d  (%.1fs)\n","5-eye ensemble K=500",
            100.0*correct/TEST_N,correct,TEST_N-correct,now_sec()-ts);
    }

    /* --- 5. 5-eye ensemble K=1000 --- */
    {
        double ts=now_sec();
        int correct=0;
        int k=1000;
        cand_t*cands_big=malloc((size_t)k*sizeof(cand_t));
        for(int i=0;i<TEST_N;i++){
            vote_ensemble(votes,i,5);
            int nc=select_top_k(votes,TRAIN_N,cands_big,k);
            compute_base(cands_big,nc,i,cent_test[i],hprof_test+(size_t)i*HP_PAD,divneg_test[i],divneg_cy_test[i]);
            /* Inline classify with bigger buffer */
            const int w_c=50,w_p=16,w_d=200,w_g=100,sc=50;
            const int16_t*qi=gdiv_2x4_test+(size_t)i*MAX_REGIONS;
            for(int j=0;j<nc;j++){
                const int16_t*ci=gdiv_2x4_train+(size_t)cands_big[j].id*MAX_REGIONS;
                int32_t l=0;for(int r=0;r<8;r++)l+=abs(qi[r]-ci[r]);
                cands_big[j].gdiv_sim=-l;
            }
            int mad=compute_mad(cands_big,nc);int64_t s2=(int64_t)sc;
            int wc=(int)(s2*w_c/(s2+mad)),wp=(int)(s2*w_p/(s2+mad)),wd=(int)(s2*w_d/(s2+mad)),wg=(int)(s2*w_g/(s2+mad));
            for(int j=0;j<nc;j++)
                cands_big[j].combined=(int64_t)256*cands_big[j].dot_px+(int64_t)192*cands_big[j].dot_vg
                    +(int64_t)wc*cands_big[j].cent_sim+(int64_t)wp*cands_big[j].prof_sim
                    +(int64_t)wd*cands_big[j].div_sim+(int64_t)wg*cands_big[j].gdiv_sim;
            qsort(cands_big,(size_t)nc,sizeof(cand_t),cmp_comb_d);
            int pred=knn_vote(cands_big,nc,3);
            if(pred==test_labels[i])correct++;
            if((i+1)%2000==0)fprintf(stderr,"  5-eye K=1000: %d/%d\r",i+1,TEST_N);
        }
        fprintf(stderr,"                              \r");
        printf("%-35s %7.2f%%  %7d  %6d  (%.1fs)\n","5-eye ensemble K=1000",
            100.0*correct/TEST_N,correct,TEST_N-correct,now_sec()-ts);
        free(cands_big);
    }

    printf("\nTotal: %.2f sec\n",now_sec()-t0);

    /* Cleanup */
    free(votes);free(cands);free(g_hist);
    for(int e=0;e<N_EYES;e++){
        free(eye_px_tr[e]);free(eye_px_te[e]);
        free(eye_hg_tr[e]);free(eye_hg_te[e]);
        free(eye_vg_tr[e]);free(eye_vg_te[e]);
        free(eye_px_idx_pool[e]);free(eye_hg_idx_pool[e]);free(eye_vg_idx_pool[e]);
    }
    free(pk_tern_train);free(pk_tern_test);
    free(pk_hgrad_train);free(pk_hgrad_test);
    free(pk_vgrad_train);free(pk_vgrad_test);
    free(cent_train);free(cent_test);
    free(hprof_train);free(hprof_test);
    free(divneg_train);free(divneg_test);
    free(divneg_cy_train);free(divneg_cy_test);
    free(gdiv_2x4_train);free(gdiv_2x4_test);
    free(raw_train_img);free(raw_test_img);
    free(train_labels);free(test_labels);
    return 0;
}
