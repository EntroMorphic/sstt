/*
 * sstt_mtfp.c — Multi-Trit Fixed Point: Native Ternary Classification
 *
 * First SSTT variant using packed ternary representation (2 bits/trit,
 * 4 trits/byte) instead of binary-stored ternary.  Fixes the multi-probe
 * topology bug by using per-channel inverted indices with trit-flip
 * neighbors (base-27) instead of joint byte-flip neighbors (base-256).
 *
 * Same val/holdout methodology as topo9_val.
 *
 * Build: gcc -O3 -mavx2 -mfma -march=native -Wall -Wextra -o build/sstt_mtfp src/core/sstt_mtfp.c -lm
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
#define PACKED_PAD 224  /* ceil(800/4)=200, padded to 224 for 32-byte AVX2 alignment */
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

/* Features */
static int16_t *cent_train,*cent_test,*hprof_train,*hprof_test;
static int16_t *divneg_train,*divneg_test,*divneg_cy_train,*divneg_cy_test;
static int16_t *gdiv_2x4_train,*gdiv_2x4_test;
static int16_t *gdiv_3x3_train,*gdiv_3x3_test;

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
static void pack_trits_single(const int8_t*src,uint8_t*dst,int src_stride,int dst_stride){
    pack_trits(src,dst,1,src_stride,dst_stride);
}
static void unpack_trits(const uint8_t*packed,int8_t*out){
    static const int8_t lut[4]={0,1,0,-1};
    int n_bytes=PIXELS/4; /* 784/4=196, only unpack actual pixel bytes */
    for(int i=0;i<n_bytes;i++){
        uint8_t b=packed[i];
        out[i*4+0]=lut[b&3];
        out[i*4+1]=lut[(b>>2)&3];
        out[i*4+2]=lut[(b>>4)&3];
        out[i*4+3]=lut[(b>>6)&3];
    }
    memset(out+PIXELS,0,PADDED-PIXELS); /* zero padding for AVX2 alignment */
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

/* === Features (computed from packed arrays via unpack) === */
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
    }
}

static int knn_vote(const cand_t*c,int nc,int k){int v[N_CLASSES]={0};if(k>nc)k=nc;for(int i=0;i<k;i++)v[train_labels[c[i].id]]++;int best=0;for(int c2=1;c2<N_CLASSES;c2++)if(v[c2]>v[best])best=c2;return best;}
static int cmp_i16(const void*a,const void*b){return *(const int16_t*)a-*(const int16_t*)b;}
static int compute_mad(const cand_t*cands,int nc){if(nc<3)return 1000;int16_t vals[TOP_K];for(int j=0;j<nc;j++)vals[j]=divneg_train[cands[j].id];qsort(vals,(size_t)nc,sizeof(int16_t),cmp_i16);int16_t med=vals[nc/2];int16_t devs[TOP_K];for(int j=0;j<nc;j++)devs[j]=(int16_t)abs(vals[j]-med);qsort(devs,(size_t)nc,sizeof(int16_t),cmp_i16);return devs[nc/2]>0?devs[nc/2]:1;}

/* === Static sort-then-vote === */
static int run_static(const cand_t*pre,const int*nc_arr,
                       const int16_t*qg,const int16_t*tg,int nreg,
                       int w_c,int w_p,int w_d,int w_g,int sc,
                       int start,int end,uint8_t*preds){
    int correct=0;
    for(int i=start;i<end;i++){
        cand_t cands[TOP_K];int nc=nc_arr[i];
        memcpy(cands,pre+(size_t)i*TOP_K,(size_t)nc*sizeof(cand_t));
        const int16_t*qi=qg+(size_t)i*MAX_REGIONS;
        for(int j=0;j<nc;j++){const int16_t*ci=tg+(size_t)cands[j].id*MAX_REGIONS;int32_t l=0;for(int r=0;r<nreg;r++)l+=abs(qi[r]-ci[r]);cands[j].gdiv_sim=-l;}
        int mad=compute_mad(cands,nc);int64_t s=(int64_t)sc;
        int wc=(int)(s*w_c/(s+mad)),wp=(int)(s*w_p/(s+mad)),wd=(int)(s*w_d/(s+mad)),wg=(int)(s*w_g/(s+mad));
        for(int j=0;j<nc;j++)cands[j].combined=(int64_t)256*cands[j].dot_px+(int64_t)192*cands[j].dot_vg+(int64_t)wc*cands[j].cent_sim+(int64_t)wp*cands[j].prof_sim+(int64_t)wd*cands[j].div_sim+(int64_t)wg*cands[j].gdiv_sim;
        qsort(cands,(size_t)nc,sizeof(cand_t),cmp_comb_d);
        int pred=knn_vote(cands,nc,3);
        if(preds)preds[i]=(uint8_t)pred;
        if(pred==test_labels[i])correct++;
    }
    return correct;
}

/* === Bayesian Sequential === */
static int run_bayesian_seq(const cand_t*pre,const int*nc_arr,
                             const int16_t*qg,const int16_t*tg,int nreg,
                             int w_c,int w_p,int w_d,int w_g,int sc,
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
        for(int j=0;j<nc;j++)cands[j].combined=(int64_t)256*cands[j].dot_px+(int64_t)192*cands[j].dot_vg+(int64_t)wc*cands[j].cent_sim+(int64_t)wp*cands[j].prof_sim+(int64_t)wd*cands[j].div_sim+(int64_t)wg*cands[j].gdiv_sim;
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

/* === CfC Sequential === */
static int run_cfc(const cand_t*pre,const int*nc_arr,
                    const int16_t*qg,const int16_t*tg,int nreg,
                    int w_c,int w_p,int w_d,int w_g,int sc,
                    int gain,int exp_S,int penalty,int K_seq,
                    int start,int end,uint8_t*preds){
    int correct=0;
    for(int i=start;i<end;i++){
        cand_t cands[TOP_K];int nc=nc_arr[i];
        memcpy(cands,pre+(size_t)i*TOP_K,(size_t)nc*sizeof(cand_t));
        const int16_t*qi=qg+(size_t)i*MAX_REGIONS;
        for(int j=0;j<nc;j++){const int16_t*ci=tg+(size_t)cands[j].id*MAX_REGIONS;int32_t l=0;for(int r=0;r<nreg;r++)l+=abs(qi[r]-ci[r]);cands[j].gdiv_sim=-l;}
        int mad=compute_mad(cands,nc);int64_t s=(int64_t)sc;
        int wc=(int)(s*w_c/(s+mad)),wp=(int)(s*w_p/(s+mad)),wd=(int)(s*w_d/(s+mad)),wg=(int)(s*w_g/(s+mad));
        for(int j=0;j<nc;j++)cands[j].combined=(int64_t)256*cands[j].dot_px+(int64_t)192*cands[j].dot_vg+(int64_t)wc*cands[j].cent_sim+(int64_t)wp*cands[j].prof_sim+(int64_t)wd*cands[j].div_sim+(int64_t)wg*cands[j].gdiv_sim;
        qsort(cands,(size_t)nc,sizeof(cand_t),cmp_comb_d);
        int64_t h[N_CLASSES];memset(h,0,sizeof(h));
        int ks=K_seq<nc?K_seq:nc;
        for(int j=0;j<ks;j++){uint8_t lbl=train_labels[cands[j].id];int64_t evidence=cands[j].combined/256;int64_t decay_num=(int64_t)exp_S;int64_t decay_den=(int64_t)exp_S+1;for(int c=0;c<N_CLASSES;c++){int64_t input=(c==lbl)?(int64_t)gain*evidence:-(int64_t)penalty;h[c]=(decay_num*h[c]+input)/decay_den;}}
        int pred=0;for(int c=1;c<N_CLASSES;c++)if(h[c]>h[pred])pred=c;
        if(preds)preds[i]=(uint8_t)pred;
        if(pred==test_labels[i])correct++;
    }
    return correct;
}

/* === Pipeline A->B === */
static int run_pipeline(const cand_t*pre,const int*nc_arr,
                         const int16_t*qg,const int16_t*tg,int nreg,
                         int w_c,int w_p,int w_d,int w_g,int sc,
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
        for(int j=0;j<nc;j++)cands[j].combined=(int64_t)256*cands[j].dot_px+(int64_t)192*cands[j].dot_vg+(int64_t)wc*cands[j].cent_sim+(int64_t)wp*cands[j].prof_sim+(int64_t)wd*cands[j].div_sim+(int64_t)wg*cands[j].gdiv_sim;
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
    printf("=== SSTT MTFP: Native Ternary Classification (%s) ===\n\n",ds);
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

    double int8_mb=(double)(TRAIN_N+TEST_N)*PADDED*3/1048576;
    double pk_mb=(double)(TRAIN_N+TEST_N)*PACKED_PAD*3/1048576;

    /* 5. Compute features BEFORE freeing temp arrays */
    printf("Computing features...\n");double tf=now_sec();
    cent_train=malloc((size_t)TRAIN_N*2);cent_test=malloc((size_t)TEST_N*2);
    hprof_train=aligned_alloc(32,(size_t)TRAIN_N*HP_PAD*2);hprof_test=aligned_alloc(32,(size_t)TEST_N*HP_PAD*2);
    divneg_train=malloc((size_t)TRAIN_N*2);divneg_test=malloc((size_t)TEST_N*2);
    divneg_cy_train=malloc((size_t)TRAIN_N*2);divneg_cy_test=malloc((size_t)TEST_N*2);
    gdiv_2x4_train=malloc((size_t)TRAIN_N*MAX_REGIONS*2);gdiv_2x4_test=malloc((size_t)TEST_N*MAX_REGIONS*2);
    gdiv_3x3_train=malloc((size_t)TRAIN_N*MAX_REGIONS*2);gdiv_3x3_test=malloc((size_t)TEST_N*MAX_REGIONS*2);

    /* Compute from packed arrays */
    compute_centroid_all_pk(pk_tern_train,cent_train,TRAIN_N,PACKED_PAD);
    compute_centroid_all_pk(pk_tern_test,cent_test,TEST_N,PACKED_PAD);
    compute_hprof_all_pk(pk_tern_train,hprof_train,TRAIN_N,PACKED_PAD);
    compute_hprof_all_pk(pk_tern_test,hprof_test,TEST_N,PACKED_PAD);
    compute_div_all_pk(pk_hgrad_train,pk_vgrad_train,divneg_train,divneg_cy_train,TRAIN_N,PACKED_PAD);
    compute_div_all_pk(pk_hgrad_test,pk_vgrad_test,divneg_test,divneg_cy_test,TEST_N,PACKED_PAD);
    compute_grid_div_pk(pk_hgrad_train,pk_vgrad_train,2,4,gdiv_2x4_train,TRAIN_N,PACKED_PAD);
    compute_grid_div_pk(pk_hgrad_test,pk_vgrad_test,2,4,gdiv_2x4_test,TEST_N,PACKED_PAD);
    compute_grid_div_pk(pk_hgrad_train,pk_vgrad_train,3,3,gdiv_3x3_train,TRAIN_N,PACKED_PAD);
    compute_grid_div_pk(pk_hgrad_test,pk_vgrad_test,3,3,gdiv_3x3_test,TEST_N,PACKED_PAD);
    printf("  Features: %.2f sec\n\n",now_sec()-tf);

    /* 6. Free temp int8 arrays */
    free(tern_train);free(tern_test);
    free(hgrad_train);free(hgrad_test);
    free(vgrad_train);free(vgrad_test);

    /* 7. Init trit neighbor table */
    init_trit_nbr();

    /* 8. Build per-channel indices */
    printf("Building per-channel indices...\n");
    printf("  PX: ");build_channel_index(px_tr,&px_bg,px_ig_w,px_idx_off,px_idx_sz,&px_idx_pool);
    printf("  HG: ");build_channel_index(hg_tr,&hg_bg,hg_ig_w,hg_idx_off,hg_idx_sz,&hg_idx_pool);
    printf("  VG: ");build_channel_index(vg_tr,&vg_bg,vg_ig_w,vg_idx_off,vg_idx_sz,&vg_idx_pool);
    printf("\n");

    /* Self-tests */
    printf("Self-tests:\n");
    {
        int8_t orig[PADDED] __attribute__((aligned(32))),recovered[PADDED] __attribute__((aligned(32)));
        uint8_t packed[PACKED_PAD];
        for(int i=0;i<PIXELS;i++)orig[i]=(i%3)-1;
        memset(orig+PIXELS,0,PADDED-PIXELS);
        pack_trits_single(orig,packed,PADDED,PACKED_PAD);
        unpack_trits(packed,recovered);
        int ok=1;
        for(int i=0;i<PADDED;i++)if(orig[i]!=recovered[i]){ok=0;break;}
        printf("  Pack/unpack roundtrip: %s\n",ok?"PASS":"FAIL");
    }
    {
        int ok=1;
        for(int v=0;v<N_BVALS;v++){
            int t[3]={v/9,(v/3)%3,v%3};
            for(int nb=0;nb<TRIT_NBR;nb++){
                uint8_t nv=trit_nbr[v][nb];
                if(nv==0xFF)continue;
                if(nv>=N_BVALS){ok=0;break;}
                int nt[3]={nv/9,(nv/3)%3,nv%3};
                int diffs=0;
                for(int p=0;p<3;p++)if(t[p]!=nt[p])diffs++;
                if(diffs!=1){ok=0;break;}
            }
        }
        printf("  Trit neighbor validity: %s\n",ok?"PASS":"FAIL");
    }
    {
        int8_t a[PADDED] __attribute__((aligned(32))),b[PADDED] __attribute__((aligned(32)));
        unpack_trits(pk_tern_train,a);
        unpack_trits(pk_tern_train+PACKED_PAD,b);
        int32_t d1=tdot(a,b);
        printf("  tdot on packed data: %d (sanity check)\n",d1);
    }

    /* Memory report */
    size_t idx_meta_sz=3*(sizeof(px_idx_off)+sizeof(px_idx_sz))+3*sizeof(uint16_t)*N_BLOCKS+3;
    printf("\nMemory:\n");
    printf("  Packed trit arrays: %.1f MB (was %.1f MB, %.1fx reduction)\n",pk_mb,int8_mb,int8_mb/pk_mb);
    printf("  Index metadata: %.1f KB\n",(double)idx_meta_sz/1024);

    /* 9. Precompute candidates */
    printf("\nPrecomputing candidates...\n");double tp=now_sec();
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

    /* ================================================================
     * PHASE 2: Grid-search feature weights on VAL only
     * ================================================================ */
    printf("=== PHASE 2: Feature weight grid search (VAL only) ===\n");
    {
        int wc_vals[]={0,25,50,100};
        int wp_vals[]={0,8,16};
        int wd_vals[]={50,100,200,400};
        int wg_vals[]={50,100,200,400};
        int sc_vals[]={20,50,100};
        int grid_opts[][3]={{2,4,8},{3,3,9}};

        int best_c=0,best_wc=50,best_wp=0,best_wd=200,best_wg=100,best_sc=50;
        int best_grid=0;
        int16_t *grid_test[]={gdiv_2x4_test,gdiv_3x3_test};
        int16_t *grid_train[]={gdiv_2x4_train,gdiv_3x3_train};

        for(int gi=0;gi<2;gi++){
            int16_t *qg=grid_test[gi],*tg=grid_train[gi];
            int nreg=grid_opts[gi][2];
            for(int ci=0;ci<4;ci++)for(int pi=0;pi<3;pi++)
            for(int di=0;di<4;di++)for(int gvi=0;gvi<4;gvi++)for(int si=0;si<3;si++){
                int c=run_static(pre,nc_arr,qg,tg,nreg,
                                 wc_vals[ci],wp_vals[pi],wd_vals[di],wg_vals[gvi],sc_vals[si],
                                 0,VAL_N,NULL);
                if(c>best_c){best_c=c;best_wc=wc_vals[ci];best_wp=wp_vals[pi];
                             best_wd=wd_vals[di];best_wg=wg_vals[gvi];best_sc=sc_vals[si];best_grid=gi;}
            }
        }

        int16_t *qg=grid_test[best_grid],*tg=grid_train[best_grid];
        int nreg=grid_opts[best_grid][2];
        int w_c=best_wc,w_p=best_wp,w_d=best_wd,w_g=best_wg,sc_val=best_sc;

        printf("  Best weights (val): grid=%dx%d w_c=%d w_p=%d w_d=%d w_g=%d sc=%d\n",
               grid_opts[best_grid][0],grid_opts[best_grid][1],w_c,w_p,w_d,w_g,sc_val);
        printf("  Val accuracy: %.2f%% (%d/%d)\n\n",100.0*best_c/VAL_N,best_c,VAL_N);

        int cA_val=best_c;
        int cA_hold=run_static(pre,nc_arr,qg,tg,nreg,w_c,w_p,w_d,w_g,sc_val,
                               HOLDOUT_START,TEST_N,preds);
        printf("  Static baseline:\n");
        printf("    Val:     %.2f%% (%d/%d, %d errors)\n",100.0*cA_val/VAL_N,cA_val,VAL_N,VAL_N-cA_val);
        printf("    Holdout: %.2f%% (%d/%d, %d errors)\n\n",100.0*cA_hold/(TEST_N-HOLDOUT_START),cA_hold,TEST_N-HOLDOUT_START,TEST_N-HOLDOUT_START-cA_hold);

        /* PHASE 3: Bayesian sequential */
        printf("=== PHASE 3: Bayesian Sequential sweep (VAL only) ===\n");
        {int bds=0,bks=0,btw=0,bc=cA_val;
          int ds_vals[]={1,2,3,5,10,20,50};
          int ks_vals[]={3,5,7,10,15,20,50};
          int tw_vals[]={0,50,100,200};
          for(int di=0;di<7;di++)for(int ki=0;ki<7;ki++)for(int ti=0;ti<4;ti++){
              int c=run_bayesian_seq(pre,nc_arr,qg,tg,nreg,w_c,w_p,w_d,w_g,sc_val,
                                     ds_vals[di],ks_vals[ki],tw_vals[ti],0,VAL_N,NULL);
              if(c>bc){bc=c;bds=ds_vals[di];bks=ks_vals[ki];btw=tw_vals[ti];}
          }
          printf("  Best (val): decay_S=%d K_seq=%d topo_w=%d -> %.2f%% (%+d vs static)\n",
                 bds,bks,btw,100.0*bc/VAL_N,bc-cA_val);
          if(bc>cA_val){
              int cB_hold=run_bayesian_seq(pre,nc_arr,qg,tg,nreg,w_c,w_p,w_d,w_g,sc_val,
                                            bds,bks,btw,HOLDOUT_START,TEST_N,preds);
              printf("  Holdout:   %.2f%% (%d/%d, %+d vs static holdout)\n\n",
                     100.0*cB_hold/(TEST_N-HOLDOUT_START),cB_hold,TEST_N-HOLDOUT_START,cB_hold-cA_hold);
              report(preds,"Bayesian seq (holdout)",cB_hold,HOLDOUT_START,TEST_N);
          } else {
              printf("  No improvement over static on val.\n\n");
          }
        }

        /* PHASE 4: CfC sequential */
        printf("=== PHASE 4: CfC Sequential sweep (VAL only) ===\n");
        {int bg2=0,be=0,bp=0,bk=0,bc=cA_val;
          int g_vals[]={1,2,4,8,16};
          int e_vals[]={2,5,10,20,50};
          int p_vals[]={0,1,5,10,50};
          int k_vals[]={3,5,10,20,50};
          for(int gi=0;gi<5;gi++)for(int ei=0;ei<5;ei++)for(int pi=0;pi<5;pi++)for(int ki=0;ki<5;ki++){
              int c=run_cfc(pre,nc_arr,qg,tg,nreg,w_c,w_p,w_d,w_g,sc_val,
                            g_vals[gi],e_vals[ei],p_vals[pi],k_vals[ki],0,VAL_N,NULL);
              if(c>bc){bc=c;bg2=g_vals[gi];be=e_vals[ei];bp=p_vals[pi];bk=k_vals[ki];}
          }
          printf("  Best (val): gain=%d exp_S=%d penalty=%d K_seq=%d -> %.2f%% (%+d vs static)\n",
                 bg2,be,bp,bk,100.0*bc/VAL_N,bc-cA_val);
          if(bc>cA_val){
              int cC_hold=run_cfc(pre,nc_arr,qg,tg,nreg,w_c,w_p,w_d,w_g,sc_val,
                                   bg2,be,bp,bk,HOLDOUT_START,TEST_N,preds);
              printf("  Holdout:   %.2f%% (%d/%d, %+d vs static holdout)\n\n",
                     100.0*cC_hold/(TEST_N-HOLDOUT_START),cC_hold,TEST_N-HOLDOUT_START,cC_hold-cA_hold);
              report(preds,"CfC seq (holdout)",cC_hold,HOLDOUT_START,TEST_N);
          } else {
              printf("  No improvement over static on val.\n\n");
          }
        }

        /* PHASE 5: Pipeline A->B */
        printf("=== PHASE 5: Pipeline A->B sweep (VAL only) ===\n");
        {int bds=0,bka=0,btw=0,bg2=0,be2=0,bp2=0,bkb=0,bc=cA_val;
          int ds_vals[]={2,5,10};
          int ka_vals[]={3,5,10};
          int tw_vals[]={0,100};
          int g_vals[]={2,4,8};
          int e_vals[]={5,10,20};
          int p_vals[]={1,5,10};
          int kb_vals[]={5,10,20,50};
          for(int d=0;d<3;d++)for(int a=0;a<3;a++)for(int t=0;t<2;t++)
          for(int g=0;g<3;g++)for(int e=0;e<3;e++)for(int p=0;p<3;p++)for(int k=0;k<4;k++){
              int c=run_pipeline(pre,nc_arr,qg,tg,nreg,w_c,w_p,w_d,w_g,sc_val,
                                 ds_vals[d],ka_vals[a],tw_vals[t],
                                 g_vals[g],e_vals[e],p_vals[p],kb_vals[k],0,VAL_N,NULL);
              if(c>bc){bc=c;bds=ds_vals[d];bka=ka_vals[a];btw=tw_vals[t];
                       bg2=g_vals[g];be2=e_vals[e];bp2=p_vals[p];bkb=kb_vals[k];}
          }
          printf("  Best (val): dS=%d Ka=%d tw=%d | g=%d eS=%d p=%d Kb=%d -> %.2f%% (%+d vs static)\n",
                 bds,bka,btw,bg2,be2,bp2,bkb,100.0*bc/VAL_N,bc-cA_val);
          if(bc>cA_val){
              int cD_hold=run_pipeline(pre,nc_arr,qg,tg,nreg,w_c,w_p,w_d,w_g,sc_val,
                                        bds,bka,btw,bg2,be2,bp2,bkb,HOLDOUT_START,TEST_N,preds);
              printf("  Holdout:   %.2f%% (%d/%d, %+d vs static holdout)\n\n",
                     100.0*cD_hold/(TEST_N-HOLDOUT_START),cD_hold,TEST_N-HOLDOUT_START,cD_hold-cA_hold);
              report(preds,"Pipeline A->B (holdout)",cD_hold,HOLDOUT_START,TEST_N);
          } else {
              printf("  No improvement over static on val.\n\n");
          }
        }

        /* Comparison with topo9 baseline */
        printf("=== COMPARISON ===\n");
        {
            int16_t *oqg,*otg;int onreg,owc,owp,owd,owg,osc;
            if(is_fashion){oqg=gdiv_3x3_test;otg=gdiv_3x3_train;onreg=9;owc=25;owp=0;owd=200;owg=200;osc=50;}
            else{oqg=gdiv_2x4_test;otg=gdiv_2x4_train;onreg=8;owc=50;owp=16;owd=200;owg=100;osc=50;}
            int cOrig_full=run_static(pre,nc_arr,oqg,otg,onreg,owc,owp,owd,owg,osc,0,TEST_N,NULL);
            int cMtfp_full=run_static(pre,nc_arr,qg,tg,nreg,w_c,w_p,w_d,w_g,sc_val,0,TEST_N,NULL);
            printf("  topo9_val baseline:  %.2f%% (full 10K, original weights)\n",100.0*cOrig_full/TEST_N);
            printf("  MTFP full 10K:       %.2f%%\n",100.0*cMtfp_full/TEST_N);
            printf("  Delta:               %+.2f pp\n",100.0*(cMtfp_full-cOrig_full)/TEST_N);
        }
    }

    printf("\nTotal: %.2f sec\n",now_sec()-t0);

    free(pre);free(nc_arr);free(preds);
    free(gdiv_2x4_train);free(gdiv_2x4_test);free(gdiv_3x3_train);free(gdiv_3x3_test);
    free(cent_train);free(cent_test);free(hprof_train);free(hprof_test);
    free(divneg_train);free(divneg_test);free(divneg_cy_train);free(divneg_cy_test);
    free(px_idx_pool);free(hg_idx_pool);free(vg_idx_pool);
    free(px_tr);free(px_te);free(hg_tr);free(hg_te);free(vg_tr);free(vg_te);
    free(pk_tern_train);free(pk_tern_test);free(pk_hgrad_train);free(pk_hgrad_test);free(pk_vgrad_train);free(pk_vgrad_test);
    free(raw_train_img);free(raw_test_img);free(train_labels);free(test_labels);
    return 0;
}
