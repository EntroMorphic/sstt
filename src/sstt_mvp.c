/*
 * sstt_mvp.c — Falsification test: Redundant Ternary vs Standard Balanced Ternary
 *
 * Hypothesis: Redundant Ternary encoding (RT, 13 levels, 3.7 bits/symbol) provides
 * better effective throughput than Standard Balanced Ternary (SBT, 27 levels,
 * 4.75 bits/symbol) under analog Gaussian noise, justifying the density loss.
 *
 * Both schemes encode 3 trits into a single analog level:
 *   SBT: t0*1 + t1*3 + t2*9  →  27 levels in [-13, +13]
 *   RT:  t0*1 + t1*2 + t2*3  →  13 levels in [-6, +6]  (redundant)
 *
 * Mapped to same voltage range [-1, +1]. Same Gaussian noise. Decode to nearest
 * valid level. Measure symbol error rate and effective throughput (bits × accuracy).
 *
 * Test A: Per-symbol error sweep across SNR 0–40 dB.
 * Test B: 16-class argmax preservation (simulated classification accuracy).
 *
 * Build: gcc -O3 -mavx2 -mfma -march=native -o sstt_mvp sstt_mvp.c -lm
 */

#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ---------- Config ---------- */
#define TRIALS_A    10000000    /* trials per SNR level (Test A) */
#define TRIALS_B    10000       /* classification trials (Test B) */
#define N_GROUPS    256         /* 3-trit groups per dot product (= 768 trits) */
#define N_CLASSES   16          /* output neurons */
#define BATCH       8           /* AVX2 float lane width */

#define SBT_BITS    4.754887502 /* log2(27) */
#define RT_BITS     3.700439718 /* log2(13) */
#define TRIT_BITS   1.584962501 /* log2(3)  */

/* ---------- xoshiro256++ PRNG ---------- */
static uint64_t rs[4];

static inline uint64_t rotl(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

static inline uint64_t rng(void) {
    uint64_t r = rotl(rs[0] + rs[3], 23) + rs[0];
    uint64_t t = rs[1] << 17;
    rs[2] ^= rs[0]; rs[3] ^= rs[1];
    rs[1] ^= rs[2]; rs[0] ^= rs[3];
    rs[2] ^= t; rs[3] = rotl(rs[3], 45);
    return r;
}

static void seed_rng(uint64_t v) {
    for (int i = 0; i < 4; i++) {
        v += 0x9e3779b97f4a7c15ULL;
        uint64_t z = v;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        rs[i] = z ^ (z >> 31);
    }
}

/* ---------- Fast trit generation: {-1, 0, 1} ---------- */
static uint64_t tbuf;
static int trem = 0;

static inline int rand_trit(void) {
    int v;
    do {
        if (trem < 2) { tbuf = rng(); trem = 64; }
        v = tbuf & 3;
        tbuf >>= 2;
        trem -= 2;
    } while (v == 3);      /* reject 3, keep 0/1/2 → map to -1/0/1 */
    return v - 1;
}

/* ---------- Box-Muller Gaussian ---------- */
static int gspare = 0;
static float gval;

static inline float gauss(void) {
    if (gspare) { gspare = 0; return gval; }
    float u, v, s;
    do {
        u = (float)(rng() >> 40) / (float)(1 << 24) * 2.0f - 1.0f;
        v = (float)(rng() >> 40) / (float)(1 << 24) * 2.0f - 1.0f;
        s = u * u + v * v;
    } while (s >= 1.0f || s == 0.0f);
    float m = sqrtf(-2.0f * logf(s) / s);
    gval = v * m;
    gspare = 1;
    return u * m;
}

/* ---------- AVX2 batch: encode → normalize → noise → decode → compare ----------
 *
 * Processes 8 trials in parallel:
 *   1. int encoded values → float
 *   2. Normalize to [-1, 1]  (divide by max level)
 *   3. Add Gaussian noise × sigma
 *   4. Denormalize (multiply by max level)
 *   5. Round to nearest integer (banker's rounding via cvtps)
 *   6. Clamp to valid range
 *   7. Compare to original, accumulate |error|
 */
static inline void avx2_batch(
    const int enc[8],       /* encoded integer values */
    const float noise[8],   /* pre-generated Gaussian samples */
    float sigma,            /* noise std dev in normalized [-1,1] space */
    float inv_range,        /* 1.0 / max_abs_level */
    float range,            /* max_abs_level */
    int clamp_lo,
    int clamp_hi,
    int *ncorrect,          /* out: count of correct decodes (0-8) */
    float *sum_abserr       /* out: sum of |decoded - true| across 8 lanes */
) {
    __m256i ve = _mm256_loadu_si256((const __m256i *)enc);
    __m256  vf = _mm256_cvtepi32_ps(ve);
    __m256  vn = _mm256_loadu_ps(noise);

    /* normalize → add noise → denormalize */
    __m256 norm   = _mm256_mul_ps(vf, _mm256_set1_ps(inv_range));
    __m256 noisy  = _mm256_fmadd_ps(vn, _mm256_set1_ps(sigma), norm);
    __m256 denorm = _mm256_mul_ps(noisy, _mm256_set1_ps(range));

    /* round to nearest integer */
    __m256i rd = _mm256_cvtps_epi32(denorm);

    /* clamp */
    rd = _mm256_max_epi32(rd, _mm256_set1_epi32(clamp_lo));
    rd = _mm256_min_epi32(rd, _mm256_set1_epi32(clamp_hi));

    /* count correct: cmpeq gives 0xFFFFFFFF per matching lane → 4 mask bits each */
    __m256i eq = _mm256_cmpeq_epi32(rd, ve);
    *ncorrect = __builtin_popcount((unsigned)_mm256_movemask_epi8(eq)) >> 2;

    /* sum of absolute errors */
    __m256i diff = _mm256_abs_epi32(_mm256_sub_epi32(rd, ve));
    __m256  df   = _mm256_cvtepi32_ps(diff);
    __m128  lo128 = _mm_add_ps(_mm256_castps256_ps128(df),
                               _mm256_extractf128_ps(df, 1));
    lo128 = _mm_hadd_ps(lo128, lo128);
    lo128 = _mm_hadd_ps(lo128, lo128);
    *sum_abserr = _mm_cvtss_f32(lo128);
}

/* ---------- Timing helper ---------- */
static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ================================================================== */
int main(void) {
    seed_rng(42);
    double t0 = now_sec();

    puts("=== SSTT MVP: Redundant Ternary Falsification Test ===");
    puts("");
    puts("Hypothesis: RT (13 levels, 3.70 bits/sym) outperforms SBT (27 levels,");
    puts("4.75 bits/sym) in effective throughput under analog noise.");
    puts("");
    printf("Test A: %d trials/SNR level, SNR 0-40 dB\n", TRIALS_A);
    printf("Test B: %d trials, %d groups/dot-product, %d classes\n\n", TRIALS_B, N_GROUPS, N_CLASSES);

    /* ================================================================
     *  TEST A: Per-Symbol Error Rate & Effective Throughput Sweep
     * ================================================================ */
    puts("--- Test A: Per-Symbol Error Rate & Effective Throughput ---");
    puts("");
    printf("%-6s | %-9s %-9s %-9s | %-9s %-9s %-9s | %s\n",
           "SNR_dB", "SBT_err%", "RT_err%", "Trit_err%",
           "SBT_tput", "RT_tput", "Tri_tput", "Winner");
    puts("-------+-------------------------------+-------------------------------+--------");

    int rt_wins_any = 0;
    float first_rt_win = -1.0f;
    float last_rt_win  = -1.0f;

    for (float snr = 0.0f; snr <= 40.01f; snr += 2.0f) {
        float sigma = 1.0f / powf(10.0f, snr / 20.0f);

        long long sbt_ok = 0, rt_ok = 0, trit_ok = 0;
        double sbt_ae = 0.0, rt_ae = 0.0, trit_ae = 0.0;

        for (int t = 0; t < TRIALS_A; t += BATCH) {
            int se[8], re[8], te[8];
            float n1[8], n2[8];
            int bsz = (TRIALS_A - t < BATCH) ? TRIALS_A - t : BATCH;

            for (int b = 0; b < bsz; b++) {
                int a = rand_trit(), c = rand_trit(), d = rand_trit();
                se[b] = a + 3 * c + 9 * d;     /* SBT encoding */
                re[b] = a + 2 * c + 3 * d;     /* RT encoding  */
                te[b] = a;                       /* single trit  */
                n1[b] = gauss();                 /* shared noise for SBT & RT */
                n2[b] = gauss();                 /* independent noise for trit */
            }
            for (int b = bsz; b < BATCH; b++)
                se[b] = re[b] = te[b] = 0, n1[b] = n2[b] = 0.0f;

            int nc;
            float ae;

            /* SBT: 27 levels in [-13, +13] */
            avx2_batch(se, n1, sigma, 1.0f / 13.0f, 13.0f, -13, 13, &nc, &ae);
            sbt_ok += (nc > bsz) ? bsz : nc;
            sbt_ae += ae;

            /* RT: 13 levels in [-6, +6], same noise as SBT */
            avx2_batch(re, n1, sigma, 1.0f / 6.0f, 6.0f, -6, 6, &nc, &ae);
            rt_ok += (nc > bsz) ? bsz : nc;
            rt_ae += ae;

            /* Trit baseline: 3 levels in [-1, +1], independent noise */
            avx2_batch(te, n2, sigma, 1.0f, 1.0f, -1, 1, &nc, &ae);
            trit_ok += (nc > bsz) ? bsz : nc;
            trit_ae += ae;
        }

        double sbt_err  = 1.0 - (double)sbt_ok  / TRIALS_A;
        double rt_err   = 1.0 - (double)rt_ok   / TRIALS_A;
        double trit_err = 1.0 - (double)trit_ok / TRIALS_A;

        double sbt_tp  = SBT_BITS  * (1.0 - sbt_err);
        double rt_tp   = RT_BITS   * (1.0 - rt_err);
        double trit_tp = TRIT_BITS * (1.0 - trit_err);

        const char *w;
        if (rt_tp > sbt_tp && rt_tp > trit_tp) {
            w = "RT *";
            rt_wins_any = 1;
            if (first_rt_win < 0) first_rt_win = snr;
            last_rt_win = snr;
        } else if (sbt_tp >= trit_tp) {
            w = "SBT";
        } else {
            w = "Trit";
        }

        printf("%5.1f  | %7.3f%%  %7.3f%%  %7.3f%%  | %7.4f   %7.4f   %7.4f   | %s\n",
               snr,
               sbt_err  * 100.0, rt_err   * 100.0, trit_err * 100.0,
               sbt_tp, rt_tp, trit_tp, w);
    }

    double t_a = now_sec();
    printf("\nTest A completed in %.2f seconds.\n", t_a - t0);

    /* ================================================================
     *  TEST B: Classification Accuracy (argmax preservation)
     *
     *  Simulates a 16-class single-layer classifier. Each neuron computes
     *  a dot product of 768 ternary values (256 groups of 3). Noise is
     *  injected per-group at the analog level. We check whether the noisy
     *  argmax matches the true argmax for each scheme independently.
     * ================================================================ */
    puts("");
    puts("--- Test B: Classification Accuracy (argmax preservation) ---");
    puts("");
    printf("%-6s | %-12s %-12s %-12s | %-10s %-10s\n",
           "SNR_dB", "SBT_class%", "RT_class%", "Trit_class%",
           "SBT_MAE", "RT_MAE");
    puts("-------+------------------------------------------+----------------------");

    float test_snrs[] = {10.0f, 15.0f, 20.0f, 25.0f, 30.0f};
    int n_test = 5;

    for (int si = 0; si < n_test; si++) {
        float snr = test_snrs[si];
        float sigma = 1.0f / powf(10.0f, snr / 20.0f);

        int sbt_match = 0, rt_match = 0, trit_match = 0;
        double sbt_mae = 0.0, rt_mae = 0.0;

        for (int trial = 0; trial < TRIALS_B; trial++) {
            int true_sbt[N_CLASSES],  noisy_sbt[N_CLASSES];
            int true_rt[N_CLASSES],   noisy_rt[N_CLASSES];
            int true_trit[N_CLASSES], noisy_trit[N_CLASSES];

            memset(true_sbt,   0, sizeof(true_sbt));
            memset(true_rt,    0, sizeof(true_rt));
            memset(true_trit,  0, sizeof(true_trit));
            memset(noisy_sbt,  0, sizeof(noisy_sbt));
            memset(noisy_rt,   0, sizeof(noisy_rt));
            memset(noisy_trit, 0, sizeof(noisy_trit));

            for (int cls = 0; cls < N_CLASSES; cls++) {
                for (int g = 0; g < N_GROUPS; g++) {
                    int a = rand_trit(), b = rand_trit(), c = rand_trit();

                    int sv = a + 3 * b + 9 * c;
                    int rv = a + 2 * b + 3 * c;

                    true_sbt[cls] += sv;
                    true_rt[cls]  += rv;
                    true_trit[cls] += a + b + c;

                    /* SBT: noisy decode */
                    float ns = (float)sv / 13.0f + gauss() * sigma;
                    int ds = (int)roundf(ns * 13.0f);
                    if (ds < -13) ds = -13;
                    if (ds >  13) ds =  13;
                    noisy_sbt[cls] += ds;

                    /* RT: noisy decode (same noise model, different range) */
                    float nr = (float)rv / 6.0f + gauss() * sigma;
                    int dr = (int)roundf(nr * 6.0f);
                    if (dr < -6) dr = -6;
                    if (dr >  6) dr =  6;
                    noisy_rt[cls] += dr;

                    /* Trit baseline: 3 independent ADC conversions */
                    int trits[3] = {a, b, c};
                    for (int ti = 0; ti < 3; ti++) {
                        float nt = (float)trits[ti] + gauss() * sigma;
                        int dt = (int)roundf(nt);
                        if (dt < -1) dt = -1;
                        if (dt >  1) dt =  1;
                        noisy_trit[cls] += dt;
                    }
                }

                sbt_mae += abs(noisy_sbt[cls] - true_sbt[cls]);
                rt_mae  += abs(noisy_rt[cls]  - true_rt[cls]);
            }

            /* argmax for each scheme (compare noisy vs true independently) */
            int ts = 0, tr_ = 0, tt = 0, ns_ = 0, nr_ = 0, nt_ = 0;
            for (int cls = 1; cls < N_CLASSES; cls++) {
                if (true_sbt[cls]   > true_sbt[ts])   ts  = cls;
                if (true_rt[cls]    > true_rt[tr_])    tr_ = cls;
                if (true_trit[cls]  > true_trit[tt])   tt  = cls;
                if (noisy_sbt[cls]  > noisy_sbt[ns_])  ns_ = cls;
                if (noisy_rt[cls]   > noisy_rt[nr_])   nr_ = cls;
                if (noisy_trit[cls] > noisy_trit[nt_])  nt_ = cls;
            }
            if (ns_ == ts)  sbt_match++;
            if (nr_ == tr_) rt_match++;
            if (nt_ == tt)  trit_match++;
        }

        printf("%5.1f  | %9.2f%%   %9.2f%%   %9.2f%%   | %8.1f   %8.1f\n",
               snr,
               100.0 * sbt_match  / TRIALS_B,
               100.0 * rt_match   / TRIALS_B,
               100.0 * trit_match / TRIALS_B,
               sbt_mae / ((double)TRIALS_B * N_CLASSES),
               rt_mae  / ((double)TRIALS_B * N_CLASSES));
    }

    double t_b = now_sec();
    printf("\nTest B completed in %.2f seconds.\n", t_b - t_a);

    /* ================================================================
     *  ANALYSIS & VERDICT
     * ================================================================ */
    puts("");
    puts("=== ANALYSIS ===");
    puts("");
    puts("Normalized voltage spacing (decision margin per symbol):");
    puts("  SBT:  1/13 = 0.0769  (27 levels over [-1,+1])");
    puts("  RT:   1/6  = 0.1667  (13 levels over [-1,+1])");
    puts("  Trit: 1/1  = 1.0000  ( 3 levels over [-1,+1])");
    puts("  RT has 2.17x the margin of SBT, but carries 22% fewer bits.");
    puts("");

    puts("=== VERDICT ===");
    if (rt_wins_any) {
        printf("RT outperforms SBT in effective throughput at SNR %.0f–%.0f dB.\n",
               first_rt_win, last_rt_win);
        if (first_rt_win >= 10 && first_rt_win <= 30)
            puts("Crossover is within realistic analog hardware range (10-30 dB).");
        puts("STATUS: IDEA HAS MERIT in the identified noise regime.");
        puts("  The redundancy buys noise tolerance that compensates for density loss");
        puts("  at moderate SNR. At high SNR, SBT's density advantage dominates.");
    } else {
        puts("RT never outperforms SBT in effective throughput at any tested SNR.");
        puts("STATUS: FALSIFIED — redundancy does not compensate for information loss.");
    }

    printf("\nTotal runtime: %.2f seconds.\n", now_sec() - t0);
    return rt_wins_any ? 0 : 1;
}
