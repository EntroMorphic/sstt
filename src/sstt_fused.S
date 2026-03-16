/*
 * sstt_fused.S — Fused AVX2 Ternary Classification Kernel
 *
 * int sstt_fused_classify(const uint8_t *pixels,     // rdi
 *                         const uint32_t *px_hot,     // rsi
 *                         const uint32_t *hg_hot,     // rdx
 *                         const uint32_t *vg_hot);    // rcx
 *
 * System V AMD64 ABI. Returns class label in eax.
 *
 * AT&T VEX 3-operand convention:
 *   vpsubb %a, %b, %c   →   c = b - a
 *   vpxor  %a, %b, %c   →   c = b ^ a
 *   vpand  %a, %b, %c   →   c = b & a
 *   vpsignb %a, %b, %c  →   c = signb(b, a)  (negate b where a<0, zero where a=0)
 *   vpaddd %a, %b, %c   →   c = b + a
 *   vpaddd mem, %b, %c  →   c = b + [mem]
 */

    .text
    .globl  sstt_fused_classify
    .type   sstt_fused_classify, @function
    .p2align 4

sstt_fused_classify:
    push    %rbx
    push    %rbp
    push    %r12
    push    %r13
    push    %r14
    push    %r15
    sub     $232, %rsp

    mov     %rdi, %r12          /* pixels */
    mov     %rsi, %r13          /* px_hot */
    mov     %rdx, %r14          /* hg_hot */
    mov     %rcx, %r15          /* vg_hot */

    /* Zero accumulators */
    vpxor   %ymm0, %ymm0, %ymm0
    vpxor   %ymm1, %ymm1, %ymm1

    /* Build constants on stack, then load to YMM */
    movabsq $0x8080808080808080, %rax
    mov     %rax, 160(%rsp)
    mov     %rax, 168(%rsp)
    mov     %rax, 176(%rsp)
    mov     %rax, 184(%rsp)
    vmovdqa 160(%rsp), %ymm2           /* ymm2 = 0x80 bias */

    movabsq $0x0101010101010101, %rax
    mov     %rax, 160(%rsp)
    mov     %rax, 168(%rsp)
    mov     %rax, 176(%rsp)
    mov     %rax, 184(%rsp)
    vmovdqa 160(%rsp), %ymm5           /* ymm5 = 0x01 ones */

    movabsq $0x2A2A2A2A2A2A2A2A, %rax
    mov     %rax, 160(%rsp)
    mov     %rax, 168(%rsp)
    mov     %rax, 176(%rsp)
    mov     %rax, 184(%rsp)
    vmovdqa 160(%rsp), %ymm3           /* ymm3 = 170^0x80 = 0x2A */

    movabsq $0xD5D5D5D5D5D5D5D5, %rax
    mov     %rax, 160(%rsp)
    mov     %rax, 168(%rsp)
    mov     %rax, 176(%rsp)
    mov     %rax, 184(%rsp)
    vmovdqa 160(%rsp), %ymm4           /* ymm4 = 85^0x80 = 0xD5 */

    /* Row mask at rsp+192: bytes 0-27 = 0xFF, bytes 28-31 = 0x00 */
    vpcmpeqd %ymm10, %ymm10, %ymm10
    vmovdqu %ymm10, 192(%rsp)
    movl    $0, 220(%rsp)

    /* ---- Row loop ---- */
    xor     %ebp, %ebp
    xor     %ebx, %ebx

.Lrow_loop:
    /* Load current row (28 pixels, mask to zero bytes 28-31) */
    imul    $28, %rbp, %rax
    vmovdqu (%r12, %rax), %ymm6
    vpand   192(%rsp), %ymm6, %ymm6

    /* Quantize: pixel > 170 → +1, pixel < 85 → -1, else 0 */
    /* vpxor %a, %b, %c → c = b ^ a */
    vpxor   %ymm2, %ymm6, %ymm10       /* ymm10 = ymm6 ^ ymm2 (bias) */
    vpcmpgtb %ymm11, %ymm10, %ymm3     /* ymm11 = (biased > 0x2A) = (pixel > 170) */
    vpcmpgtb %ymm12, %ymm4, %ymm10     /* ymm12 = (0xD5 > biased) = (pixel < 85) */
    /* vpand %a, %b, %c → c = b & a */
    vpand   %ymm5, %ymm11, %ymm11      /* ymm11 = +1 where pixel > 170 */
    vpand   %ymm5, %ymm12, %ymm12      /* ymm12 = +1 where pixel < 85 */
    /* vpsubb %a, %b, %c → c = b - a. Want: trit = pos - neg = ymm11 - ymm12 */
    vpsubb  %ymm12, %ymm11, %ymm6      /* ymm6 = ymm11 - ymm12 = pixel trits */

    /* Load and quantize next row (or zero if last) */
    cmp     $27, %ebp
    je      .Llast_row

    lea     28(%rax), %r9
    vmovdqu (%r12, %r9), %ymm7
    vpand   192(%rsp), %ymm7, %ymm7
    vpxor   %ymm2, %ymm7, %ymm10
    vpcmpgtb %ymm11, %ymm10, %ymm3
    vpcmpgtb %ymm12, %ymm4, %ymm10
    vpand   %ymm5, %ymm11, %ymm11
    vpand   %ymm5, %ymm12, %ymm12
    vpsubb  %ymm12, %ymm11, %ymm7      /* ymm7 = next row trits */
    jmp     .Lcompute_grads

.Llast_row:
    vpxor   %ymm7, %ymm7, %ymm7

.Lcompute_grads:
    /* H-gradient: clamp(trit[x+1] - trit[x]) */
    vmovdqu %ymm6, 64(%rsp)            /* store pixel trits */
    vmovdqu 65(%rsp), %ymm10           /* ymm10 = pixel trits shifted right 1 byte */
    /* vpsubb %a, %b, %c → c = b - a. Want: diff = shifted - original = ymm10 - ymm6 */
    vpsubb  %ymm6, %ymm10, %ymm8       /* ymm8 = ymm10 - ymm6 = raw h-diff */
    /* vpsignb %a, %b, %c → c = signb(b, a): for each byte,
     *   a>0: c=b, a=0: c=0, a<0: c=-b.
     * We want clamp(diff): diff>0→+1, diff=0→0, diff<0→-1.
     * Use vpsignb(diff, ones): diff>0→+1, diff=0→0, diff<0→-1. */
    vpsignb %ymm8, %ymm5, %ymm8        /* ymm8 = signb(ymm5=ones, ymm8=diff) = clamp */
    /* Zero byte 27 (no valid h-grad for last pixel in row) */
    vmovdqu %ymm8, 96(%rsp)
    movb    $0, 123(%rsp)               /* 96+27 = 123 */
    vmovdqu 96(%rsp), %ymm8

    /* V-gradient: clamp(next_row - curr_row) */
    cmp     $27, %ebp
    je      .Lzero_vgrad
    /* vpsubb %a, %b, %c → c = b - a. Want: ymm7 - ymm6 */
    vpsubb  %ymm6, %ymm7, %ymm9        /* ymm9 = ymm7 - ymm6 = raw v-diff */
    vpsignb %ymm9, %ymm5, %ymm9        /* clamp */
    jmp     .Lstore_trits

.Lzero_vgrad:
    vpxor   %ymm9, %ymm9, %ymm9

.Lstore_trits:
    /* pixel trits at rsp+64 (already stored above) */
    vmovdqu %ymm8, 96(%rsp)            /* h-grad at rsp+96 */
    vmovdqu %ymm9, 128(%rsp)           /* v-grad at rsp+128 */

    /* ---- Strip loop: 9 strips per row ---- */
    xor     %r8d, %r8d

.Lstrip_loop:
    lea     (%r8, %r8, 2), %r9         /* r9 = strip * 3 */

    /* ============ PIXEL CHANNEL ============ */
    movsbq  64(%rsp, %r9), %rax
    movsbq  65(%rsp, %r9), %rcx
    movsbq  66(%rsp, %r9), %rdx
    /* bv = (t0+1)*9 + (t1+1)*3 + (t2+1) */
    add     $1, %eax
    imul    $9, %eax, %eax
    add     $1, %ecx
    lea     (%rcx, %rcx, 2), %ecx
    add     %ecx, %eax
    add     $1, %edx
    add     %edx, %eax

    test    %eax, %eax
    jz      .Lskip_px
    cmp     $26, %eax
    ja      .Lskip_px           /* bounds check: bv must be 0-26 */

    imul    $1728, %r8d, %r10d
    add     %rbx, %r10
    shl     $6, %eax
    add     %rax, %r10
    prefetcht0 1728(%r13, %r10)
    vpaddd  (%r13, %r10), %ymm0, %ymm0
    vpaddd  32(%r13, %r10), %ymm1, %ymm1

.Lskip_px:
    /* ============ H-GRADIENT CHANNEL ============ */
    movsbq  96(%rsp, %r9), %rax
    movsbq  97(%rsp, %r9), %rcx
    movsbq  98(%rsp, %r9), %rdx
    add     $1, %eax
    imul    $9, %eax, %eax
    add     $1, %ecx
    lea     (%rcx, %rcx, 2), %ecx
    add     %ecx, %eax
    add     $1, %edx
    add     %edx, %eax

    cmp     $13, %eax
    je      .Lskip_hg
    cmp     $26, %eax
    ja      .Lskip_hg

    imul    $1728, %r8d, %r10d
    add     %rbx, %r10
    shl     $6, %eax
    add     %rax, %r10
    prefetcht0 1728(%r14, %r10)
    vpaddd  (%r14, %r10), %ymm0, %ymm0
    vpaddd  32(%r14, %r10), %ymm1, %ymm1

.Lskip_hg:
    /* ============ V-GRADIENT CHANNEL ============ */
    movsbq  128(%rsp, %r9), %rax
    movsbq  129(%rsp, %r9), %rcx
    movsbq  130(%rsp, %r9), %rdx
    add     $1, %eax
    imul    $9, %eax, %eax
    add     $1, %ecx
    lea     (%rcx, %rcx, 2), %ecx
    add     %ecx, %eax
    add     $1, %edx
    add     %edx, %eax

    cmp     $13, %eax
    je      .Lskip_vg
    cmp     $26, %eax
    ja      .Lskip_vg

    imul    $1728, %r8d, %r10d
    add     %rbx, %r10
    shl     $6, %eax
    add     %rax, %r10
    prefetcht0 1728(%r15, %r10)
    vpaddd  (%r15, %r10), %ymm0, %ymm0
    vpaddd  32(%r15, %r10), %ymm1, %ymm1

.Lskip_vg:
    inc     %r8d
    cmp     $9, %r8d
    jl      .Lstrip_loop

    add     $15552, %ebx
    inc     %ebp
    cmp     $28, %ebp
    jl      .Lrow_loop

    /* ---- Argmax ---- */
    vmovdqa %ymm0, (%rsp)
    vmovdqa %ymm1, 32(%rsp)
    xor     %eax, %eax
    mov     (%rsp), %r8d
    mov     $1, %ecx
.Largmax_loop:
    cmp     $10, %ecx
    jge     .Largmax_done
    mov     (%rsp, %rcx, 4), %r9d
    cmp     %r8d, %r9d
    jle     .Largmax_next
    mov     %r9d, %r8d
    mov     %ecx, %eax
.Largmax_next:
    inc     %ecx
    jmp     .Largmax_loop
.Largmax_done:

    vzeroupper
    add     $232, %rsp
    pop     %r15
    pop     %r14
    pop     %r13
    pop     %r12
    pop     %rbp
    pop     %rbx
    ret

    .size   sstt_fused_classify, .-sstt_fused_classify
