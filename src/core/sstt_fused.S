/*
 * sstt_fused.S — Fused AVX2 Ternary Classification Kernel (TURBO EDITION)
 *
 * Performance Fixes:
 * 1. Replaced stack-based byte-shifting with vpsrldq (In-register).
 * 2. Replaced 81 stack loads per row with vpextrb (Direct extraction).
 * 3. Replaced imul with pointer increments.
 * 4. Eliminated STLF (Store-to-Load Forwarding) blocks.
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

    /* Zero accumulators (ymm0, ymm1 hold class counts) */
    vpxor   %ymm0, %ymm0, %ymm0
    vpxor   %ymm1, %ymm1, %ymm1

    /* Constants */
    movabsq $0x8080808080808080, %rax
    mov     %rax, 160(%rsp)
    mov     %rax, 168(%rsp)
    mov     %rax, 176(%rsp)
    mov     %rax, 184(%rsp)
    vmovdqu 160(%rsp), %ymm2           /* ymm2 = 0x80 bias */

    movabsq $0x0101010101010101, %rax
    mov     %rax, 160(%rsp)
    mov     %rax, 168(%rsp)
    mov     %rax, 176(%rsp)
    mov     %rax, 184(%rsp)
    vmovdqu 160(%rsp), %ymm5           /* ymm5 = 0x01 ones */

    movabsq $0x2A2A2A2A2A2A2A2A, %rax
    mov     %rax, 160(%rsp)
    mov     %rax, 168(%rsp)
    mov     %rax, 176(%rsp)
    mov     %rax, 184(%rsp)
    vmovdqu 160(%rsp), %ymm3           /* ymm3 = 170^0x80 = 0x2A */

    movabsq $0xD5D5D5D5D5D5D5D5, %rax
    mov     %rax, 160(%rsp)
    mov     %rax, 168(%rsp)
    mov     %rax, 176(%rsp)
    mov     %rax, 184(%rsp)
    vmovdqu 160(%rsp), %ymm4           /* ymm4 = 85^0x80 = 0xD5 */

    /* Row mask: zero bytes 28-31 */
    vpcmpeqd %ymm10, %ymm10, %ymm10
    vmovdqu %ymm10, 192(%rsp)
    movl    $0, 220(%rsp)
    vmovdqu 192(%rsp), %ymm15

    /* ---- Row loop ---- */
    xor     %ebp, %ebp          /* row counter */
    xor     %ebx, %ebx          /* map base offset */

.Lrow_loop:
    /* 1. Load and Quantize Current Row */
    imul    $28, %rbp, %rax
    vmovdqu (%r12, %rax), %ymm6
    vpand   %ymm15, %ymm6, %ymm6
    vpxor   %ymm2, %ymm6, %ymm10       /* biased */
    vpcmpgtb %ymm3, %ymm10, %ymm11     /* biased > 0x2A? (pixel > 170) */
    vpcmpgtb %ymm10, %ymm4, %ymm12     /* 0xD5 > biased? (pixel < 85) */
    vpand   %ymm5, %ymm11, %ymm11      /* +1 where pos */
    vpand   %ymm5, %ymm12, %ymm12      /* +1 where neg */
    vpsubb  %ymm12, %ymm11, %ymm6      /* ymm6 = current row trits */

    /* 2. Load and Quantize Next Row */
    cmp     $27, %ebp
    je      .Llast_row_turbo
    vmovdqu 28(%r12, %rax), %ymm7
    vpand   %ymm15, %ymm7, %ymm7
    vpxor   %ymm2, %ymm7, %ymm10
    vpcmpgtb %ymm3, %ymm10, %ymm11
    vpcmpgtb %ymm10, %ymm4, %ymm12
    vpand   %ymm5, %ymm11, %ymm11
    vpand   %ymm5, %ymm12, %ymm12
    vpsubb  %ymm12, %ymm11, %ymm7      /* ymm7 = next row trits */
    jmp     .Lcompute_grads_turbo

.Llast_row_turbo:
    vpxor   %ymm7, %ymm7, %ymm7

.Lcompute_grads_turbo:
    /* 3. Compute Gradients */
    /* H-grad: clamp(trit[x+1] - trit[x]) */
    vpsrldq $1, %ymm6, %ymm10
    vpsubb  %ymm6, %ymm10, %ymm8
    vpsignb %ymm8, %ymm5, %ymm8

    /* V-grad: clamp(next - curr) */
    vpsubb  %ymm6, %ymm7, %ymm9
    vpsignb %ymm9, %ymm5, %ymm9

    /* Store everything to stack to avoid vpextrb issues with YMM lanes */
    vmovdqu %ymm6, 64(%rsp)
    vmovdqu %ymm8, 96(%rsp)
    vmovdqu %ymm9, 128(%rsp)
    /* Mask last pixel of h-grad on stack */
    movb    $0, 123(%rsp)

    /* ---- Strip loop: 9 strips ---- */
    xor     %r8d, %r8d
    mov     %rbx, %r11

.Lstrip_loop_turbo:
    lea     (%r8, %r8, 2), %r9d         /* r9 = strip * 3 */

    /* ---- Pixel Channel ---- */
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
    jz      .Lskip_px_turbo
    shl     $6, %rax
    add     %r11, %rax
    vpaddd  (%r13, %rax), %ymm0, %ymm0
    vpaddd  32(%r13, %rax), %ymm1, %ymm1

.Lskip_px_turbo:
    /* ---- H-Grad Channel ---- */
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
    je      .Lskip_hg_turbo
    shl     $6, %rax
    add     %r11, %rax
    vpaddd  (%r14, %rax), %ymm0, %ymm0
    vpaddd  32(%r14, %rax), %ymm1, %ymm1

.Lskip_hg_turbo:
    /* ---- V-Grad Channel ---- */
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
    je      .Lskip_vg_turbo
    shl     $6, %rax
    add     %r11, %rax
    vpaddd  (%r15, %rax), %ymm0, %ymm0
    vpaddd  32(%r15, %rax), %ymm1, %ymm1

.Lskip_vg_turbo:
    add     $1728, %r11
    inc     %r8d
    cmp     $9, %r8d
    jl      .Lstrip_loop_turbo

    add     $15552, %ebx        /* 9 * 1728 */
    inc     %ebp
    cmp     $28, %ebp
    jl      .Lrow_loop

    /* ---- Argmax ---- */
    vmovdqa %ymm0, (%rsp)
    vmovdqa %ymm1, 32(%rsp)
    xor     %eax, %eax
    mov     (%rsp), %r8d
    mov     $1, %ecx
.Largmax_loop_turbo:
    cmp     $10, %ecx
    jge     .Largmax_done_turbo
    mov     (%rsp, %rcx, 4), %r9d
    cmp     %r8d, %r9d
    jle     .Largmax_next_turbo
    mov     %r9d, %r8d
    mov     %ecx, %eax
.Largmax_next_turbo:
    inc     %ecx
    jmp     .Largmax_loop_turbo
.Largmax_done_turbo:

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
