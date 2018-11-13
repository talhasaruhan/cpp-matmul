#include <iostream>
#include <array>
#include <chrono>
#include <string.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <immintrin.h>
#include <stdlib.h>
#include <intrin.h>
#include <malloc.h>
#include <cassert>

using namespace std;

#define AVX_ALIGNMENT 32

float VecDotIntrinsicExplicit1(float* const a, float* const b, const unsigned N)
{
    float* vsum = (float*)aligned_alloc(8 * sizeof(float), AVX_ALIGNMENT);
    for (int i = 0; i<8; ++i) vsum[i] = 0;

    __m256 sum = _mm256_setzero_ps();
    __m256 a1, a2, a3, a4, a5, a6, a7, a8;
    __m256 b1, b2, b3, b4, b5, b6, b7, b8;

    for (int i = 0; i<N; i += 64) {
        a1 = _mm256_load_ps(&a[i + 0]);
        a2 = _mm256_load_ps(&a[i + 8]);
        a3 = _mm256_load_ps(&a[i + 16]);
        a4 = _mm256_load_ps(&a[i + 24]);
        a5 = _mm256_load_ps(&a[i + 32]);
        a6 = _mm256_load_ps(&a[i + 40]);
        a7 = _mm256_load_ps(&a[i + 48]);
        a8 = _mm256_load_ps(&a[i + 56]);

        b1 = _mm256_load_ps(&b[i + 0]);
        b2 = _mm256_load_ps(&b[i + 8]);
        b3 = _mm256_load_ps(&b[i + 16]);
        b4 = _mm256_load_ps(&b[i + 24]);
        b5 = _mm256_load_ps(&b[i + 32]);
        b6 = _mm256_load_ps(&b[i + 40]);
        b7 = _mm256_load_ps(&b[i + 48]);
        b8 = _mm256_load_ps(&b[i + 56]);

        a1 = _mm256_mul_ps(a1, b1);
        a2 = _mm256_mul_ps(a2, b2);
        a3 = _mm256_mul_ps(a3, b3);
        a4 = _mm256_mul_ps(a4, b4);
        a5 = _mm256_mul_ps(a5, b5);
        a6 = _mm256_mul_ps(a6, b6);
        a7 = _mm256_mul_ps(a7, b7);
        a8 = _mm256_mul_ps(a8, b8);

        a1 = _mm256_add_ps(a1, a2);
        a3 = _mm256_add_ps(a3, a4);
        a5 = _mm256_add_ps(a5, a6);
        a7 = _mm256_add_ps(a7, a8);

        a1 = _mm256_add_ps(a1, a3);
        a5 = _mm256_add_ps(a5, a7);

        a1 = _mm256_add_ps(a1, a5);

        sum = _mm256_add_ps(sum, a1);
    }

    _mm256_store_ps(&vsum[0], sum);

    float acc = 0;
    for (int i = 0; i<8; ++i) {
        acc += vsum[i];
    }

    return acc;
}

float VecDotIntrinsicExplicit2(float* const a, float* const b, const unsigned N)
{
    float* vsum = (float*)aligned_alloc(8 * sizeof(float), AVX_ALIGNMENT);
    for (int i = 0; i<8; ++i) vsum[i] = 0;

    __m256 a1, a2, a3, a4;
    __m256 b1, b2, b3, b4;
    __m256 c1 = _mm256_setzero_ps();
    __m256 c2 = _mm256_setzero_ps();
    __m256 c3 = _mm256_setzero_ps();
    __m256 c4 = _mm256_setzero_ps();

    for (int i = 0; i<N; i += 32) {
        a1 = _mm256_load_ps(&a[i + 0]);
        a2 = _mm256_load_ps(&a[i + 8]);
        a3 = _mm256_load_ps(&a[i + 16]);
        a4 = _mm256_load_ps(&a[i + 24]);

        b1 = _mm256_load_ps(&b[i + 0]);
        b2 = _mm256_load_ps(&b[i + 8]);
        b3 = _mm256_load_ps(&b[i + 16]);
        b4 = _mm256_load_ps(&b[i + 24]);

        a1 = _mm256_mul_ps(a1, b1);
        a2 = _mm256_mul_ps(a2, b2);
        a3 = _mm256_mul_ps(a3, b3);
        a4 = _mm256_mul_ps(a4, b4);

        c1 = _mm256_add_ps(c1, a1);
        c2 = _mm256_add_ps(c2, a2);
        c3 = _mm256_add_ps(c3, a3);
        c4 = _mm256_add_ps(c4, a4);
    }

    c1 = _mm256_add_ps(c1, c2);
    c3 = _mm256_add_ps(c3, c4);

    c1 = _mm256_add_ps(c1, c3);

    _mm256_store_ps(&vsum[0], c1);

    float acc = 0;
    for (int i = 0; i<8; ++i) {
        acc += vsum[i];
    }

    return acc;
}

float VecDotIntrinsicExplicit3(float* const a, float* const b, const unsigned N)
{
    float* vsum = (float*)aligned_alloc(8 * sizeof(float), AVX_ALIGNMENT);
    for (int i = 0; i<8; ++i) vsum[i] = 0;

    __m256 a1, a2, a3, a4, a5, a6;
    __m256 b1, b2, b3, b4, b5, b6;
    __m256 c1 = _mm256_setzero_ps();
    __m256 c2 = _mm256_setzero_ps();
    __m256 c3 = _mm256_setzero_ps();
    __m256 c4 = _mm256_setzero_ps();

    int i = 0;
    for (; i<N - 48; i += 48) {
        a1 = _mm256_load_ps(&a[i + 0]);
        a2 = _mm256_load_ps(&a[i + 8]);
        a3 = _mm256_load_ps(&a[i + 16]);
        a4 = _mm256_load_ps(&a[i + 24]);
        a5 = _mm256_load_ps(&a[i + 32]);
        a6 = _mm256_load_ps(&a[i + 40]);

        b1 = _mm256_load_ps(&b[i + 0]);
        b2 = _mm256_load_ps(&b[i + 8]);
        b3 = _mm256_load_ps(&b[i + 16]);
        b4 = _mm256_load_ps(&b[i + 24]);
        b5 = _mm256_load_ps(&b[i + 32]);
        b6 = _mm256_load_ps(&b[i + 40]);

        a1 = _mm256_mul_ps(a1, b1);
        a2 = _mm256_mul_ps(a2, b2);
        a3 = _mm256_mul_ps(a3, b3);
        a4 = _mm256_mul_ps(a4, b4);
        a5 = _mm256_mul_ps(a5, b5);
        a6 = _mm256_mul_ps(a6, b6);

        a1 = _mm256_add_ps(a1, a2);
        a3 = _mm256_add_ps(a3, a4);
        a5 = _mm256_add_ps(a5, a6);

        c1 = _mm256_add_ps(c1, a1);
        c2 = _mm256_add_ps(c2, a3);
        c3 = _mm256_add_ps(c3, a5);
    }
    for (; i<N; i += 16) {
        a1 = _mm256_load_ps(&a[i + 0]);
        a2 = _mm256_load_ps(&a[i + 8]);

        b1 = _mm256_load_ps(&b[i + 0]);
        b2 = _mm256_load_ps(&b[i + 8]);

        a1 = _mm256_mul_ps(a1, b1);
        a2 = _mm256_mul_ps(a2, b2);

        c1 = _mm256_add_ps(c1, a1);
        c2 = _mm256_add_ps(c2, a2);
    }

    c1 = _mm256_add_ps(c1, c2);
    c1 = _mm256_add_ps(c1, c3);

    _mm256_store_ps(&vsum[0], c1);

    float acc = 0;
    for (int i = 0; i<8; ++i) {
        acc += vsum[i];
    }

    return acc;
}


float VecDotASMExplicit1(float* const a, float* const b, const unsigned N)
{
    float* vsum = (float*)aligned_alloc(8 * sizeof(float), AVX_ALIGNMENT);
    for (int i = 0; i<8; ++i) vsum[i] = 0;

    for (int i = 0; i<N; i += 64) {
        asm volatile("vmovaps ymm0, ymmword ptr [%0];" : : "r"(&a[i + 0]) : );
        asm volatile("vmovaps ymm1, ymmword ptr [%0];" : : "r"(&a[i + 8]) : );
        asm volatile("vmovaps ymm2, ymmword ptr [%0];" : : "r"(&a[i + 16]) : );
        asm volatile("vmovaps ymm3, ymmword ptr [%0];" : : "r"(&a[i + 24]) : );
        asm volatile("vmovaps ymm4, ymmword ptr [%0];" : : "r"(&a[i + 32]) : );
        asm volatile("vmovaps ymm5, ymmword ptr [%0];" : : "r"(&a[i + 40]) : );
        asm volatile("vmovaps ymm6, ymmword ptr [%0];" : : "r"(&a[i + 48]) : );
        asm volatile("vmovaps ymm7, ymmword ptr [%0];" : : "r"(&a[i + 56]) : );

        asm volatile("vmovaps ymm8, ymmword ptr [%0];" : : "r"(&b[i + 0]) : );
        asm volatile("vmovaps ymm9, ymmword ptr [%0];" : : "r"(&b[i + 8]) : );
        asm volatile("vmovaps ymm10, ymmword ptr [%0];" : : "r"(&b[i + 16]) : );
        asm volatile("vmovaps ymm11, ymmword ptr [%0];" : : "r"(&b[i + 24]) : );
        asm volatile("vmovaps ymm12, ymmword ptr [%0];" : : "r"(&b[i + 32]) : );
        asm volatile("vmovaps ymm13, ymmword ptr [%0];" : : "r"(&b[i + 40]) : );
        asm volatile("vmovaps ymm14, ymmword ptr [%0];" : : "r"(&b[i + 48]) : );
        asm volatile("vmovaps ymm15, ymmword ptr [%0];" : : "r"(&b[i + 56]) : );

        asm volatile("vmulps ymm0, ymm0, ymm8");
        asm volatile("vmulps ymm1, ymm1, ymm9");
        asm volatile("vmulps ymm2, ymm2, ymm10");
        asm volatile("vmulps ymm3, ymm3, ymm11");
        asm volatile("vmulps ymm4, ymm4, ymm12");
        asm volatile("vmulps ymm5, ymm5, ymm13");
        asm volatile("vmulps ymm6, ymm6, ymm14");
        asm volatile("vmulps ymm7, ymm7, ymm15");

        asm volatile("vaddps ymm0, ymm0, ymm1");
        asm volatile("vaddps ymm2, ymm2, ymm3");
        asm volatile("vaddps ymm4, ymm4, ymm5");
        asm volatile("vaddps ymm6, ymm6, ymm7");

        asm volatile("vaddps ymm0, ymm0, ymm2");
        asm volatile("vaddps ymm4, ymm4, ymm6");

        asm volatile("vaddps ymm0, ymm0, ymm4");

        asm volatile("vmovaps ymm15, ymmword ptr [%0];" : : "r"(vsum) : );
        asm volatile("vaddps ymm15, ymm15, ymm0");
        asm volatile("vmovaps ymmword ptr [%0], ymm15;" : : "r"(vsum) : );
    }

    // asm volatilevolatile("vzeroall");
    _mm256_zeroall();

    float acc = 0;
    for (int i = 0; i<8; ++i) {
        acc += vsum[i];
    }

    return acc;
}

float VecDotASMExplicit2(float* const a, float* const b, const unsigned N)
{
    float* vsum = (float*)aligned_alloc(8 * sizeof(float), AVX_ALIGNMENT);
    for (int i = 0; i<8; ++i) vsum[i] = 0;

    for (int i = 0; i<N; i += 32) {
        asm volatile("vmovaps ymm0, ymmword ptr [%0];" : : "r"(&a[i + 0]) : );
        asm volatile("vmovaps ymm1, ymmword ptr [%0];" : : "r"(&a[i + 8]) : );
        asm volatile("vmovaps ymm2, ymmword ptr [%0];" : : "r"(&a[i + 16]) : );
        asm volatile("vmovaps ymm3, ymmword ptr [%0];" : : "r"(&a[i + 24]) : );

        asm volatile("vmovaps ymm4, ymmword ptr [%0];" : : "r"(&b[i + 0]) : );
        asm volatile("vmovaps ymm5, ymmword ptr [%0];" : : "r"(&b[i + 8]) : );
        asm volatile("vmovaps ymm6, ymmword ptr [%0];" : : "r"(&b[i + 16]) : );
        asm volatile("vmovaps ymm7, ymmword ptr [%0];" : : "r"(&b[i + 24]) : );

        asm volatile("vmulps ymm0, ymm0, ymm4");
        asm volatile("vmulps ymm1, ymm1, ymm5");
        asm volatile("vmulps ymm2, ymm2, ymm6");
        asm volatile("vmulps ymm3, ymm3, ymm7");

        asm volatile("vaddps ymm8, ymm8, ymm0");
        asm volatile("vaddps ymm9, ymm9, ymm1");
        asm volatile("vaddps ymm10, ymm10, ymm2");
        asm volatile("vaddps ymm11, ymm11, ymm3");
    }


    asm volatile("vaddps ymm8, ymm8, ymm9");
    asm volatile("vaddps ymm10, ymm10, ymm11");

    asm volatile("vaddps ymm8, ymm8, ymm10");

    asm volatile("vmovaps ymmword ptr [%0], ymm8;" : : "r"(vsum) : );

    // asm volatile("vzeroall");
    _mm256_zeroall();

    float acc = 0;
    for (int i = 0; i<8; ++i) {
        acc += vsum[i];
    }

    return acc;
}


void ILPSum() {
    const unsigned T = 1000;
    const unsigned K = 20;
    const unsigned N = 1 << K;

    cout << N << "\n";

    float* ar = (float*)aligned_alloc(N * sizeof(float), AVX_ALIGNMENT);
    float* ar2 = (float*)aligned_alloc(N * sizeof(float), AVX_ALIGNMENT);

    for (int i = 0; i<N; ++i) {
        ar[i] = 1;
        ar2[i] = 2;
    }

    float t1 = 0, t2 = 0, t3 = 0, t4 = 0, t5 = 0, t6 = 0, t7 = 0, t8 = 0, t9 = 0;

    /*****************************************************/

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i<T; ++i)
        t1 += VecDotASMExplicit2(ar, ar2, N);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Dot ASM volatile explicit 2: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " milliseconds.\n";

    /*****************************************************/

    cout << t1 << endl;
}

//int main() {
//    ILPSum();
//}