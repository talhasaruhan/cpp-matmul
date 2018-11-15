//#include <iostream>
//#include <array>
//#include <chrono>
//#include <string.h>
//#include <xmmintrin.h>
//#include <emmintrin.h>
//#include <immintrin.h>
//
//using namespace std;
//
//#define AVX_ALIGNMENT 32
//
///* naive sum using intrinsics */
//float VecSumIntrinsicNaiveLoop(const float* const __restrict  c, const unsigned N)
//{
//    _declspec(align(32)) float vsum[8];
//    for (int i = 0; i<8; ++i) vsum[i] = 0;
//
//    __m256 sum = _mm256_setzero_ps();
//    __m256 x0, x1;
//
//    for (int i = 0; i<N; i += 16) {
//        x0 = _mm256_load_ps(&c[i]);
//        x1 = _mm256_load_ps(&c[i + 8]);
//        x0 = _mm256_add_ps(x0, x1);
//        sum = _mm256_add_ps(sum, x0);
//    }
//
//    _mm256_store_ps(&vsum[0], sum);
//
//    float acc = 0;
//    for (int i = 0; i<8; ++i) {
//        acc += vsum[i];
//    }
//
//    return acc;
//}
//
///* 
//* Load 16x8f vectors, and sum them, hierarchically, two by two. 
//* Note that towards the end, we're giving up on Intruction Level Parallelism as ops get increasingly dependent on each other.
//*/
//float VecSumIntrinsicExplicit1(const float* const __restrict c, const unsigned N)
//{
//    _declspec(align(32)) float vsum[8];
//    for (int i = 0; i<8; ++i) vsum[i] = 0;
//
//    /* this will probably be written on stack by the compiler as we use all 16 registers */
//    __m256 sum = _mm256_setzero_ps();
//
//    __m256 c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16;
//
//    for (int i = 0; i<N; i += 128) {
//        c1 = _mm256_load_ps(&c[i + 0]);
//        c2 = _mm256_load_ps(&c[i + 8]);
//        c3 = _mm256_load_ps(&c[i + 16]);
//        c4 = _mm256_load_ps(&c[i + 24]);
//        c5 = _mm256_load_ps(&c[i + 32]);
//        c6 = _mm256_load_ps(&c[i + 40]);
//        c7 = _mm256_load_ps(&c[i + 48]);
//        c8 = _mm256_load_ps(&c[i + 56]);
//        c9 = _mm256_load_ps(&c[i + 64]);
//        c10 = _mm256_load_ps(&c[i + 72]);
//        c11 = _mm256_load_ps(&c[i + 80]);
//        c12 = _mm256_load_ps(&c[i + 88]);
//        c13 = _mm256_load_ps(&c[i + 96]);
//        c14 = _mm256_load_ps(&c[i + 104]);
//        c15 = _mm256_load_ps(&c[i + 112]);
//        c16 = _mm256_load_ps(&c[i + 120]);
//
//        c1 = _mm256_add_ps(c1, c2);
//        c3 = _mm256_add_ps(c3, c4);
//        c5 = _mm256_add_ps(c5, c6);
//        c7 = _mm256_add_ps(c7, c8);
//        c9 = _mm256_add_ps(c9, c10);
//        c11 = _mm256_add_ps(c11, c12);
//        c13 = _mm256_add_ps(c13, c12);
//        c15 = _mm256_add_ps(c15, c16);
//
//        c1 = _mm256_add_ps(c1, c3);
//        c5 = _mm256_add_ps(c5, c7);
//        c9 = _mm256_add_ps(c9, c11);
//        c13 = _mm256_add_ps(c13, c15);
//
//        c1 = _mm256_add_ps(c1, c5);
//        c9 = _mm256_add_ps(c9, c13);
//
//        c1 = _mm256_add_ps(c1, c9);
//
//        sum = _mm256_add_ps(sum, c1);
//    }
//
//    _mm256_store_ps(&vsum[0], sum);
//
//    float acc = 0;
//    for (int i = 0; i<8; ++i) {
//        acc += vsum[i];
//    }
//
//    return acc;
//}
//
///*
//* Load 8x8f vectors, accumulate into 4x8f vecs. The upside is that we don't have interdependencies.
//* The downside is obvious, we only process 8 vecs at a time and there are effectively 4 wasted registers
//*/
//float VecSumIntrinsicExplicit2(const float* const __restrict c, const unsigned N)
//{
//    _declspec(align(32)) float vsum[8];
//    for (int i = 0; i<8; ++i) vsum[i] = 0;
//    __m256 sum = _mm256_setzero_ps();
//
//    __m256 c1 = _mm256_setzero_ps();
//    __m256 c2 = _mm256_setzero_ps();
//    __m256 c3 = _mm256_setzero_ps();
//    __m256 c4 = _mm256_setzero_ps();
//    __m256 c5 = _mm256_setzero_ps();
//    __m256 c6 = _mm256_setzero_ps();
//    __m256 c7 = _mm256_setzero_ps();
//    __m256 c8 = _mm256_setzero_ps();
//    __m256 c9 = _mm256_setzero_ps();
//    __m256 c10 = _mm256_setzero_ps();
//    __m256 c11 = _mm256_setzero_ps();
//    __m256 c12 = _mm256_setzero_ps();
//    __m256 c13 = _mm256_setzero_ps();
//    __m256 c14 = _mm256_setzero_ps();
//    __m256 c15 = _mm256_setzero_ps();
//    __m256 c16 = _mm256_setzero_ps();
//
//    // keep sums at 4 registers
//    for (int i = 0; i<N; i += 64) {
//        c9 = _mm256_load_ps(&c[i + 0]);
//        c10 = _mm256_load_ps(&c[i + 8]);
//        c11 = _mm256_load_ps(&c[i + 16]);
//        c12 = _mm256_load_ps(&c[i + 24]);
//        c13 = _mm256_load_ps(&c[i + 32]);
//        c14 = _mm256_load_ps(&c[i + 40]);
//        c15 = _mm256_load_ps(&c[i + 48]);
//        c16 = _mm256_load_ps(&c[i + 56]);
//
//        c5 = _mm256_add_ps(c9, c10);
//        c6 = _mm256_add_ps(c11, c12);
//        c7 = _mm256_add_ps(c13, c14);
//        c8 = _mm256_add_ps(c15, c16);
//
//        c1 = _mm256_add_ps(c1, c5);
//        c2 = _mm256_add_ps(c2, c6);
//        c3 = _mm256_add_ps(c3, c7);
//        c4 = _mm256_add_ps(c4, c8);
//    }
//
//    c1 = _mm256_add_ps(c1, c2);
//    c3 = _mm256_add_ps(c3, c4);
//
//    sum = _mm256_add_ps(c1, c3);
//
//    _mm256_store_ps(&vsum[0], sum);
//
//    float acc = 0;
//    for (int i = 0; i<8; ++i) {
//        acc += vsum[i];
//    }
//
//    return acc;
//}
//
///*
//* Spoiler: While I think this idea is cool, it'll only perform worse than the previous method.
//*
//* This is a different idea, like a rolling accumulator if that makes sense.
//* We reduce-sum to 8 vectors to 4 in one iteration, in the next iteration, we sum them up again and accumulate into 2 vectors.
//* The upside is that since we're working on independent data, we can take advantage of ILP.
//*
//* Imagine registers hypothetically divided as such:
//* [ ymm0 | ymm1 ] [ ymm2 | ymm3 ] [ ymm4 | ymm5  | ymm6 | ymm7 ] [ ymm8:15 ]
//* 
//* in an iteration:
//* reduce sum ymm4:7 to ymm0:1, then accumulate results onto ymm2:3
//* load a1:8 to ymm8:15, reduce-sum to ymm4:7
//*
//*/
//float VecSumIntrinsicExplicit3(const float* const __restrict c, const unsigned N)
//{
//    _declspec(align(32)) float vsum[8];
//    for (int i = 0; i<8; ++i) vsum[i] = 0;
//    __m256 sum = _mm256_setzero_ps();
//
//    __m256 c1 = _mm256_setzero_ps();
//    __m256 c2 = _mm256_setzero_ps();
//    __m256 c3 = _mm256_setzero_ps();
//    __m256 c4 = _mm256_setzero_ps();
//    __m256 c5 = _mm256_setzero_ps();
//    __m256 c6 = _mm256_setzero_ps();
//    __m256 c7 = _mm256_setzero_ps();
//    __m256 c8 = _mm256_setzero_ps();
//    __m256 c9 = _mm256_setzero_ps();
//    __m256 c10 = _mm256_setzero_ps();
//    __m256 c11 = _mm256_setzero_ps();
//    __m256 c12 = _mm256_setzero_ps();
//    __m256 c13 = _mm256_setzero_ps();
//    __m256 c14 = _mm256_setzero_ps();
//    __m256 c15 = _mm256_setzero_ps();
//    __m256 c16 = _mm256_setzero_ps();
//
//    for (int i = 0; i<N; i += 64) {
//        /* load new data */
//        c9 = _mm256_load_ps(&c[i + 0]);
//        c10 = _mm256_load_ps(&c[i + 8]);
//        c11 = _mm256_load_ps(&c[i + 16]);
//        c12 = _mm256_load_ps(&c[i + 24]);
//        c13 = _mm256_load_ps(&c[i + 32]);
//        c14 = _mm256_load_ps(&c[i + 40]);
//        c15 = _mm256_load_ps(&c[i + 48]);
//        c16 = _mm256_load_ps(&c[i + 56]);
//
//        /* reduce sum previous iterations 4 vecs into 2 */
//        c1 = _mm256_add_ps(c5, c6);
//        c2 = _mm256_add_ps(c7, c8);
//
//        /* dependent on the c1,c2 ops above, accumulate c1,c2 into c3,c4 */
//        c3 = _mm256_add_ps(c3, c1);
//        c4 = _mm256_add_ps(c4, c2);
//
//        /* reduce-sum new data into c5:8, 
//        notice that these ops are not dependent on the 2 ops above, so they can be excecuted in parallel. */
//        c5 = _mm256_add_ps(c9, c10);
//        c6 = _mm256_add_ps(c11, c12);
//        c7 = _mm256_add_ps(c13, c14);
//        c8 = _mm256_add_ps(c15, c16);
//    }
//
//    c1 = _mm256_add_ps(c5, c6);
//    c2 = _mm256_add_ps(c7, c8);
//    c3 = _mm256_add_ps(c3, c1);
//    c4 = _mm256_add_ps(c4, c2);
//    sum = _mm256_add_ps(c3, c4);
//
//    _mm256_store_ps(&vsum[0], sum);
//
//    float acc = 0;
//    for (int i = 0; i<8; ++i) {
//        acc += vsum[i];
//    }
//
//    return acc;
//}
//
///* just an idea, sum the vec as if you'd sum nodes in a binary tree, bottom up, not expected to work well */
//float VecSumIntrinsicBinary(float* const c, const unsigned N, const unsigned K)
//{
//    __m256 c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16;
//
//    for (int k = 0; k <= K - 7; ++k) {
//        for (int i = 0; i<N / (1 << k); i += 128) {
//            c1 = _mm256_load_ps(&c[i + 0]);
//            c2 = _mm256_load_ps(&c[i + 8]);
//            c3 = _mm256_load_ps(&c[i + 16]);
//            c4 = _mm256_load_ps(&c[i + 24]);
//            c5 = _mm256_load_ps(&c[i + 32]);
//            c6 = _mm256_load_ps(&c[i + 40]);
//            c7 = _mm256_load_ps(&c[i + 48]);
//            c8 = _mm256_load_ps(&c[i + 56]);
//            c9 = _mm256_load_ps(&c[i + 64]);
//            c10 = _mm256_load_ps(&c[i + 72]);
//            c11 = _mm256_load_ps(&c[i + 80]);
//            c12 = _mm256_load_ps(&c[i + 88]);
//            c13 = _mm256_load_ps(&c[i + 96]);
//            c14 = _mm256_load_ps(&c[i + 104]);
//            c15 = _mm256_load_ps(&c[i + 112]);
//            c16 = _mm256_load_ps(&c[i + 120]);
//
//            c1 = _mm256_add_ps(c1, c2);
//            c2 = _mm256_add_ps(c3, c4);
//            c3 = _mm256_add_ps(c5, c6);
//            c4 = _mm256_add_ps(c7, c8);
//            c5 = _mm256_add_ps(c9, c10);
//            c6 = _mm256_add_ps(c11, c12);
//            c7 = _mm256_add_ps(c13, c12);
//            c8 = _mm256_add_ps(c15, c16);
//
//            const unsigned j = i >> 1;
//            _mm256_store_ps(&c[j + 0], c1);
//            _mm256_store_ps(&c[j + 8], c2);
//            _mm256_store_ps(&c[j + 16], c3);
//            _mm256_store_ps(&c[j + 24], c4);
//            _mm256_store_ps(&c[j + 32], c5);
//            _mm256_store_ps(&c[j + 40], c6);
//            _mm256_store_ps(&c[j + 48], c7);
//            _mm256_store_ps(&c[j + 56], c8);
//        }
//    }
//
//    return VecSumIntrinsicNaiveLoop(c, 64);
//}
//
///* scalar sum */
//float VecSumScalarAccumulate(const float* const __restrict c, const unsigned N) {
//    /*
//    * compiler optimizes this by keeping t in an xmm register
//    * s.t at every iteration, we do 1 load and 1 add
//    * but t <- add(t, ai) is obviously dependent on t
//    * so there goes the ILP.
//    */
//
//    float t = 0;
//    for (int i = 0; i<N; ++i) {
//        t += c[i];
//    }
//    return t;
//}
//
///* binary idea, scalar case. Note that this performs way better than above for some optimization levels. */
//float VecSumScalarBinary(float* c, const unsigned N, const unsigned K) {
//    /*
//    * Note that this yields more instructions (2 loads, 1 add, 1 store)
//    * but since we don't have data dependency between iterations,
//    * we can fully utilize ILP.
//    */
//
//    for (int k = 1; k <= K; ++k) {
//        for (int i = 0; i<N / (1 << k); ++i) {
//            c[i] = c[2 * i] + c[2 * i + 1];
//        }
//    }
//
//    return c[0];
//}
//
//void ILPSum() {
//    const unsigned Ts = 1, T = 1000;
//    const unsigned K = 20;
//    const unsigned N = 1 << K;
//
//    float* ar = (float*)_aligned_malloc(N * sizeof(float), AVX_ALIGNMENT);
//    float* ar_cpy = (float*)_aligned_malloc(N * sizeof(float), AVX_ALIGNMENT);
//
//    for (int i = 0; i<N; ++i) {
//        ar[i] = 0.005;
//    }
//
//    float t1 = 0, t2 = 0, t3 = 0, t4 = 0, t5 = 0, t6 = 0, t7 = 0, t8 = 0, t9 = 0;
//
//    /*****************************************************/
//
//    auto start = std::chrono::high_resolution_clock::now();
//    for (int i = 0; i<T; ++i)
//        t1 += VecSumScalarAccumulate(ar, N);
//    auto end = std::chrono::high_resolution_clock::now();
//    std::cout << "C++ Accumulative sum: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " milliseconds.\n";
//
//    /*****************************************************/
//
//    //memcpy(ar_cpy, ar, N * sizeof(float));
//
//    //start = std::chrono::high_resolution_clock::now();
//    //t2 = VecSumScalarBinary(ar_cpy, N, K);
//    //end = std::chrono::high_resolution_clock::now();
//    //std::cout << "C++ Binary sum: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " milliseconds.\n";
//
//    /*****************************************************/
//
//    start = std::chrono::high_resolution_clock::now();
//    for (int i = 0; i<T; ++i)
//        t3 += VecSumIntrinsicNaiveLoop(ar, N);
//    end = std::chrono::high_resolution_clock::now();
//    std::cout << "Intrinsic naive sum: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " milliseconds.\n";
//
//    /*****************************************************/
//
//    start = std::chrono::high_resolution_clock::now();
//    for (int i = 0; i<T; ++i)
//        t4 += VecSumIntrinsicExplicit1(ar, N);
//    end = std::chrono::high_resolution_clock::now();
//    std::cout << "Intrinsic unrolled sum: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " milliseconds.\n";
//
//    /*****************************************************/
//
//    start = std::chrono::high_resolution_clock::now();
//    for (int i = 0; i<T; ++i)
//        t5 += VecSumIntrinsicExplicit2(ar, N);
//    end = std::chrono::high_resolution_clock::now();
//    std::cout << "Intrinsic unrolled sum 2: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " milliseconds.\n";
//
//    /*****************************************************/
//
//    start = std::chrono::high_resolution_clock::now();
//    for (int i = 0; i<T; ++i)
//        t6 += VecSumIntrinsicExplicit3(ar, N);
//    end = std::chrono::high_resolution_clock::now();
//    std::cout << "Intrinsic unrolled sum 3: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " milliseconds.\n";
//
//    /*****************************************************/
//
//    cout << t1 << endl;
//    cout << t3 << endl;
//    cout << t4 << endl;
//    cout << t5 << endl;
//    cout << t6 << endl;
//}
//
//int main() {
//    ILPSum();
//}