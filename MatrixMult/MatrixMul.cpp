#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include <chrono>
#include <sstream>
#include <iostream>
#include <fstream>
#include <cstdint>
#include <random>
#include <functional>
#include <cstdio>
#include <memory>
#include <mutex>
#include <thread>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <immintrin.h>
#include "ThreadPool.h"

#define NOMINMAX

#define AVX_ALIGN 32
#define SSE_ALIGN 16

constexpr unsigned L3BlockX = 32, L3BlockY = 64;
constexpr unsigned L2BlockX = 2, L2BlockY = 4;
constexpr unsigned cacheLineSz = 64;

constexpr int doPrefetch = 1;
int prefetched[1024][1024];

std::mutex prefetchMutex;

typedef struct Mat
{
    unsigned width;
    unsigned height;
    unsigned rowSpan;
    float *mat;
} Mat;

void FreeMat(const Mat &mat) {
    _aligned_free(mat.mat);
}

const Mat LoadMat(const char * const filename) {
    Mat mat;
    uint32_t matSize;

    std::ifstream in(filename, std::ios::binary | std::ios::in);

    if (!in.is_open()) {
        std::cout << "Err loading!\n";
        in.close();
        return {};
    }

    in.read((char*)&mat, 3 * sizeof(uint32_t));
    in.read((char*)&matSize, sizeof(uint32_t));
    in.seekg(12 * sizeof(uint32_t), std::ios::cur);
    mat.mat = (float*)_aligned_malloc(matSize, AVX_ALIGN);
    in.read((char*)mat.mat, matSize);

    in.close();

    return mat;
}

static void DumpMat(const char *filename, const Mat &m)
{
    uint32_t header[16];
    std::ofstream out(filename, std::ofstream::binary | std::ofstream::out);

    header[0] = m.width;
    header[1] = m.height;
    header[2] = m.rowSpan;
    header[3] = m.height * m.rowSpan * sizeof(float);

    out.write(reinterpret_cast<const char*>(header), sizeof(header));
    out.write(reinterpret_cast<const char*>(m.mat), header[3]);

    out.close();
}

static unsigned RoundUpPwr2(unsigned val, unsigned pwr2)
{
    return (val + (pwr2 - 1)) & (~(pwr2 - 1));
}

/* Single threaded, should I multithread this as well?
Honestly, I don't think it will have any significant effect. n^2 vs n^3 */
__declspec(noalias)
const Mat TransposeMat(const Mat &mat) {
    const unsigned tRowSpan = RoundUpPwr2(mat.height, 64 / sizeof(float));
    float * __restrict const tData = (float*)_aligned_malloc(mat.width*tRowSpan * sizeof(float), AVX_ALIGN);

    Mat T{
        mat.height,
        mat.width,
        tRowSpan,
        tData
    };

    // hah, the loops are truly interchangable as we encounter a cache miss either ways
    for (int rowT = 0; rowT < T.height; ++rowT) {
        for (int colT = 0; colT < T.width; ++colT) {
            tData[rowT*tRowSpan + colT] = mat.mat[colT*mat.rowSpan + rowT];
        }
    }

    return T;
}

static void PrintMat(const Mat &mat) {
    printf("w, h, rS: %d %d %d\n", mat.width, mat.height, mat.rowSpan);
    for (int i = 0; i < mat.height; i++) {
        for (int j = 0; j < mat.width; ++j) {
            printf("%f ", mat.mat[i*mat.rowSpan + j]);
        }
        printf("\n");
    }
}

const Mat ST_NaiveMatMul(const Mat& matA, const Mat& matB) {
    /*
    * First : naive solution with but with some tricks to make compiler (MVC) behave
    * Note that, in this case, manually unrolling the loop helps as the compiler can't auto-vectorize non-contagious memory access
    */
    float * __restrict const matData = (float*)_aligned_malloc(matA.height * matB.rowSpan * sizeof(float), AVX_ALIGN);

    Mat matC{
        matB.width,
        matA.height,
        matB.rowSpan,
        matData
    };

    for (int rowC = 0; rowC < matA.height; ++rowC) {
        for (int colC = 0; colC < matB.width; ++colC) {
            // have a local accumulator, o.w compiler fetches the value at each += operator.
            float accumulate = 0;
            int pos = 0;
            // interestingly, manual unrolling IS helpful, it takes 1000x1000 multiplication from about 990ms to 710ms
            for (; pos < matA.width - 4; pos += 4) {
                accumulate += matA.mat[rowC*matA.rowSpan + pos] * matB.mat[pos*matB.rowSpan + colC] +
                    matA.mat[rowC*matA.rowSpan + pos + 1] * matB.mat[(pos + 1)*matB.rowSpan + colC] +
                    matA.mat[rowC*matA.rowSpan + pos + 2] * matB.mat[(pos + 2)*matB.rowSpan + colC] +
                    matA.mat[rowC*matA.rowSpan + pos + 3] * matB.mat[(pos + 3)*matB.rowSpan + colC];
            }
            for (; pos < matA.width ; ++pos) {
                accumulate += matA.mat[rowC*matA.rowSpan + pos] * matB.mat[pos*matB.rowSpan + colC];
            }
            matData[rowC*matB.rowSpan + colC] = accumulate;
        }
    }
 
    return matC;
}

const Mat ST_TransposedBMatMul(const Mat& matA, const Mat& matB) {
    /*
    * Now, I thought transposing B and then traversing it row order would help and it does!
    * Also, note that, if we manually unrolled the loop here, compiler wouldn't vectorize the loop for some reason
    * (1301: Loop stride is not +1.) is the exact compiler message.
    */
    float * __restrict const matData = (float*)_aligned_malloc(matA.height * matB.rowSpan * sizeof(float), AVX_ALIGN);

    Mat matC{
        matB.width,
        matA.height,
        matB.rowSpan,
        matData
    };

    const Mat matBT = TransposeMat(matB);
    for (int rowC = 0; rowC < matA.height; ++rowC) {
        //if (rowC % 10 == 0)
        //    printf("row: %d of %d\n", rowC, matA.height);
        for (int colC = 0; colC < matB.width; ++colC) {
            float accumulate = 0;
            for (int pos=0; pos < matA.width; ++pos) {
                accumulate += matA.mat[rowC*matA.rowSpan + pos] * matBT.mat[colC*matBT.rowSpan + pos];
            }
            matData[rowC*matB.rowSpan + colC] = accumulate;
        }
    }

    _aligned_free(matBT.mat);

    return matC;
}

const Mat ST_BlockMult(const Mat& matA, const Mat& matB) {
    /*
    * Now, once we fetch column col from B, we use these cached values to populate C(row, col:col+8),
    * Any more than that, and we lose the old cached values.
    * But notice that, the C(row+1, col:col+8) uses the exact same columns.
    * So instead of traversing in row order, we could do blocks!
    * Notice that I'm using transposed B,
    * That's because MSVC refuses to vectorize the loop with non-contagious memory access.
    * So even though the floats themselves will be in the cache, we won't have SIMD, which kills the performance.
    *
    * Notice that I had to assign almost everything to temporary constants,
    * because otherwise MSVC can't guarantee the loop is not self dependent.
    */
    float * __restrict const matData = (float*)_aligned_malloc(matA.height * matB.rowSpan * sizeof(float), AVX_ALIGN);

    Mat matC{
        matB.width,
        matA.height,
        matB.rowSpan,
        matData
    };

    const unsigned blockX = 16, blockY = 16;
    
    const Mat matBT = TransposeMat(matB);
    
    int rowC = 0;
    for (; rowC < matA.height - blockY; rowC += blockY) {
        int colC = 0;
        for (; colC < matB.width - blockX; colC += blockX) {
            for (int blockRow = 0; blockRow < blockY; ++blockRow) {
                for (int blockCol = 0; blockCol < blockX; ++blockCol) {
                    const unsigned r = rowC + blockRow, c = colC + blockCol;
                    const unsigned matAoffset = r*matA.rowSpan, matBoffset = c*matBT.rowSpan;
                    float accumulate = 0;
                    for (int pos=0; pos < matA.width; ++pos) {
                        accumulate += matA.mat[matAoffset + pos] * matBT.mat[matBoffset + pos];
                    }
                    matData[r*matB.rowSpan + c] = accumulate;
                }
            }
        }
        for (int blockRow = 0; blockRow < blockY; ++blockRow) {
            for (int c = colC; c < matB.width; ++c) {
                const unsigned r = rowC + blockRow;
                const unsigned matAoffset = r * matA.rowSpan, matBoffset = c * matBT.rowSpan;
                float accumulate = 0;
                for (int pos = 0; pos < matA.width; ++pos) {
                    accumulate += matA.mat[matAoffset + pos] * matBT.mat[matBoffset + pos];
                }
                matData[r*matB.rowSpan + c] = accumulate;
            }
        }
    }
    for (; rowC < matA.height; ++rowC) {
        for (int colC = 0; colC < matB.width; ++colC) {
            const unsigned matAoffset = rowC * matA.rowSpan, matBoffset = colC * matBT.rowSpan;
            float accumulate = 0;
            for (int pos=0; pos < matA.width; ++pos) {
                accumulate += matA.mat[matAoffset + pos] * matBT.mat[matBoffset + pos];
            }
            matData[rowC*matB.rowSpan + colC] = accumulate;
        }
    }

    _aligned_free(matBT.mat);
    
    return matC;
}

/* Previous pure C++ implementation */
__declspec(noalias)
void MMHelper_MultBlocks__AutoVec(float* __restrict const matData, const unsigned blockX, const unsigned blockY,
    const unsigned rowC, const unsigned colC,
    const Mat& matA, const Mat& matB, const Mat& matBT)
{
    for (int blockRow = 0; blockRow < blockY; ++blockRow) {
        for (int blockCol = 0; blockCol < blockX; ++blockCol) {
            const unsigned r = rowC + blockRow, c = colC + blockCol;
            const unsigned matAoffset = r * matA.rowSpan, matBoffset = c * matBT.rowSpan;
            float accumulate = 0;

            // vectorized, can also be parallelized
            for (int pos = 0; pos < matA.width; ++pos) {
                accumulate += matA.mat[matAoffset + pos] * matBT.mat[matBoffset + pos];
            }

            matData[r*matB.rowSpan + c] = accumulate;
        }
    }
}

/*
* A naive loop where we load a, b multiply then sum vertically into vsum. 
* then ve horizontally sum vsum to get the final dot product.
*
* Can we do better? Of course! most of these instructions have really latecy/throughput ratio.
* we can issue multiple (non-blocking) commands at once.
*
* These are very specific to the actual hardware. 
* So I'm just using going to try tune this to my own hardware specifications.
*
* on my hardware:
* _mm256_load_ps, latency:1, tp:0.25
* _mm256_mul_ps, latency:4, tp:0.5
* _mm256_add_ps, latency:4, tp:0.5
*
* A naive loop with singular a, b loads and accumulation will yield an asm like this:
* 1: load a1, b1,
* 4: a1 <- mul(a1, b1)
* 5:  4: vsum <- add(vsum, a1),
*       1: load a2, b2
*       4: a2 <- mul(a2, b2)
* 5:  4: vsum <- add(vsum, a2),
*       1: load a3, b3
*       4: a3 <- mul(a3, b3)
* 5:  4: vsum <- add(vsum, a3),
*       1: load a4, b4
*       4: a4 <- mul(a4, b4)
* ...
* 4: vsum <- add(vsum, a4)
*
* This looks wasteful, as almost every op is dependent on the previous ones.* 
* We should be able to do better by rearranging these intrinsics, and take advantage of Intruction Level Parallelism.
*
* Below are some of the ideas I came up with. 
* For now, to test a function I simply rename it. Maybe I'll find a automated compile time soln. to this.
*
*/

/* naive loop */
__declspec(noalias)
void MMHelper_MultBlocks_Intrinsics_1(float* __restrict const matData, const unsigned blockX, const unsigned blockY,
    const unsigned rowC, const unsigned colC,
    const Mat& matA, const Mat& matB, const Mat& matBT)
{
    __declspec(align(32)) float fps[8];

    for (int blockRow = 0; blockRow < blockY; ++blockRow) {
        for (int blockCol = 0; blockCol < blockX; ++blockCol) {
            const unsigned r = rowC + blockRow, c = colC + blockCol;
            const unsigned matAoffset = r * matA.rowSpan, matBoffset = c * matBT.rowSpan;
            float accumulate = 0;

            __m256 vsum = _mm256_setzero_ps();
            __m256 a1, b1;

            /* zero padded, no edge cases */
            #pragma code_align 32
            for (int pos = 0; pos < matA.width; pos += 8) {
                /* load 8f vectors, mul, add then accumulate */
                a1 = _mm256_load_ps(&matA.mat[matAoffset + pos]);
                b1 = _mm256_load_ps(&matBT.mat[matBoffset + pos]);
                vsum = _mm256_fmadd_ps(a1, b1, vsum);
            }

            /* sum 8 floats in the __m256 */
            _mm256_store_ps(fps, vsum);
            for (int i = 0; i < 8; ++i) {
                accumulate += fps[i];
            }

            matData[r*matB.rowSpan + c] = accumulate;
        }
    }
}

/* load 8x8f from a, and 8x8f from b, all 16 registers are used, at each iteration sum of product of these are calculated.
Note that as we sum up the vectors, the pipeline gets increasingly sequential as next operations will depend on the current one.
*/
__declspec(noalias)
void MMHelper_MultBlocks_Intrinsics_2(float* __restrict const matData, const unsigned blockX, const unsigned blockY,
    const unsigned rowC, const unsigned colC,
    const Mat& matA, const Mat& matB, const Mat& matBT)
{
    __declspec(align(32)) float fps[8];

    for (int blockRow = 0; blockRow < blockY; ++blockRow) {
        for (int blockCol = 0; blockCol < blockX; ++blockCol) {
            const unsigned r = rowC + blockRow, c = colC + blockCol;
            const unsigned matAoffset = r * matA.rowSpan, matBoffset = c * matBT.rowSpan;
            float accumulate = 0;

            /* will be written on stack at every iteration as we use all registers for a,b1:8 */
            __m256 vsum = _mm256_setzero_ps();
            __m256 a1, a2, a3, a4, a5, a6, a7, a8, b1, b2, b3, b4, b5, b6, b7, b8;

            int pos = 0;
            for (; pos < matA.width - 64; pos += 8 * 8) {
                /* load 8x8f into a1:8 */
                a1 = _mm256_load_ps(&matA.mat[matAoffset + pos + 0 * 8]);
                a2 = _mm256_load_ps(&matA.mat[matAoffset + pos + 1 * 8]);
                a3 = _mm256_load_ps(&matA.mat[matAoffset + pos + 2 * 8]);
                a4 = _mm256_load_ps(&matA.mat[matAoffset + pos + 3 * 8]);
                a5 = _mm256_load_ps(&matA.mat[matAoffset + pos + 4 * 8]);
                a6 = _mm256_load_ps(&matA.mat[matAoffset + pos + 5 * 8]);
                a7 = _mm256_load_ps(&matA.mat[matAoffset + pos + 6 * 8]);
                a8 = _mm256_load_ps(&matA.mat[matAoffset + pos + 7 * 8]);

                /* load 8x8f into b1:8 */
                b1 = _mm256_load_ps(&matBT.mat[matBoffset + pos + 0 * 8]);
                b2 = _mm256_load_ps(&matBT.mat[matBoffset + pos + 1 * 8]);
                b3 = _mm256_load_ps(&matBT.mat[matBoffset + pos + 2 * 8]);
                b4 = _mm256_load_ps(&matBT.mat[matBoffset + pos + 3 * 8]);
                b5 = _mm256_load_ps(&matBT.mat[matBoffset + pos + 4 * 8]);
                b6 = _mm256_load_ps(&matBT.mat[matBoffset + pos + 5 * 8]);
                b7 = _mm256_load_ps(&matBT.mat[matBoffset + pos + 6 * 8]);
                b8 = _mm256_load_ps(&matBT.mat[matBoffset + pos + 7 * 8]);

                /* 8 independent muls */
                a1 = _mm256_mul_ps(a1, b1);
                a2 = _mm256_mul_ps(a2, b2);
                a3 = _mm256_mul_ps(a3, b3);
                a4 = _mm256_mul_ps(a4, b4);
                a5 = _mm256_mul_ps(a5, b5);
                a6 = _mm256_mul_ps(a6, b6);
                a7 = _mm256_mul_ps(a7, b7);
                a8 = _mm256_mul_ps(a8, b8);

                /* 4 independent adds */
                a1 = _mm256_add_ps(a1, a2);
                a3 = _mm256_add_ps(a3, a4);
                a5 = _mm256_add_ps(a5, a6);
                a7 = _mm256_add_ps(a7, a8);

                /* 2 independent adds */
                a1 = _mm256_add_ps(a1, a3);
                a5 = _mm256_add_ps(a5, a7);

                /* 1 add */
                a1 = _mm256_add_ps(a1, a5);

                /* 1 add, note that compiler will write/read stack as we only have 16 regs */
                vsum = _mm256_add_ps(vsum, a1);
            }
            for (; pos < matA.width; pos += 8 * 2) {
                a1 = _mm256_load_ps(&matA.mat[matAoffset + pos + 0 * 8]);
                a2 = _mm256_load_ps(&matA.mat[matAoffset + pos + 1 * 8]);

                b1 = _mm256_load_ps(&matBT.mat[matBoffset + pos + 0 * 8]);
                b2 = _mm256_load_ps(&matBT.mat[matBoffset + pos + 1 * 8]);

                a1 = _mm256_mul_ps(a1, b1);
                a2 = _mm256_mul_ps(a2, b2);

                a1 = _mm256_add_ps(a1, a2);

                vsum = _mm256_add_ps(vsum, a1);
            }

            /* sum 8 floats in the __m256 */
            _mm256_store_ps(fps, vsum);
            for (int i = 0; i < 8; ++i) {
                accumulate += fps[i];
            }

            matData[r*matB.rowSpan + c] = accumulate;
        }
    }
}

/*
* 2 4x8f vectors (a1:4, b1:4) are loaded, 
* Instead of reduce-summing to one vector, we reduce to 4 vectors, meaning that we'll have much less interdependent operations.
* On the other hand, we're only handling 4 vecs at a time and we're wasting some of the registers.
*/
__declspec(noalias)
void MMHelper_MultBlocks(float* __restrict const matData, const unsigned blockX, const unsigned blockY,
    const unsigned rowC, const unsigned colC,
    const Mat& matA, const Mat& matB, const Mat& matBT)
{
    __declspec(align(32)) float fps[8*4];
    const float* __restrict const matAmat = matA.mat;
    const float* __restrict const matBTmat = matBT.mat;
    
    /* try to prefetch next L3 block into memory while still handling this one */
    {
        if constexpr (doPrefetch) {
            std::unique_lock<std::mutex> lock(prefetchMutex);
            if (!prefetched[rowC / L3BlockY][colC / L3BlockX] && colC + L3BlockX < matBT.height && rowC + L3BlockY < matA.height) {
                for (int r = rowC; r < rowC + L3BlockY; ++r) {
                    for (int pos = 0; pos < matA.rowSpan; pos += cacheLineSz) {
                        _mm_prefetch((const char*)&matA.mat[r*matA.rowSpan + pos], _MM_HINT_T2);
                    }
                }
                for (int c = colC; c < colC + L3BlockX; ++c) {
                    for (int pos = 0; pos < matA.rowSpan; pos += cacheLineSz) {
                        _mm_prefetch((const char*)&matBT.mat[c*matBT.rowSpan + pos], _MM_HINT_T2);
                    }
                }
                prefetched[rowC / L3BlockY][colC / L3BlockX]++;
                //printf("L3 block starting from %d %d NOW FETCHING\n", rowC / L3BlockY, colC / L3BlockX);
            }
            else {
                //printf("L3 block starting from %d %d already prefetched\n",  rowC/L3BlockY, colC/L3BlockX);
            }
            if (!prefetched[rowC / L3BlockY][colC / L3BlockX + 1] && colC + 2 * L3BlockX < matBT.height) {
                for (int c = colC + L3BlockX; c < colC + L3BlockX + L3BlockX / 2; ++c) {
                    for (int pos = 0; pos < matA.rowSpan; pos += cacheLineSz) {
                        _mm_prefetch((const char*)&matBT.mat[c*matBT.rowSpan + pos], _MM_HINT_T2);
                    }
                }
                prefetched[rowC / L3BlockY][colC / L3BlockX + 1]++;
            }
        }
    }

    for (int blockRowC = rowC; blockRowC < rowC+L3BlockY; blockRowC += L2BlockY) {
        for (int blockColC = colC; blockColC < colC+(L3BlockX>>1); blockColC += L2BlockX) {
            for (int blockRow = 0; blockRow < L2BlockY; ++blockRow) {
                for (int blockCol = 0; blockCol < L2BlockX; ++blockCol) {
                    const unsigned r = blockRowC + blockRow, c = blockColC + blockCol;
                    const unsigned matAoffset = r * matA.rowSpan, matBoffset = c * matBT.rowSpan;
                    float accumulate = 0;

                    __m256 a1, a2, a3, a4, b1, b2, b3, b4;
                    __m256 c1 = _mm256_setzero_ps();
                    __m256 c2 = _mm256_setzero_ps();
                    __m256 c3 = _mm256_setzero_ps();
                    __m256 c4 = _mm256_setzero_ps();

                    /* matrices are only guaranteed to have 16f aligned rowspans. */
                    int pos = 0;
                    for (; pos < matA.width - 32; pos += 8 * 4) {
                        /* load 4x8f */
                        a1 = _mm256_load_ps(&matAmat[matAoffset + pos + 0 * 8]);
                        a2 = _mm256_load_ps(&matAmat[matAoffset + pos + 1 * 8]);
                        a3 = _mm256_load_ps(&matAmat[matAoffset + pos + 2 * 8]);
                        a4 = _mm256_load_ps(&matAmat[matAoffset + pos + 3 * 8]);

                        /* load 4x8f */
                        b1 = _mm256_load_ps(&matBTmat[matBoffset + pos + 0 * 8]);
                        b2 = _mm256_load_ps(&matBTmat[matBoffset + pos + 1 * 8]);
                        b3 = _mm256_load_ps(&matBTmat[matBoffset + pos + 2 * 8]);
                        b4 = _mm256_load_ps(&matBTmat[matBoffset + pos + 3 * 8]);

                        c1 = _mm256_fmadd_ps(a1, b1, c1);
                        c2 = _mm256_fmadd_ps(a2, b2, c2);
                        c3 = _mm256_fmadd_ps(a3, b3, c3);
                        c4 = _mm256_fmadd_ps(a4, b4, c4);
                    }
                    /* handle the remaining */
                    for (; pos < matA.width; pos += 8*2) {
                        a1 = _mm256_load_ps(&matAmat[matAoffset + pos + 0 * 8]);
                        a2 = _mm256_load_ps(&matAmat[matAoffset + pos + 1 * 8]);

                        b1 = _mm256_load_ps(&matBTmat[matBoffset + pos + 0 * 8]);
                        b2 = _mm256_load_ps(&matBTmat[matBoffset + pos + 1 * 8]);

                        c1 = _mm256_fmadd_ps(a1, b1, c1);
                        c2 = _mm256_fmadd_ps(a2, b2, c2);
                    }

                    /* horizontal sum */
                    _mm256_store_ps(&fps[0], c1);
                    _mm256_store_ps(&fps[8], c2);
                    _mm256_store_ps(&fps[16], c3);
                    _mm256_store_ps(&fps[24], c4);
                    for (int i = 0; i < 8*4; ++i) {
                        accumulate += fps[i];
                    }

                    matData[r*matB.rowSpan + c] = accumulate;
                }
            }
        }
    }
}

/*
* This method tries to solve the problems of the last one, by handling 6x8f vecs at a time.
* This way, we're keeping cycles per iteration low while increasing throughput.
*/
__declspec(noalias)
void MMHelper_MultBlocks_Intrinsics_4(float* __restrict const matData, const unsigned blockX, const unsigned blockY,
    const unsigned rowC, const unsigned colC,
    const Mat& matA, const Mat& matB, const Mat& matBT)
{
    __declspec(align(32)) float fps[8 * 3];

    for (int blockRow = 0; blockRow < blockY; ++blockRow) {
        for (int blockCol = 0; blockCol < blockX; ++blockCol) {
            const unsigned r = rowC + blockRow, c = colC + blockCol;
            const unsigned matAoffset = r * matA.rowSpan, matBoffset = c * matBT.rowSpan;
            float accumulate = 0;

            __m256 a1, a2, a3, a4, a5, a6, b1, b2, b3, b4, b5, b6;
            __m256 c1 = _mm256_setzero_ps();
            __m256 c2 = _mm256_setzero_ps();
            __m256 c3 = _mm256_setzero_ps();

            int pos = 0;
            for (; pos < matA.width-48; pos += 8 * 6) {
                a1 = _mm256_load_ps(&matA.mat[matAoffset + pos + 0 * 8]);
                a2 = _mm256_load_ps(&matA.mat[matAoffset + pos + 1 * 8]);
                a3 = _mm256_load_ps(&matA.mat[matAoffset + pos + 2 * 8]);
                a4 = _mm256_load_ps(&matA.mat[matAoffset + pos + 3 * 8]);
                a5 = _mm256_load_ps(&matA.mat[matAoffset + pos + 4 * 8]);
                a6 = _mm256_load_ps(&matA.mat[matAoffset + pos + 5 * 8]);

                b1 = _mm256_load_ps(&matBT.mat[matBoffset + pos + 0 * 8]);
                b2 = _mm256_load_ps(&matBT.mat[matBoffset + pos + 1 * 8]);
                b3 = _mm256_load_ps(&matBT.mat[matBoffset + pos + 2 * 8]);
                b4 = _mm256_load_ps(&matBT.mat[matBoffset + pos + 3 * 8]);
                b5 = _mm256_load_ps(&matBT.mat[matBoffset + pos + 4 * 8]);
                b6 = _mm256_load_ps(&matBT.mat[matBoffset + pos + 5 * 8]);

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

            // zero padded, no edge cases
            for (; pos < matA.width; pos += 8 * 2) {
                a1 = _mm256_load_ps(&matA.mat[matAoffset + pos + 0 * 8]);
                a2 = _mm256_load_ps(&matA.mat[matAoffset + pos + 1 * 8]);

                b1 = _mm256_load_ps(&matBT.mat[matBoffset + pos + 0 * 8]);
                b2 = _mm256_load_ps(&matBT.mat[matBoffset + pos + 1 * 8]);

                a1 = _mm256_mul_ps(a1, b1);
                a2 = _mm256_mul_ps(a2, b2);

                c1 = _mm256_add_ps(c1, a1);
                c2 = _mm256_add_ps(c2, a2);
            }

            /* sum 8 floats in the __m256, I'm sure I can write this using intrinsics as well */
            _mm256_store_ps(&fps[0], c1);
            _mm256_store_ps(&fps[8], c2);
            _mm256_store_ps(&fps[16], c3);
            for (int i = 0; i < 8 * 3; ++i) {
                accumulate += fps[i];
            }

            matData[r*matB.rowSpan + c] = accumulate;
        }
    }
}

/* Utilize FMA instructions, vfmaddps (4 cycles latency, 0.5 CPI, just like vaddps) */
__declspec(noalias)
void MMHelper_MultBlocks_Intrinsics_5(float* __restrict const matData, const unsigned blockX, const unsigned blockY,
    const unsigned rowC, const unsigned colC,
    const Mat& matA, const Mat& matB, const Mat& matBT)
{
    __declspec(align(32)) float fps[8 * 5];

    for (int blockRow = 0; blockRow < blockY; ++blockRow) {
        for (int blockCol = 0; blockCol < blockX; ++blockCol) {
            const unsigned r = rowC + blockRow, c = colC + blockCol;
            const unsigned matAoffset = r * matA.rowSpan, matBoffset = c * matBT.rowSpan;
            float accumulate = 0;

            __m256 a1, a2, a3, a4, a5, b1, b2, b3, b4, b5;
            __m256 c1 = _mm256_setzero_ps();
            __m256 c2 = _mm256_setzero_ps();
            __m256 c3 = _mm256_setzero_ps();
            __m256 c4 = _mm256_setzero_ps();
            __m256 c5 = _mm256_setzero_ps();

            int pos = 0;
            for (; pos < matA.width - 40; pos += 8 * 5) {
                a1 = _mm256_load_ps(&matA.mat[matAoffset + pos + 0 * 8]);
                a2 = _mm256_load_ps(&matA.mat[matAoffset + pos + 1 * 8]);
                a3 = _mm256_load_ps(&matA.mat[matAoffset + pos + 2 * 8]);
                a4 = _mm256_load_ps(&matA.mat[matAoffset + pos + 3 * 8]);
                a5 = _mm256_load_ps(&matA.mat[matAoffset + pos + 4 * 8]);

                b1 = _mm256_load_ps(&matBT.mat[matBoffset + pos + 0 * 8]);
                b2 = _mm256_load_ps(&matBT.mat[matBoffset + pos + 1 * 8]);
                b3 = _mm256_load_ps(&matBT.mat[matBoffset + pos + 2 * 8]);
                b4 = _mm256_load_ps(&matBT.mat[matBoffset + pos + 3 * 8]);
                b5 = _mm256_load_ps(&matBT.mat[matBoffset + pos + 4 * 8]);

                c1 = _mm256_fmadd_ps(a1, b1, c1);
                c2 = _mm256_fmadd_ps(a2, b2, c2);
                c3 = _mm256_fmadd_ps(a3, b3, c3);
                c4 = _mm256_fmadd_ps(a4, b4, c4);
                c5 = _mm256_fmadd_ps(a5, b5, c5);
            }

            // handle edge
            for (; pos < matA.width; pos += 8) {
                a1 = _mm256_load_ps(&matA.mat[matAoffset + pos + 0 * 8]);
                b1 = _mm256_load_ps(&matBT.mat[matBoffset + pos + 0 * 8]);
                c1 = _mm256_fmadd_ps(a1, b1, c1);
            }

            /* sum 8 floats in the __m256, I'm sure I can write this using intrinsics as well */
            
            c1 = _mm256_add_ps(c1, c2);
            c3 = _mm256_add_ps(c3, c4);
            c1 = _mm256_add_ps(c1, c3);
            c1 = _mm256_add_ps(c1, c5);
            
            _mm256_store_ps(&fps[0], c1);
            for (int i = 0; i < 8; ++i) {
                accumulate += fps[i];
            }

            matData[r*matB.rowSpan + c] = accumulate;
        }
    }
}

/*
* You will notice some of the loops are manually unrolled and some are kept as one liners
* To ensure use of packed floating operations (SSE/AVX), we need to write the write loops in very specific ways,
* While in some cases, manually unrolling the loop helps (usually when compiler can't do it itself), 
* in other cases it actually prevents compiler from creating vectorized assembly.
* So, this is pretty compiler dependent and honestly I'm not a fan.
* I'd just use compiler intrinsics if I could, but I guess they'd count as ASM.
*/
__declspec(noalias)
const Mat MTMatMul(const Mat& matA, const Mat& matB) {
    /*
    * Now, let's parallelize the code!
    * We're already doing block by block multiplication, which are independent of each other.
    * I'm on an i7 system with HT, so I will assign 2 threads to process a block so they operate on same cache.
    *
    * I will use a thread-couple pool to process blocks.
    * Read HWLocalThreadPool.h for more details.
    */
    
    float * __restrict const matData = (float*)_aligned_malloc(matA.height * matB.rowSpan * sizeof(float), AVX_ALIGN);
    
    memset(&prefetched[0][0], 0, 1024*1024*sizeof(int));

    Mat matC{
        matB.width,
        matA.height,
        matB.rowSpan,
        matData
    };

    /*
    * When block size is small enough to fit L2 cache, we have a lot of L3 misses, DRAM fetches. 
    * When the block size is large, the block data won't fit into L2 cache. Lose-Lose.
    * So, we should apply the blocking idea at one more level, for L3 cahce
    * We'll issue large blocks which will fit into L3 cache, and then  we'll do smaller, L2 sized blocks on them.
    */

    const Mat matBT = TransposeMat(matB);

    /* Using hardware local threads reduces 5Kx5K MM from 2.45s to 2.15s 12% improvement! */
    HWLocalThreadPool<6, 2> tp;

    //for (int r = 0; r < L3BlockY; ++r) {
    //    for (int pos = 0; pos < matA.rowSpan; pos += cacheLineSz) {
    //        _mm_prefetch((const char*)&matA.mat[r*matA.rowSpan + pos], _MM_HINT_T2);
    //    }
    //}
    //for (int c = 0; c < L3BlockX; ++c) {
    //    for (int pos = 0; pos < matA.rowSpan; pos += cacheLineSz) {
    //        _mm_prefetch((const char*)&matBT.mat[c*matBT.rowSpan + pos], _MM_HINT_T2);
    //    }
    //}

    //prefetched[0][0]++;

    //int largeBlockRowC = 0;
    //for (; largeBlockRowC <= matA.height - L3BlockY; largeBlockRowC += L3BlockY) {
    //    int largeBlockColC = 0;
    //    for (; largeBlockColC <= matB.width - L3BlockX; largeBlockColC += L3BlockX) {
    //        for (int blockRowC = 0; blockRowC < L3BlockY; blockRowC += L2BlockY) {
    //            for (int blockColC = 0; blockColC < L3BlockX; blockColC += 2*L2BlockX) {
    //                tp.Add({
    //                    HWLocalThreadPool<>::WrapFunc(MMHelper_MultBlocks,
    //                        matData, 0, 0, largeBlockRowC + blockRowC,
    //                            largeBlockColC + blockColC, matA, matB, matBT),
    //                    HWLocalThreadPool<>::WrapFunc(MMHelper_MultBlocks,
    //                        matData, 0, 0, largeBlockRowC + blockRowC,
    //                            largeBlockColC+ blockColC + L2BlockX, matA, matB, matBT)
    //                    });
    //            }
    //        }
    //    }
    //}

    int largeBlockRowC = 0;
    for (; largeBlockRowC <= matA.height - L3BlockY; largeBlockRowC += L3BlockY) {
        /* Process BlockX X BlockY blocks */
        int largeBlockColC = 0;
        for (; largeBlockColC <= matB.width - L3BlockX; largeBlockColC += L3BlockX) {
            tp.Add({
                HWLocalThreadPool<>::WrapFunc(MMHelper_MultBlocks,
                    matData, 0, 0, largeBlockRowC,
                        largeBlockColC, matA, matB, matBT),
                HWLocalThreadPool<>::WrapFunc(MMHelper_MultBlocks,
                    matData, 0, 0, largeBlockRowC,
                        largeBlockColC + (L3BlockX >> 1), matA, matB, matBT)
            });
        }
    }

    //const unsigned blockX = 8, blockY = 128;
    //const unsigned subX = blockX >> 1;

    //int rowC = 0;
    //for (; rowC < matA.height - blockY; rowC += blockY) {
    //    int colC = 0;
    //    /* Process BlockX X BlockY blocks */
    //    for (; colC < matB.width - blockX; colC += blockX) {
    //        tp.Add({
    //            HWLocalThreadPool<>::WrapFunc(MMHelper_MultBlocks,
    //            matData, subX, blockY, rowC, colC, matA, matB, matBT),
    //            HWLocalThreadPool<>::WrapFunc(MMHelper_MultBlocks,
    //            matData, subX, blockY, rowC, colC + subX, matA, matB, matBT)
    //            });
    //    }
    //    /* Process remainings at the end of the row, width < blockX */
    //    tp.Add({
    //        HWLocalThreadPool<>::WrapFunc(MMHelper_MultBlocks,
    //        matData, matB.width - colC, blockY, rowC, colC, matA, matB, matBT),
    //        []() {}
    //        });
    //}

    ///* Process last row, height < blockY, col+=blockX */
    //int colC = 0;
    //for (; colC < matB.width - blockX; colC += blockX) {
    //    tp.Add({
    //        HWLocalThreadPool<>::WrapFunc(MMHelper_MultBlocks,
    //        matData, subX, matA.height - rowC, rowC, colC, matA, matB, matBT) ,
    //        HWLocalThreadPool<>::WrapFunc(MMHelper_MultBlocks,
    //        matData, subX, matA.height - rowC, rowC, colC + subX, matA, matB, matBT)
    //        });
    //}

    ///* Process bottom right block, h < bY, w < bX */
    //tp.Add({
    //    HWLocalThreadPool<>::WrapFunc(MMHelper_MultBlocks,
    //    matData, matB.width - colC, matA.height - rowC, rowC, colC, matA, matB, matBT),
    //    []() {}
    //    });

    tp.Close();
    _aligned_free(matBT.mat);

    return matC;
}

/* 
* EDIT: --NEEDS MORE TUNING--
* A very, very simple toggle between two functions depending on the total number of ops 
*/
const Mat MatMul(const Mat& matA, const Mat& matB) {
    /* A:  a,b B: b,c => # of op: a*b*b*c */
    //if (matA.height*matA.width*matA.width*matB.width < 125000000) {
    //    return ST_TransposedBMatMul(matA, matB);
    //}
    return MTMatMul(matA, matB);
}

int __cdecl main(int argc, char *argv[])
{
    //if (argc < 4)
    //{
    //std::cout << "No args\n";
    //return 0;
    //}

    //const char * inputMtxAFile = argv[1];
    //const char * inputMtxBFile = argv[2];
    //const char * outMtxABFile = argv[3];

    const char * inputMtxAFile = "matrixA.bin";
    const char * inputMtxBFile = "matrixB.bin";
    const char * outMtxABFile = "matrixAB-out.bin";

    const Mat inputMtxA = LoadMat(inputMtxAFile);
    const Mat inputMtxB = LoadMat(inputMtxBFile);

    auto start = std::chrono::high_resolution_clock::now();

    const Mat outMtxAB = MTMatMul(inputMtxA, inputMtxB);

    auto end = std::chrono::high_resolution_clock::now();

    std::cout << "Matrix Multiplication: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " microseconds.\n";
    //std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    DumpMat(outMtxABFile, outMtxAB);

    return 0;
}
