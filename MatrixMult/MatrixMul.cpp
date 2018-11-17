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

/* Define for AVX alignment requirements */
#define AVX_ALIGN 32

constexpr int doPrefetch = 1;
int prefetched[1024][1024];
std::mutex prefetchMutex;
constexpr unsigned cacheLineSz = 64;

typedef struct Mat {
    unsigned width;
    unsigned height;
    unsigned rowSpan;
    /* guarantee that mat will not be aliased (__restrict), 
    no need for two matrices to point at sama data */
    float* __restrict mat;
} Mat;

/* This struct holds the information for multiple levels of block sizes.
 * Constraints on block sizes:
 * L2BlockX % 3 == L2BlockY % 3 == 0,
 * L3BlockX % 2 == L3BlockY % 2 == 0,
 * (L3BlockX / 2) % L2BlockX == 0 */
typedef struct MMBlockInfo {
    const unsigned L3BlockX = 60, L3BlockY = 60;
    const unsigned L2BlockX = 6, L2BlockY = 6;
} MMBlockInfo;

/* This function loads a previously saved matrix from disk */
const Mat LoadMat(const char* const filename)
{
    Mat mat;
    uint32_t matSize;

    std::ifstream in(filename, std::ios::binary | std::ios::in);

    if (!in.is_open()) {
        std::cout << "Err loading!\n";
        in.close();
        return {0, 0, 0, NULL};
    }

    in.read((char*)&mat, 3 * sizeof(uint32_t));
    in.read((char*)&matSize, sizeof(uint32_t));
    in.seekg(12 * sizeof(uint32_t), std::ios::cur);
    mat.mat = (float*)_aligned_malloc(matSize, AVX_ALIGN);
    in.read((char*)mat.mat, matSize);

    in.close();

    return mat;
}

/* This function dumps the given matrix to the disk. */
static void DumpMat(const char* filename, const Mat& m)
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

/* This function rounds a given number to the nearest multiple of K, 
 * where K is a parameter and is a power of 2 */
static unsigned RoundUpPwr2(unsigned val, unsigned pwr2)
{
    return (val + (pwr2 - 1)) & (~(pwr2 - 1));
}

/* This function computes the transpose of a given matrix. 
 * It currently uses a singlethreaded implementation. */
__declspec(noalias) const Mat TransposeMat(const Mat& mat)
{
    const unsigned tRowSpan = RoundUpPwr2(mat.height, 64 / sizeof(float));
    float* __restrict const tData =
      (float*)_aligned_malloc(mat.width * tRowSpan * sizeof(float), AVX_ALIGN);

    Mat T{mat.height, mat.width, tRowSpan, tData};

    // hah, the loops are truly interchangable as we encounter a cache miss either ways
    for (int rowT = 0; rowT < T.height; ++rowT) {
        for (int colT = 0; colT < T.width; ++colT) {
            tData[rowT * tRowSpan + colT] = mat.mat[colT * mat.rowSpan + rowT];
        }
    }

    return T;
}

/* This function prints the given matrix to given std::ostream */
static void PrintMat(const Mat& mat, std::ostream& stream)
{
    stream << "w, h, rS: " << mat.width << " " << mat.height << "  " << mat.rowSpan
           << "\n";
    for (int i = 0; i < mat.height; i++) {
        for (int j = 0; j < mat.width; ++j) {
            stream << mat.mat[i * mat.rowSpan + j] << " ";
        }
        stream << "\n";
    }
}

const Mat ST_NaiveMatMul(const Mat& matA, const Mat& matB)
{
    /* First : naive solution with but with some tricks to make compiler (MVC) behave
     * Note that, in this case, manually unrolling the loop helps 
     * as the compiler can't auto-vectorize non-contagious memory access */
    float* __restrict const matData =
      (float*)_aligned_malloc(matA.height * matB.rowSpan * sizeof(float), AVX_ALIGN);

    Mat matC{matB.width, matA.height, matB.rowSpan, matData};

    for (int rowC = 0; rowC < matA.height; ++rowC) {
        for (int colC = 0; colC < matB.width; ++colC) {
            // have a local accumulator, o.w compiler fetches the value at each += operator.
            float accumulate = 0;
            int pos = 0;
            // interestingly, manual unrolling IS helpful, it takes 1000x1000 multiplication from about 990ms to 710ms
            for (; pos < matA.width - 4; pos += 4) {
                accumulate += matA.mat[rowC * matA.rowSpan + pos] *
                                matB.mat[pos * matB.rowSpan + colC] +
                              matA.mat[rowC * matA.rowSpan + pos + 1] *
                                matB.mat[(pos + 1) * matB.rowSpan + colC] +
                              matA.mat[rowC * matA.rowSpan + pos + 2] *
                                matB.mat[(pos + 2) * matB.rowSpan + colC] +
                              matA.mat[rowC * matA.rowSpan + pos + 3] *
                                matB.mat[(pos + 3) * matB.rowSpan + colC];
            }
            for (; pos < matA.width; ++pos) {
                accumulate += matA.mat[rowC * matA.rowSpan + pos] *
                              matB.mat[pos * matB.rowSpan + colC];
            }
            matData[rowC * matB.rowSpan + colC] = accumulate;
        }
    }

    return matC;
}

const Mat ST_TransposedBMatMul(const Mat& matA, const Mat& matB)
{
    /* Now, I thought transposing B and then traversing it row order would help and it does!
     * Also, note that, if we manually unrolled the loop here, compiler wouldn't vectorize the loop for some reason
     * (1301: Loop stride is not +1.) is the exact compiler message. */
    float* __restrict const matData =
      (float*)_aligned_malloc(matA.height * matB.rowSpan * sizeof(float), AVX_ALIGN);

    Mat matC{matB.width, matA.height, matB.rowSpan, matData};

    const Mat matBT = TransposeMat(matB);
    for (int rowC = 0; rowC < matA.height; ++rowC) {
        for (int colC = 0; colC < matB.width; ++colC) {
            float accumulate = 0;
            for (int pos = 0; pos < matA.width; ++pos) {
                accumulate += matA.mat[rowC * matA.rowSpan + pos] *
                              matBT.mat[colC * matBT.rowSpan + pos];
            }
            matData[rowC * matB.rowSpan + colC] = accumulate;
        }
    }

    _aligned_free(matBT.mat);

    return matC;
}

const Mat ST_BlockMult(const Mat& matA, const Mat& matB)
{
    /* Now, once we fetch column col from B, we use these cached values 
     * to populate C(row, col:col+8), Any more than that, 
     * and we lose the old cached values. But notice that, 
     * the C(row+1, col:col+8) uses the exact same columns.
     * So instead of traversing in row order, we could do blocks!
     * Notice that I'm using transposed B,
     * That's because MSVC refuses to vectorize the loop with 
     * non-contagious memory access.
     * So even though the floats themselves will be in the cache, 
     * we won't have SIMD, which kills the performance.
     *
     * Also, I had to assign offsets to temporary constants,
     * because otherwise MSVC can't auto-vectorize. */
    float* __restrict const matData =
      (float*)_aligned_malloc(matA.height * matB.rowSpan * sizeof(float), AVX_ALIGN);

    Mat matC{matB.width, matA.height, matB.rowSpan, matData};

    const unsigned blockX = 16, blockY = 16;

    const Mat matBT = TransposeMat(matB);

    int rowC = 0;
    for (; rowC < matA.height - blockY; rowC += blockY) {
        int colC = 0;
        for (; colC < matB.width - blockX; colC += blockX) {
            for (int blockRow = 0; blockRow < blockY; ++blockRow) {
                for (int blockCol = 0; blockCol < blockX; ++blockCol) {
                    const unsigned r = rowC + blockRow;
                    const unsigned c = colC + blockCol;
                    const unsigned matAoffset = r * matA.rowSpan;
                    const unsigned matBoffset = c * matBT.rowSpan;

                    float accumulate = 0;
                    for (int pos = 0; pos < matA.width; ++pos) {
                        accumulate +=
                          matA.mat[matAoffset + pos] * matBT.mat[matBoffset + pos];
                    }
                    matData[r * matB.rowSpan + c] = accumulate;
                }
            }
        }
        for (int blockRow = 0; blockRow < blockY; ++blockRow) {
            for (int c = colC; c < matB.width; ++c) {
                const unsigned r = rowC + blockRow;
                const unsigned matAoffset = r * matA.rowSpan;
                const unsigned matBoffset = c * matBT.rowSpan;
                float accumulate = 0;
                for (int pos = 0; pos < matA.width; ++pos) {
                    accumulate +=
                      matA.mat[matAoffset + pos] * matBT.mat[matBoffset + pos];
                }
                matData[r * matB.rowSpan + c] = accumulate;
            }
        }
    }
    for (; rowC < matA.height; ++rowC) {
        for (int colC = 0; colC < matB.width; ++colC) {
            const unsigned matAoffset = rowC * matA.rowSpan;
            const unsigned matBoffset = colC * matBT.rowSpan;
            float accumulate = 0;
            for (int pos = 0; pos < matA.width; ++pos) {
                accumulate += matA.mat[matAoffset + pos] * matBT.mat[matBoffset + pos];
            }
            matData[rowC * matB.rowSpan + colC] = accumulate;
        }
    }

    _aligned_free(matBT.mat);

    return matC;
}

/* This function handles blocks smaller than full sized L3X x L3Y blocks.
* It is placed farther from the actual hot code. For the full sized blocks,
* go to MMHelper_MultFullBlocks. The block handled here is defined as
* next blockX cols and blockY rows from top left:
* (rowC, colC) to (rowC+blockY-1, colC+blockX-1)
* blockX and blockY need to be smaller than L3BlockX and L3BlockY respectively.
* This requirement however, is not asserted. */
__declspec(noalias) void MMHelper_MultRemBlocks(
  float* __restrict const matData, const unsigned rowSpan, const Mat& matA,
  const Mat& matBT, const unsigned colC, const unsigned rowC, const int blockX,
  const int blockY, const MMBlockInfo& mmBlockInfo)
{
    /* if no work to be done, exit */
    if (blockX <= 0 || blockY <= 0)
        return;

    /* Allocate an aligned array on stack for doing horizontal vector sum.
    * This step is only performed once and compiler already optimizes it.
    * So leaving it like this is better for readibility at no cost. */
    __declspec(align(32)) float fps[8 * 10];
    /* similar setup */
    float accumulate[9];

    /* shorthand for some parameters */
    const unsigned L2BlockX = mmBlockInfo.L2BlockX, L2BlockY = mmBlockInfo.L2BlockY,
                   L3BlockX = mmBlockInfo.L3BlockX, L3BlockY = mmBlockInfo.L3BlockY;

    int blockRowC = rowC;
    /* handle full L2Y sized rows */
    for (; blockRowC <= rowC + blockY - L2BlockY; blockRowC += L2BlockY) {
        int blockColC = colC;
        /* handle (L2X x L2Y) blocks */
        for (; blockColC <= colC + blockX - L2BlockX; blockColC += L2BlockX) {
            for (int blockRow = 0; blockRow < L2BlockY; blockRow += 3) {
                const unsigned matAoffset1 = (blockRowC + blockRow + 0) * matA.rowSpan,
                               matAoffset2 = (blockRowC + blockRow + 1) * matA.rowSpan,
                               matAoffset3 = (blockRowC + blockRow + 2) * matA.rowSpan;
                for (int blockCol = 0; blockCol < L2BlockX; blockCol += 3) {
                    /* note that we're handling 3 rows at a time, assuming L2BlockX = 3 */
                    const unsigned matBToffset1 =
                                     (blockColC + blockCol + 0) * matBT.rowSpan,
                                   matBToffset2 =
                                     (blockColC + blockCol + 1) * matBT.rowSpan,
                                   matBToffset3 =
                                     (blockColC + blockCol + 2) * matBT.rowSpan;

                    /* set up accumulator SIMD variables */
                    __m256 a1, a2, a3, b1, b2, b3;
                    __m256 c1 = _mm256_setzero_ps();
                    __m256 c2 = _mm256_setzero_ps();
                    __m256 c3 = _mm256_setzero_ps();
                    __m256 c4 = _mm256_setzero_ps();
                    __m256 c5 = _mm256_setzero_ps();
                    __m256 c6 = _mm256_setzero_ps();
                    __m256 c7 = _mm256_setzero_ps();
                    __m256 c8 = _mm256_setzero_ps();
                    __m256 c9 = _mm256_setzero_ps();

                    /* 0.75 arithmetic intensity, 6 loads (3 a, 3 b) -> 9 fma instructions. */
                    for (int pos = 0; pos < matA.width; pos += 8) {
                        /* 6 8f vector loads */
                        a1 = _mm256_load_ps(&matA.mat[matAoffset1 + pos]);
                        a2 = _mm256_load_ps(&matA.mat[matAoffset2 + pos]);
                        a3 = _mm256_load_ps(&matA.mat[matAoffset3 + pos]);

                        b1 = _mm256_load_ps(&matBT.mat[matBToffset1 + pos]);
                        b2 = _mm256_load_ps(&matBT.mat[matBToffset2 + pos]);
                        b3 = _mm256_load_ps(&matBT.mat[matBToffset3 + pos]);

                        /* 9 fma instructions */
                        c1 = _mm256_fmadd_ps(a1, b1, c1);
                        c2 = _mm256_fmadd_ps(a1, b2, c2);
                        c3 = _mm256_fmadd_ps(a1, b3, c3);

                        c4 = _mm256_fmadd_ps(a2, b1, c4);
                        c5 = _mm256_fmadd_ps(a2, b2, c5);
                        c6 = _mm256_fmadd_ps(a2, b3, c6);

                        c7 = _mm256_fmadd_ps(a3, b1, c7);
                        c8 = _mm256_fmadd_ps(a3, b2, c8);
                        c9 = _mm256_fmadd_ps(a3, b3, c9);
                    }

                    /* horizontal sum */

                    memset(&accumulate[0], 0, 9 * sizeof(float));

                    _mm256_store_ps(&fps[0], c1);
                    _mm256_store_ps(&fps[8], c2);
                    _mm256_store_ps(&fps[16], c3);
                    _mm256_store_ps(&fps[24], c4);
                    _mm256_store_ps(&fps[32], c5);
                    _mm256_store_ps(&fps[40], c6);
                    _mm256_store_ps(&fps[48], c7);
                    _mm256_store_ps(&fps[56], c8);
                    _mm256_store_ps(&fps[64], c9);

                    for (int i = 0; i < 9; ++i) {
                        for (int j = 0; j < 8; ++j) {
                            accumulate[i] += fps[i * 8 + j];
                        }
                    }

                    /* stores into matData */
                    matData[(blockRowC + blockRow + 0) * rowSpan + blockColC +
                            blockCol + 0] = accumulate[0];
                    matData[(blockRowC + blockRow + 0) * rowSpan + blockColC +
                            blockCol + 1] = accumulate[1];
                    matData[(blockRowC + blockRow + 0) * rowSpan + blockColC +
                            blockCol + 2] = accumulate[2];

                    matData[(blockRowC + blockRow + 1) * rowSpan + blockColC +
                            blockCol + 0] = accumulate[3];
                    matData[(blockRowC + blockRow + 1) * rowSpan + blockColC +
                            blockCol + 1] = accumulate[4];
                    matData[(blockRowC + blockRow + 1) * rowSpan + blockColC +
                            blockCol + 2] = accumulate[5];

                    matData[(blockRowC + blockRow + 2) * rowSpan + blockColC +
                            blockCol + 0] = accumulate[6];
                    matData[(blockRowC + blockRow + 2) * rowSpan + blockColC +
                            blockCol + 1] = accumulate[7];
                    matData[(blockRowC + blockRow + 2) * rowSpan + blockColC +
                            blockCol + 2] = accumulate[8];
                }
            }
        }
        /* handle the remaining columns, (w<L2X, h=L2Y) */
        for (int blockRow = 0; blockRow < L2BlockY; blockRow += 3) {
            const unsigned matAoffset1 = (blockRowC + blockRow + 0) * matA.rowSpan,
                           matAoffset2 = (blockRowC + blockRow + 1) * matA.rowSpan,
                           matAoffset3 = (blockRowC + blockRow + 2) * matA.rowSpan;
            for (int blockCol = blockColC; blockCol < colC + blockX; ++blockCol) {
                const unsigned matBToffset = blockCol * matBT.rowSpan;

                /* set up accumulators */
                __m256 a11, a12, a21, a22, a31, a32, b1, b2;
                __m256 c1 = _mm256_setzero_ps();
                __m256 c2 = _mm256_setzero_ps();
                __m256 c3 = _mm256_setzero_ps();
                __m256 c4 = _mm256_setzero_ps();
                __m256 c5 = _mm256_setzero_ps();
                __m256 c6 = _mm256_setzero_ps();

                /* note that b1:2 and aij, aik are consequent 8f vecs,
                 * extracted from the same row, not 8f vecs from parallel rows, 
                 * like aij, akj are. Since we can't have multiple b rows,
                 * this is how we can best utilize the registers available.
                 * Also rowSpan is guaranteed to be a multiple of 16f,
                 * so we can handle 2x2x8f at a time
                 *
                 * [---- [a11] [a12] ---- ]
                 * [---- [a21] [a22] ---- ]
                 * [---- [a31] [a32] ---- ]
                 * [---- [b1 ] [ b2] ---- ]
                 */

                for (int pos = 0; pos < matA.width; pos += 16) {
                    a11 = _mm256_load_ps(&matA.mat[matAoffset1 + pos]);
                    a12 = _mm256_load_ps(&matA.mat[matAoffset1 + pos + 8]);

                    a21 = _mm256_load_ps(&matA.mat[matAoffset2 + pos]);
                    a22 = _mm256_load_ps(&matA.mat[matAoffset2 + pos + 8]);

                    a31 = _mm256_load_ps(&matA.mat[matAoffset3 + pos]);
                    a32 = _mm256_load_ps(&matA.mat[matAoffset3 + pos + 8]);

                    b1 = _mm256_load_ps(&matBT.mat[matBToffset + pos]);
                    b2 = _mm256_load_ps(&matBT.mat[matBToffset + pos + 8]);

                    c1 = _mm256_fmadd_ps(a11, b1, c1);
                    c2 = _mm256_fmadd_ps(a21, b1, c2);
                    c3 = _mm256_fmadd_ps(a31, b1, c3);

                    c4 = _mm256_fmadd_ps(a12, b2, c4);
                    c5 = _mm256_fmadd_ps(a22, b2, c5);
                    c6 = _mm256_fmadd_ps(a32, b2, c6);
                }

                /* horizontal sum */

                memset(&accumulate[0], 0, 3 * sizeof(float));

                c1 = _mm256_add_ps(c1, c4);
                c2 = _mm256_add_ps(c2, c5);
                c3 = _mm256_add_ps(c3, c6);

                _mm256_store_ps(&fps[0], c1);
                _mm256_store_ps(&fps[8], c2);
                _mm256_store_ps(&fps[16], c3);

                for (int i = 0; i < 3; ++i) {
                    for (int j = 0; j < 8; ++j) {
                        accumulate[i] += fps[i * 8 + j];
                    }
                }

                /* stores */
                matData[(blockRowC + blockRow + 0) * rowSpan + blockCol] =
                  accumulate[0];
                matData[(blockRowC + blockRow + 1) * rowSpan + blockCol] =
                  accumulate[1];
                matData[(blockRowC + blockRow + 2) * rowSpan + blockCol] =
                  accumulate[2];
            }
        }
    }
    /* handle the last row, h<L2Y 
     * we should still handle rows as groups of 3.
     * for small matrices, L2 block can be quite a bit larger than 3x3,
     * don't fallback to low arithmetic intensity computing if that's the case */
    for (; blockRowC <= rowC + blockY - 3; blockRowC += 3) {
        const unsigned matAoffset1 = (blockRowC + 0) * matA.rowSpan,
                       matAoffset2 = (blockRowC + 1) * matA.rowSpan,
                       matAoffset3 = (blockRowC + 2) * matA.rowSpan;
        int blockColC = colC;
        /* handle (L2X x h<L2Y) blocks */
        for (; blockColC <= colC + blockX - L2BlockX; blockColC += L2BlockX) {
            for (int blockCol = 0; blockCol < L2BlockX; blockCol += 3) {
                /* note that we're handling 3 rows at a time, assuming L2BlockX = 3 */
                const unsigned matBToffset1 =
                                 (blockColC + blockCol + 0) * matBT.rowSpan,
                               matBToffset2 =
                                 (blockColC + blockCol + 1) * matBT.rowSpan,
                               matBToffset3 =
                                 (blockColC + blockCol + 2) * matBT.rowSpan;

                /* set up accumulators */
                __m256 a1, a2, a3, b1, b2, b3;
                __m256 c1 = _mm256_setzero_ps();
                __m256 c2 = _mm256_setzero_ps();
                __m256 c3 = _mm256_setzero_ps();
                __m256 c4 = _mm256_setzero_ps();
                __m256 c5 = _mm256_setzero_ps();
                __m256 c6 = _mm256_setzero_ps();
                __m256 c7 = _mm256_setzero_ps();
                __m256 c8 = _mm256_setzero_ps();
                __m256 c9 = _mm256_setzero_ps();

                /* 0.75 arithmetic intensity, 6 loads (3 a, 3 b) -> 9 fma instructions. */
                for (int pos = 0; pos < matA.width; pos += 8) {
                    /* 6 8f vector loads */
                    a1 = _mm256_load_ps(&matA.mat[matAoffset1 + pos]);
                    a2 = _mm256_load_ps(&matA.mat[matAoffset2 + pos]);
                    a3 = _mm256_load_ps(&matA.mat[matAoffset3 + pos]);

                    b1 = _mm256_load_ps(&matBT.mat[matBToffset1 + pos]);
                    b2 = _mm256_load_ps(&matBT.mat[matBToffset2 + pos]);
                    b3 = _mm256_load_ps(&matBT.mat[matBToffset3 + pos]);

                    /* 9 fma instructions */
                    c1 = _mm256_fmadd_ps(a1, b1, c1);
                    c2 = _mm256_fmadd_ps(a1, b2, c2);
                    c3 = _mm256_fmadd_ps(a1, b3, c3);

                    c4 = _mm256_fmadd_ps(a2, b1, c4);
                    c5 = _mm256_fmadd_ps(a2, b2, c5);
                    c6 = _mm256_fmadd_ps(a2, b3, c6);

                    c7 = _mm256_fmadd_ps(a3, b1, c7);
                    c8 = _mm256_fmadd_ps(a3, b2, c8);
                    c9 = _mm256_fmadd_ps(a3, b3, c9);
                }

                /* horizontal sum */

                memset(&accumulate[0], 0, 9 * sizeof(float));

                _mm256_store_ps(&fps[0], c1);
                _mm256_store_ps(&fps[8], c2);
                _mm256_store_ps(&fps[16], c3);
                _mm256_store_ps(&fps[24], c4);
                _mm256_store_ps(&fps[32], c5);
                _mm256_store_ps(&fps[40], c6);
                _mm256_store_ps(&fps[48], c7);
                _mm256_store_ps(&fps[56], c8);
                _mm256_store_ps(&fps[64], c9);

                for (int i = 0; i < 9; ++i) {
                    for (int j = 0; j < 8; ++j) {
                        accumulate[i] += fps[i * 8 + j];
                    }
                }

                /* stores */

                matData[(blockRowC + 0) * rowSpan + blockColC + blockCol + 0] =
                  accumulate[0];
                matData[(blockRowC + 0) * rowSpan + blockColC + blockCol + 1] =
                  accumulate[1];
                matData[(blockRowC + 0) * rowSpan + blockColC + blockCol + 2] =
                  accumulate[2];

                matData[(blockRowC + 1) * rowSpan + blockColC + blockCol + 0] =
                  accumulate[3];
                matData[(blockRowC + 1) * rowSpan + blockColC + blockCol + 1] =
                  accumulate[4];
                matData[(blockRowC + 1) * rowSpan + blockColC + blockCol + 2] =
                  accumulate[5];

                matData[(blockRowC + 2) * rowSpan + blockColC + blockCol + 0] =
                  accumulate[6];
                matData[(blockRowC + 2) * rowSpan + blockColC + blockCol + 1] =
                  accumulate[7];
                matData[(blockRowC + 2) * rowSpan + blockColC + blockCol + 2] =
                  accumulate[8];
            }
        }
        /* handle remanining columns (w<L2X x h<L2Y) */
        for (; blockColC < colC + blockX; ++blockColC) {
            const unsigned matBToffset = blockColC * matBT.rowSpan;

            /* set up accumulators */
            __m256 a11, a12, a21, a22, a31, a32, b1, b2;
            __m256 c1 = _mm256_setzero_ps();
            __m256 c2 = _mm256_setzero_ps();
            __m256 c3 = _mm256_setzero_ps();
            __m256 c4 = _mm256_setzero_ps();
            __m256 c5 = _mm256_setzero_ps();
            __m256 c6 = _mm256_setzero_ps();

            /* a and b's represent different segments, 
             * as they did in the previous one incremented loop.
             * see (w<L2X, h=L2Y) case for more details. */

            for (int pos = 0; pos < matA.width; pos += 16) {
                a11 = _mm256_load_ps(&matA.mat[matAoffset1 + pos]);
                a12 = _mm256_load_ps(&matA.mat[matAoffset1 + pos + 8]);

                a21 = _mm256_load_ps(&matA.mat[matAoffset2 + pos]);
                a22 = _mm256_load_ps(&matA.mat[matAoffset2 + pos + 8]);

                a31 = _mm256_load_ps(&matA.mat[matAoffset3 + pos]);
                a32 = _mm256_load_ps(&matA.mat[matAoffset3 + pos + 8]);

                b1 = _mm256_load_ps(&matBT.mat[matBToffset + pos]);
                b2 = _mm256_load_ps(&matBT.mat[matBToffset + pos + 8]);

                c1 = _mm256_fmadd_ps(a11, b1, c1);
                c2 = _mm256_fmadd_ps(a21, b1, c2);
                c3 = _mm256_fmadd_ps(a31, b1, c3);

                c4 = _mm256_fmadd_ps(a12, b2, c4);
                c5 = _mm256_fmadd_ps(a22, b2, c5);
                c6 = _mm256_fmadd_ps(a32, b2, c6);
            }

            /* horizontal sum */

            memset(&accumulate[0], 0, 3 * sizeof(float));

            c1 = _mm256_add_ps(c1, c4);
            c2 = _mm256_add_ps(c2, c5);
            c3 = _mm256_add_ps(c3, c6);

            _mm256_store_ps(&fps[0], c1);
            _mm256_store_ps(&fps[8], c2);
            _mm256_store_ps(&fps[16], c3);

            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 8; ++j) {
                    accumulate[i] += fps[i * 8 + j];
                }
            }

            matData[(blockRowC + 0) * rowSpan + blockColC] = accumulate[0];
            matData[(blockRowC + 1) * rowSpan + blockColC] = accumulate[1];
            matData[(blockRowC + 2) * rowSpan + blockColC] = accumulate[2];
        }
    }
    /* handle the very last row, h < 3 */
    for (; blockRowC < rowC + blockY; ++blockRowC) {
        const unsigned matAoffset = blockRowC * matA.rowSpan;
        int blockColC = colC;
        /* handle (L2X x h<3) blocks */
        for (; blockColC <= colC + blockX - L2BlockX; blockColC += L2BlockX) {
            for (int blockCol = 0; blockCol < L2BlockX; blockCol += 3) {
                /* note that we're handling 3 rows at a time, assuming L2BlockX = 3 */
                const unsigned matBToffset1 =
                                 (blockColC + blockCol + 0) * matBT.rowSpan,
                               matBToffset2 =
                                 (blockColC + blockCol + 1) * matBT.rowSpan,
                               matBToffset3 =
                                 (blockColC + blockCol + 2) * matBT.rowSpan;

                /* set up accumulators */
                __m256 a1, b1, b2, b3;
                __m256 c1 = _mm256_setzero_ps();
                __m256 c2 = _mm256_setzero_ps();
                __m256 c3 = _mm256_setzero_ps();

                /* 0.75 arithmetic intensity, 6 loads (3 a, 3 b) -> 9 fma instructions. */
                for (int pos = 0; pos < matA.width; pos += 8) {
                    /* 6 8f vector loads */
                    a1 = _mm256_load_ps(&matA.mat[matAoffset + pos]);

                    b1 = _mm256_load_ps(&matBT.mat[matBToffset1 + pos]);
                    b2 = _mm256_load_ps(&matBT.mat[matBToffset2 + pos]);
                    b3 = _mm256_load_ps(&matBT.mat[matBToffset3 + pos]);

                    /* 9 fma instructions */
                    c1 = _mm256_fmadd_ps(a1, b1, c1);
                    c2 = _mm256_fmadd_ps(a1, b2, c2);
                    c3 = _mm256_fmadd_ps(a1, b3, c3);
                }

                /* horizontal sum */

                memset(&accumulate[0], 0, 3 * sizeof(float));

                _mm256_store_ps(&fps[0], c1);
                _mm256_store_ps(&fps[8], c2);
                _mm256_store_ps(&fps[16], c3);

                for (int i = 0; i < 3; ++i) {
                    for (int j = 0; j < 8; ++j) {
                        accumulate[i] += fps[i * 8 + j];
                    }
                }

                /* stores */
                matData[blockRowC * rowSpan + blockColC + blockCol + 0] = accumulate[0];
                matData[blockRowC * rowSpan + blockColC + blockCol + 1] = accumulate[1];
                matData[blockRowC * rowSpan + blockColC + blockCol + 2] = accumulate[2];
            }
        }
        /* handle remanining columns (w<L2X x h<3) */
        for (; blockColC < colC + blockX; ++blockColC) {
            const unsigned matBToffset = blockColC * matBT.rowSpan;

            /* set up accumulators */
            __m256 a11, a12, a21, a22, a31, a32, b1, b2;
            __m256 c1 = _mm256_setzero_ps();
            __m256 c2 = _mm256_setzero_ps();

            /* now we only handle 1 x 1 rows,
             * <--------- A.w --------> 
             * [---- [a11] [a12] ---- ]
             * [---- [b1 ] [ b2] ---- ]
             */

            for (int pos = 0; pos < matA.width; pos += 16) {
                a11 = _mm256_load_ps(&matA.mat[matAoffset + pos]);
                a12 = _mm256_load_ps(&matA.mat[matAoffset + pos + 8]);

                b1 = _mm256_load_ps(&matBT.mat[matBToffset + pos]);
                b2 = _mm256_load_ps(&matBT.mat[matBToffset + pos + 8]);

                c1 = _mm256_fmadd_ps(a11, b1, c1);
                c2 = _mm256_fmadd_ps(a12, b2, c2);
            }

            /* horizontal sum */

            memset(&accumulate[0], 0, sizeof(float));

            c1 = _mm256_add_ps(c1, c2);

            _mm256_store_ps(&fps[0], c1);

            for (int j = 0; j < 8; ++j) {
                accumulate[0] += fps[j];
            }

            /* store */
            matData[blockRowC * rowSpan + blockColC] = accumulate[0];
        }
    }
}

/* This function computes the L3Y x L3X sized block of the output matrix C.
 * In order to keep this code nice and hot in instruction cache, 
 * keep it restricted to full blocks of L3X x L3Y. 
 * See MMHelper_MultRemBlocks for handling of the edges */
__declspec(noalias) void MMHelper_MultFullBlocks(float* __restrict const matData,
                                                 const unsigned rowSpan,
                                                 const Mat& matA, const Mat& matBT,
                                                 const unsigned colC,
                                                 const unsigned rowC,
                                                 const MMBlockInfo& mmBlockInfo)
{
    __declspec(align(32)) float fps[8 * 10];

    /* shorthand for some parameters */
    const unsigned L2BlockX = mmBlockInfo.L2BlockX, L2BlockY = mmBlockInfo.L2BlockY,
                   L3BlockX = mmBlockInfo.L3BlockX, L3BlockY = mmBlockInfo.L3BlockY;

    /* try to prefetch next L3 block into memory while still handling this one */
    {
        if constexpr (doPrefetch) {
            std::unique_lock<std::mutex> lock(prefetchMutex);
            if (!prefetched[rowC / L3BlockY][colC / L3BlockX] &&
                colC + L3BlockX < matBT.height && rowC + L3BlockY < matA.height) {
                for (int r = rowC; r < rowC + L3BlockY; ++r) {
                    for (int pos = 0; pos < matA.rowSpan; pos += cacheLineSz) {
                        _mm_prefetch((const char*)&matA.mat[r * matA.rowSpan + pos],
                                     _MM_HINT_T2);
                    }
                }
                for (int c = colC; c < colC + L3BlockX; ++c) {
                    for (int pos = 0; pos < matA.rowSpan; pos += cacheLineSz) {
                        _mm_prefetch((const char*)&matBT.mat[c * matBT.rowSpan + pos],
                                     _MM_HINT_T2);
                    }
                }
                prefetched[rowC / L3BlockY][colC / L3BlockX]++;
                //printf("L3 block starting from %d %d NOW FETCHING\n", rowC / L3BlockY, colC / L3BlockX);
            } else {
                //printf("L3 block starting from %d %d already prefetched\n",  rowC/L3BlockY, colC/L3BlockX);
            }
            if (!prefetched[rowC / L3BlockY][colC / L3BlockX + 1] &&
                colC + 2 * L3BlockX < matBT.height) {
                for (int c = colC + L3BlockX; c < colC + L3BlockX + L3BlockX / 2; ++c) {
                    for (int pos = 0; pos < matA.rowSpan; pos += cacheLineSz) {
                        _mm_prefetch((const char*)&matBT.mat[c * matBT.rowSpan + pos],
                                     _MM_HINT_T2);
                    }
                }
                prefetched[rowC / L3BlockY][colC / L3BlockX + 1]++;
            }
        }
    }

    /* we're issuing 2 threads per core, each handles L3BlockX/2 x L2BlockY blocks 
     * also, for dense arithmetic operations, optimize the inner loop
     * assuming L2BlockX % 3 == L2BlockY % 3 == 0 */
    for (int blockColC = colC; blockColC < colC + (L3BlockX >> 1);
         blockColC += L2BlockX) {
        for (int blockRowC = rowC; blockRowC < rowC + L3BlockY; blockRowC += L2BlockY) {
            for (int blockRow = 0; blockRow < L2BlockY; blockRow += 3) {
                for (int blockCol = 0; blockCol < L2BlockX; blockCol += 3) {
                    /* note that we're handling 3 rows at a time, assuming L2BlockX = 3 */
                    const unsigned matAoffset1 =
                                     (blockRowC + blockRow + 0) * matA.rowSpan,
                                   matAoffset2 =
                                     (blockRowC + blockRow + 1) * matA.rowSpan,
                                   matAoffset3 =
                                     (blockRowC + blockRow + 2) * matA.rowSpan,
                                   matBToffset1 =
                                     (blockColC + blockCol + 0) * matBT.rowSpan,
                                   matBToffset2 =
                                     (blockColC + blockCol + 1) * matBT.rowSpan,
                                   matBToffset3 =
                                     (blockColC + blockCol + 2) * matBT.rowSpan;

                    /* Visualization:
                     * 
                     * <-----A.w----> <-----A.w---->
                     * [----[a1]----] [----[b1]----]
                     * [----[a2]----] [----[b2]----]
                     * [----[a3]----] [----[b3]----]
                     *      ^ row          ^col   
                     * 
                     * Unlike previous iterations where the program computed 
                     * the dot product between 2 rows using 8x8f vectors, 
                     * we are now computing dot product of 3 rows and 3 columns 
                     * at the same time, 1x8f vectors at a time.
                     * This allows for much better register usage and FLOP/load ratio. */

                    /* set up accumulator SIMD variables */
                    __m256 a1, a2, a3, b1, b2, b3;
                    __m256 c1 = _mm256_setzero_ps();
                    __m256 c2 = _mm256_setzero_ps();
                    __m256 c3 = _mm256_setzero_ps();
                    __m256 c4 = _mm256_setzero_ps();
                    __m256 c5 = _mm256_setzero_ps();
                    __m256 c6 = _mm256_setzero_ps();
                    __m256 c7 = _mm256_setzero_ps();
                    __m256 c8 = _mm256_setzero_ps();
                    __m256 c9 = _mm256_setzero_ps();

                    /* 0.75 arithmetic intensity, 6 loads (3 a, 3 b) -> 9 fma instructions. */
                    for (int pos = 0; pos < matA.width; pos += 8) {
                        /* 6 8f vector loads */
                        a1 = _mm256_load_ps(&matA.mat[matAoffset1 + pos]);
                        a2 = _mm256_load_ps(&matA.mat[matAoffset2 + pos]);
                        a3 = _mm256_load_ps(&matA.mat[matAoffset3 + pos]);

                        b1 = _mm256_load_ps(&matBT.mat[matBToffset1 + pos]);
                        b2 = _mm256_load_ps(&matBT.mat[matBToffset2 + pos]);
                        b3 = _mm256_load_ps(&matBT.mat[matBToffset3 + pos]);

                        /* 9 fma instructions */
                        c1 = _mm256_fmadd_ps(a1, b1, c1);
                        c2 = _mm256_fmadd_ps(a1, b2, c2);
                        c3 = _mm256_fmadd_ps(a1, b3, c3);

                        c4 = _mm256_fmadd_ps(a2, b1, c4);
                        c5 = _mm256_fmadd_ps(a2, b2, c5);
                        c6 = _mm256_fmadd_ps(a2, b3, c6);

                        c7 = _mm256_fmadd_ps(a3, b1, c7);
                        c8 = _mm256_fmadd_ps(a3, b2, c8);
                        c9 = _mm256_fmadd_ps(a3, b3, c9);
                    }

                    /* horizontal sum */

                    __declspec(align(32)) float accumulate[9];
                    memset(&accumulate[0], 0, 9 * sizeof(float));

                    _mm256_store_ps(&fps[0], c1);
                    _mm256_store_ps(&fps[8], c2);
                    _mm256_store_ps(&fps[16], c3);
                    _mm256_store_ps(&fps[24], c4);
                    _mm256_store_ps(&fps[32], c5);
                    _mm256_store_ps(&fps[40], c6);
                    _mm256_store_ps(&fps[48], c7);
                    _mm256_store_ps(&fps[56], c8);
                    _mm256_store_ps(&fps[64], c9);

                    for (int i = 0; i < 9; ++i) {
                        for (int j = 0; j < 8; ++j) {
                            accumulate[i] += fps[i * 8 + j];
                        }
                    }

                    /* stores */
                    matData[(blockRowC + blockRow + 0) * rowSpan + blockColC +
                            blockCol + 0] = accumulate[0];
                    matData[(blockRowC + blockRow + 0) * rowSpan + blockColC +
                            blockCol + 1] = accumulate[1];
                    matData[(blockRowC + blockRow + 0) * rowSpan + blockColC +
                            blockCol + 2] = accumulate[2];

                    matData[(blockRowC + blockRow + 1) * rowSpan + blockColC +
                            blockCol + 0] = accumulate[3];
                    matData[(blockRowC + blockRow + 1) * rowSpan + blockColC +
                            blockCol + 1] = accumulate[4];
                    matData[(blockRowC + blockRow + 1) * rowSpan + blockColC +
                            blockCol + 2] = accumulate[5];

                    matData[(blockRowC + blockRow + 2) * rowSpan + blockColC +
                            blockCol + 0] = accumulate[6];
                    matData[(blockRowC + blockRow + 2) * rowSpan + blockColC +
                            blockCol + 1] = accumulate[7];
                    matData[(blockRowC + blockRow + 2) * rowSpan + blockColC +
                            blockCol + 2] = accumulate[8];
                }
            }
        }
    }
}

/* This function divides the matrix multiplication into segments and
 * issues commands for a cache aware thread pool to handle them. */
__declspec(noalias) const Mat MTMatMul(const Mat& matA, const Mat& matB)
{
    /* allocate the aligned float array for our new matrix C */
    float* __restrict const matData =
      (float*)_aligned_malloc(matA.height * matB.rowSpan * sizeof(float), AVX_ALIGN);

    /* construct matrix C */
    Mat matC{matB.width, matA.height, matB.rowSpan, matData};

    /* for the sake of cache, we'll be working with transposed B */
    const Mat matBT = TransposeMat(matB);

    /* initialize the HWLocalThreadPool with 2 threads per physical core 
     * and 6 physical cores */
    HWLocalThreadPool<6, 2> tp;

    /* before we even being, start prefetching the first L3 level block */
    memset(&prefetched[0][0], 0, 1024 * 1024 * sizeof(int));
    //for (int r = 0; r < L3BlockY; ++r) {
    //    for (int pos = 0; pos < matA.rowSpan; pos += cacheLineSz) {
    //        _mm_prefetch((const char*)&matA.mat[r*matA.rowSpan + pos], _MM_HINT_T2);
    //    }
    //}
    //for (int c = 0; c < L3BlockX; ++c) {
    //    for (int pos = 0; pos < matA.rowSpan; pos += cacheLineSz) {
    //        _mm_prefetch((const char*)&matBT.mat[c*matBT.rowSpan + pos], _MM_HINT_T2);
    //    }
    //
    //prefetched[0][0]++;

    /* decide the best block sizes for the given matrix and CPU */
    const int L3BlockX = 60, L3BlockY = 60;
    const int L2BlockX = 6, L2BlockY = 6;
    const int halfL3X = L3BlockX >> 1;


    MMBlockInfo mmBlockInfo{L3BlockX, L3BlockY, L2BlockX, L2BlockY};

    /* start issuing jobs for the thread pool */

    int largeBlockRowC = 0;
    /* handle L3Y sized rows 
     * cast unsigned dimensions to signed to avoid UB 
     * for small matrices where N < L3 block size */
    for (; largeBlockRowC <= (int)matA.height - L3BlockY; largeBlockRowC += L3BlockY) {
        int largeBlockColC = 0;
        /* handle L3X x L3Y sized blocks */
        for (; largeBlockColC <= (int)matB.width - L3BlockX; largeBlockColC += L3BlockX) {
            tp.Add({HWLocalThreadPool<>::WrapFunc(
                      MMHelper_MultFullBlocks, matData, matB.rowSpan, matA, matBT,
                      largeBlockColC, largeBlockRowC, mmBlockInfo),
                    HWLocalThreadPool<>::WrapFunc(
                      MMHelper_MultFullBlocks, matData, matB.rowSpan, matA, matBT,
                      largeBlockColC + halfL3X, largeBlockRowC, mmBlockInfo)});
        }
        /* handle the block w < L3X, h = L3Y at the end of the column */
        if (matB.width > largeBlockColC) {
            const unsigned remSubX = (matB.width - largeBlockColC) / 2;
            tp.Add({HWLocalThreadPool<>::WrapFunc(
                      MMHelper_MultRemBlocks, matData, matB.rowSpan, matA, matBT,
                      largeBlockColC, largeBlockRowC, remSubX, L3BlockY, mmBlockInfo),
                    HWLocalThreadPool<>::WrapFunc(
                      MMHelper_MultRemBlocks, matData, matB.rowSpan, matA, matBT,
                      largeBlockColC + remSubX, largeBlockRowC,
                      matB.width - largeBlockColC - remSubX, L3BlockY, mmBlockInfo)});
        }
    }
    int largeBlockColC = 0;
    /* handle last row, h < L3Y */
    /* first handle blocks of w = L3X, h < L3Y */
    for (; largeBlockColC <= (int)matB.width - L3BlockX; largeBlockColC += L3BlockX) {
        tp.Add({HWLocalThreadPool<>::WrapFunc(
                    MMHelper_MultRemBlocks, matData, matB.rowSpan, matA, matBT,
                    largeBlockColC, largeBlockRowC, halfL3X,
                    matA.height - largeBlockRowC, mmBlockInfo),
                HWLocalThreadPool<>::WrapFunc(
                    MMHelper_MultRemBlocks, matData, matB.rowSpan, matA, matBT,
                    largeBlockColC + halfL3X, largeBlockRowC, halfL3X,
                    matA.height - largeBlockRowC, mmBlockInfo)});
    }
    /* now handle the rightmost block of w < L3X, h < L3Y */
    tp.Add({HWLocalThreadPool<>::WrapFunc(MMHelper_MultRemBlocks, matData, matB.rowSpan,
                                          matA, matBT, largeBlockColC, largeBlockRowC,
                                          matB.width - largeBlockColC,
                                          matA.height - largeBlockRowC, mmBlockInfo),
            []() {}});

    /* -- commands issued -- */

    /* wait for the thread pool to finish */
    tp.Close();
    /* free the temporary bT matrix */
    _aligned_free(matBT.mat);

    return matC;
}

/* 
 * EDIT: --NEEDS MORE TUNING--
 * A very, very simple toggle between two functions depending on the total number of ops 
 */
const Mat MatMul(const Mat& matA, const Mat& matB)
{
    /* A:  a,b B: b,c => # of op: a*b*b*c */
    //if (matA.height*matA.width*matA.width*matB.width < 125000000) {
    //    return ST_TransposedBMatMul(matA, matB);
    //}
    return MTMatMul(matA, matB);
}

int __cdecl main(int argc, char* argv[])
{
    //if (argc < 4) {
    //    std::cout << "No args\n";
    //    return 0;
    //}

    //const char* inputMtxAFile = argv[1];
    //const char* inputMtxBFile = argv[2];
    //const char* outMtxABFile = argv[3];

    const char* inputMtxAFile = "matrixA.bin";
    const char* inputMtxBFile = "matrixB.bin";
    const char* outMtxABFile = "matrixAB-out.bin";

    const Mat inputMtxA = LoadMat(inputMtxAFile);
    const Mat inputMtxB = LoadMat(inputMtxBFile);

    auto start = std::chrono::high_resolution_clock::now();

    const Mat outMtxAB = MTMatMul(inputMtxA, inputMtxB);

    auto end = std::chrono::high_resolution_clock::now();

    std::cout
      << "Matrix Multiplication: "
      << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
      << " microseconds.\n";

    DumpMat(outMtxABFile, outMtxAB);

    return 0;
}
