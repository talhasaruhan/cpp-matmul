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
#include <numeric>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <immintrin.h>
#include "ThreadPool.h"

/* Define for AVX alignment requirements */
#define AVX_ALIGN 32

/* Define CPU related variables, actual values will be queried on runtime. */
int CPUInfoQueried = 0;
int L2Size = 256 * 1024;
int L3Size = 12 * 1024 * 1024;
int cacheLineSz = 64;
int numHWCores = 6;

/* Prefetching switches, if multiple MatMul operations are intended to run in parallel,
 * individual mutexes should be created for each one. */
constexpr int doL3Prefetch = 0;
constexpr int doL12Prefetch = 0;
int prefetched[1024][1024];
std::mutex prefetchMutex;

/* Matrix structure */
typedef struct Mat {
    unsigned width;
    unsigned height;
    unsigned rowSpan;
    /* guarantee that mat will not be aliased (__restrict),
    no need for two matrices to point at sama data */
    float* __restrict mat;
} Mat;

/* 
 * This struct holds the information for multiple levels of block sizes.
 * It's used to keep function parameters short and readable
 * Constraints on block sizes:
 * L2BlockX % 3 == L2BlockY % 4 == 0,
 * L3BlockX % 2 == L3BlockY % 2 == 0,
 * (L3BlockX / 2) % L2BlockX == 0  
 */
typedef struct MMBlockInfo {
    const unsigned L3BlockX, L3BlockY;
    const unsigned L2BlockX, L2BlockY;
    const unsigned issuedBlockSzX, issuedBlockSzY;
} MMBlockInfo;

/* Load a previously saved matrix from disk */
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

/* Dump the given matrix to the disk. */
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

/* Deallocate matrix data */
void FreeMat(Mat& mat)
{
    if (!mat.mat)
        return;
    _aligned_free(mat.mat);
    mat.mat = NULL;
}
void FreeMat(const Mat& mat)
{
    if (!mat.mat)
        return;
    _aligned_free(mat.mat);
}

/* Round a given number to the nearest multiple of K,
* where K is a parameter and is a power of 2 */
static unsigned RoundUpPwr2(unsigned val, unsigned pwr2)
{
    return (val + (pwr2 - 1)) & (~(pwr2 - 1));
}

/* Compute the transpose of a given matrix.
 * A singlethreaded implementation without block tiling. */
__declspec(noalias) const Mat TransposeMat(const Mat& mat)
{
    const unsigned tRowSpan = RoundUpPwr2(mat.height, 64 / sizeof(float));
    float* __restrict const tData =
      (float*)_aligned_malloc(mat.width * tRowSpan * sizeof(float), AVX_ALIGN);

    Mat T{mat.height, mat.width, tRowSpan, tData};

    // the loops are truly interchangable as we encounter a cache miss either ways
    for (int rowT = 0; rowT < T.height; ++rowT) {
        for (int colT = 0; colT < T.width; ++colT) {
            tData[rowT * tRowSpan + colT] = mat.mat[colT * mat.rowSpan + rowT];
        }
    }

    return T;
}

/* Print the given matrix to given std::ostream */
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

/**************** Naive, initial implementations ****************/

/* Naive MatMul */
const Mat ST_NaiveMatMul(const Mat& matA, const Mat& matB)
{
    /* First : naive solution with but with some tricks to make compiler (MSVC) behave
     * Note that, in this case, manually unrolling the loop helps
     * as the compiler can't auto-vectorize non-contagious memory access */
    float* __restrict const matData =
      (float*)_aligned_malloc(matA.height * matB.rowSpan * sizeof(float), AVX_ALIGN);

    Mat matC{matB.width, matA.height, matB.rowSpan, matData};

    for (int rowC = 0; rowC < matA.height; ++rowC) {
        for (int colC = 0; colC < matB.width; ++colC) {
            /* an independent, local accumulator. */
            float accumulate = 0;
            int pos = 0;
            /* manual unrolling IS helpful in this case */
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

/* MatMul with transposed B for improved cache behavior. */
const Mat ST_TransposedBMatMul(const Mat& matA, const Mat& matB)
{
    /* 
     * Now, transposing B and then traversing it row order seemed promising!
     * Also, note that, if we manually unrolled the loop here, 
     * compiler wouldn't vectorize the loop, 
     * so we keep it simple and let MSVC auto vectorize this.
     */
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

/* 
 * MatMul with a different traversal order. 
 * Instead of linearly running thru whole rows of output matrix C, 
 * calculate blocks of a certain size at a time. 
 */
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

/************** ~~Naive, initial implementations~~ **************/

/* Declerations for helper functions for the final implementation */

__declspec(noalias) void MMHelper_MultAnyBlocks(float* __restrict const matData,
                                                const unsigned rowSpan, const Mat& matA,
                                                const Mat& matBT, const unsigned colC,
                                                const unsigned rowC, const int blockX,
                                                const int blockY,
                                                const MMBlockInfo& mmBlockInfo);

__declspec(noalias) void MMHelper_MultL2Blocks(float* __restrict const matData,
                                               const unsigned rowSpan, const Mat& matA,
                                               const Mat& matBT, const unsigned col,
                                               const unsigned row,
                                               const unsigned L2BlockX,
                                               const unsigned L2BlockY);

__declspec(noalias) void MMHelper_MultFullBlocks(float* __restrict const matData,
                                                 const unsigned rowSpan,
                                                 const Mat& matA, const Mat& matBT,
                                                 const unsigned colC,
                                                 const unsigned rowC,
                                                 const MMBlockInfo& mmBlockInfo);

/* Declarations for helper functions that handle NxM blocks */

__declspec(noalias) void MMHelper_Mult4x3Blocks(float* __restrict const matData,
                                                const unsigned rowSpan, const Mat& matA,
                                                const Mat& matBT, const unsigned col,
                                                const unsigned row);
__declspec(noalias) void MMHelper_Mult4x1Blocks(float* __restrict const matData,
                                                const unsigned rowSpan, const Mat& matA,
                                                const Mat& matBT, const unsigned col,
                                                const unsigned row);
__declspec(noalias) void MMHelper_Mult1x3Blocks(float* __restrict const matData,
                                                const unsigned rowSpan, const Mat& matA,
                                                const Mat& matBT, const unsigned col,
                                                const unsigned row);
__declspec(noalias) void MMHelper_Mult1x1Blocks(float* __restrict const matData,
                                                const unsigned rowSpan, const Mat& matA,
                                                const Mat& matBT, const unsigned col,
                                                const unsigned row);

/* 
 * Helper function for computing a block out of the output matrix C.
 * This function is used for the residues at the edges 
 * after the majority of the matrix is computed as KxK sized blocks.
 * (t,l,b,r)->(row, col, row+blockY, col+blockX). 
 */
__declspec(noalias) void MMHelper_MultAnyBlocks(float* __restrict const matData,
                                                const unsigned rowSpan, const Mat& matA,
                                                const Mat& matBT, const unsigned colC,
                                                const unsigned rowC, const int blockX,
                                                const int blockY,
                                                const MMBlockInfo& mmBlockInfo)
{
    /* if no work to be done, exit */
    if (blockX <= 0 || blockY <= 0)
        return;

    /* shorthand for some parameters */
    const unsigned L2BlockX = mmBlockInfo.L2BlockX, L2BlockY = mmBlockInfo.L2BlockY,
                   L3BlockX = mmBlockInfo.L3BlockX, L3BlockY = mmBlockInfo.L3BlockY;

    int blockRowC = rowC;
    /* handle full L2Y sized rows */
    for (; blockRowC <= rowC + blockY - L2BlockY; blockRowC += L2BlockY) {
        int blockColC = colC;
        /* handle (L2X x L2Y) blocks */
        for (; blockColC <= colC + blockX - L2BlockX; blockColC += L2BlockX) {
            MMHelper_MultL2Blocks(matData, rowSpan, matA, matBT, blockColC, blockRowC,
                                  L2BlockX, L2BlockY);
        }
        /* handle the remaining columns, (w<L2X, h=L2Y) */
        for (int blockRow = blockRowC; blockRow < blockRowC + L2BlockY; blockRow += 4) {
            int blockCol = blockColC;
            if ((colC + blockX - blockColC) > 4) {
                for (; blockCol <= colC + blockX - 3; blockCol += 3) {
                    MMHelper_Mult4x3Blocks(matData, rowSpan, matA, matBT, blockCol,
                                           blockRow);
                }
            }
            for (; blockCol < colC + blockX; ++blockCol) {
                MMHelper_Mult4x1Blocks(matData, rowSpan, matA, matBT, blockCol,
                                       blockRow);
            }
        }
    }
    /* handle rest of the rows, h<L2Y, h%4=0 */
    for (; blockRowC <= rowC + blockY - 4; blockRowC += 4) {
        int blockColC = colC;
        /* handle (L2X x h<L2Y), h%4==0 blocks */
        for (; blockColC <= colC + blockX - L2BlockX; blockColC += L2BlockX) {
            for (int blockCol = 0; blockCol < L2BlockX; blockCol += 3) {
                MMHelper_Mult4x3Blocks(matData, rowSpan, matA, matBT,
                                       blockColC + blockCol, blockRowC);
            }
        }
        /* handle remanining columns (w<L2X x h<L2Y), h%4==0 */
        for (; blockColC < colC + blockX; ++blockColC) {
            MMHelper_Mult4x1Blocks(matData, rowSpan, matA, matBT, blockColC, blockRowC);
        }
    }
    /* handle the very last row, h < 4 */
    for (; blockRowC < rowC + blockY; ++blockRowC) {
        int blockColC = colC;
        /* handle (L2X x h<3) blocks */
        for (; blockColC <= colC + blockX - L2BlockX; blockColC += L2BlockX) {
            for (int blockCol = 0; blockCol < L2BlockX; blockCol += 3) {
                MMHelper_Mult1x3Blocks(matData, rowSpan, matA, matBT,
                                       blockColC + blockCol, blockRowC);
            }
        }
        /* handle remanining columns (w<L2X x h<3) */
        for (; blockColC < colC + blockX; ++blockColC) {
            MMHelper_Mult1x1Blocks(matData, rowSpan, matA, matBT, blockColC, blockRowC);
        }
    }
}

/* Calculates the dot product corresponding to a single entry in matrix C. */
__declspec(noalias) void MMHelper_Mult1x1Blocks(float* __restrict const matData,
                                                const unsigned rowSpan, const Mat& matA,
                                                const Mat& matBT, const unsigned col,
                                                const unsigned row)
{
    /* scalar accumulator */
    __declspec(align(32)) float fps[8];
    __declspec(align(32)) float accumulate;

    const unsigned matAoffset = row * matA.rowSpan;
    const unsigned matBToffset = col * matBT.rowSpan;

    /* SIMD accumulators */
    __m256 a1, a2, b1, b2;
    __m256 c1 = _mm256_setzero_ps();
    __m256 c2 = _mm256_setzero_ps();

    /* handle 1 x 1 rows, 2x8f vectors at a time
    * <-------- A.w ------->
    * [---- [a1] [a2] ---- ]
    * [---- [b1] [b2] ---- ]
    */

    for (int pos = 0; pos < matA.width; pos += 16) {
        a1 = _mm256_load_ps(&matA.mat[matAoffset + pos]);
        a2 = _mm256_load_ps(&matA.mat[matAoffset + pos + 8]);

        b1 = _mm256_load_ps(&matBT.mat[matBToffset + pos]);
        b2 = _mm256_load_ps(&matBT.mat[matBToffset + pos + 8]);

        c1 = _mm256_fmadd_ps(a1, b1, c1);
        c2 = _mm256_fmadd_ps(a2, b2, c2);
    }

    c1 = _mm256_add_ps(c1, c2);
    _mm256_store_ps(&fps[0], c1);

    accumulate = 0;
    for (int i = 0; i < 8; ++i) {
        accumulate += fps[i];
    }

    /* store */
    matData[row * rowSpan + col] = accumulate;
}

/* Calculates a 1x3 block on the matrix C, (t,l,b,r)->(row,col,row+1,col+3) */
__declspec(noalias) void MMHelper_Mult1x3Blocks(float* __restrict const matData,
                                                const unsigned rowSpan, const Mat& matA,
                                                const Mat& matBT, const unsigned col,
                                                const unsigned row)
{
    /* set up scalar array and accumulators for doing the horizontal sum (__m256 -> f32)
     * and storing its value. Horizontal sum is auto-vectorized by the compiler anyways. */
    __declspec(align(32)) float fps[8 * 3];
    __declspec(align(32)) float accumulate[3];

    /* we will be reusing these */
    const unsigned matAoffset = row * matA.rowSpan;
    const unsigned matBToffset1 = (col + 0) * matBT.rowSpan,
                   matBToffset2 = (col + 1) * matBT.rowSpan,
                   matBToffset3 = (col + 2) * matBT.rowSpan;

    /* set up accumulators */
    __m256 a1, b1, b2, b3;
    __m256 c1 = _mm256_setzero_ps();
    __m256 c2 = _mm256_setzero_ps();
    __m256 c3 = _mm256_setzero_ps();

    for (int pos = 0; pos < matA.width; pos += 8) {
        a1 = _mm256_load_ps(&matA.mat[matAoffset + pos]);

        b1 = _mm256_load_ps(&matBT.mat[matBToffset1 + pos]);
        b2 = _mm256_load_ps(&matBT.mat[matBToffset2 + pos]);
        b3 = _mm256_load_ps(&matBT.mat[matBToffset3 + pos]);

        c1 = _mm256_fmadd_ps(a1, b1, c1);
        c2 = _mm256_fmadd_ps(a1, b2, c2);
        c3 = _mm256_fmadd_ps(a1, b3, c3);
    }

    /* horizontal sum */

    memset(&accumulate[0], 0, 3 * sizeof(float));

    _mm256_store_ps(&fps[0], c1);
    _mm256_store_ps(&fps[8], c2);
    _mm256_store_ps(&fps[16], c3);

    /* autovectorized */
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 8; ++j) {
            accumulate[i] += fps[i * 8 + j];
        }
    }

    /* stores */
    matData[row * rowSpan + col + 0] = accumulate[0];
    matData[row * rowSpan + col + 1] = accumulate[1];
    matData[row * rowSpan + col + 2] = accumulate[2];
}

/* Calculates a 4x1 block on output matrix C. (t,l,b,r)->(row,col,row+4,col+1) */
__declspec(noalias) void MMHelper_Mult4x1Blocks(float* __restrict const matData,
                                                const unsigned rowSpan, const Mat& matA,
                                                const Mat& matBT, const unsigned col,
                                                const unsigned row)
{
    /* set up scalar array and accumulators for doing the horizontal sum (__m256 -> f32)
    * and storing its value. Horizontal sum is auto-vectorized by the compiler anyways. */
    __declspec(align(32)) float fps[8 * 12];
    __declspec(align(32)) float accumulate[8 * 12];

    const unsigned matAoffset1 = (row + 0) * matA.rowSpan,
                   matAoffset2 = (row + 1) * matA.rowSpan,
                   matAoffset3 = (row + 2) * matA.rowSpan,
                   matAoffset4 = (row + 3) * matA.rowSpan;

    const unsigned matBToffset = col * matBT.rowSpan;

    /* set up accumulators */
    __m256 a11, a12, a21, a22, a31, a32, a41, a42, b1, b2;
    __m256 c1 = _mm256_setzero_ps();
    __m256 c2 = _mm256_setzero_ps();
    __m256 c3 = _mm256_setzero_ps();
    __m256 c4 = _mm256_setzero_ps();
    __m256 c5 = _mm256_setzero_ps();
    __m256 c6 = _mm256_setzero_ps();
    __m256 c7 = _mm256_setzero_ps();
    __m256 c8 = _mm256_setzero_ps();

    for (int pos = 0; pos < matA.width; pos += 16) {
        a11 = _mm256_load_ps(&matA.mat[matAoffset1 + pos]);
        a12 = _mm256_load_ps(&matA.mat[matAoffset1 + pos + 8]);

        a21 = _mm256_load_ps(&matA.mat[matAoffset2 + pos]);
        a22 = _mm256_load_ps(&matA.mat[matAoffset2 + pos + 8]);

        a31 = _mm256_load_ps(&matA.mat[matAoffset3 + pos]);
        a32 = _mm256_load_ps(&matA.mat[matAoffset3 + pos + 8]);

        a41 = _mm256_load_ps(&matA.mat[matAoffset4 + pos]);
        a42 = _mm256_load_ps(&matA.mat[matAoffset4 + pos + 8]);

        b1 = _mm256_load_ps(&matBT.mat[matBToffset + pos]);
        b2 = _mm256_load_ps(&matBT.mat[matBToffset + pos + 8]);

        c1 = _mm256_fmadd_ps(a11, b1, c1);
        c2 = _mm256_fmadd_ps(a21, b1, c2);
        c3 = _mm256_fmadd_ps(a31, b1, c3);
        c4 = _mm256_fmadd_ps(a41, b1, c4);

        c5 = _mm256_fmadd_ps(a12, b2, c5);
        c6 = _mm256_fmadd_ps(a22, b2, c6);
        c7 = _mm256_fmadd_ps(a32, b2, c7);
        c8 = _mm256_fmadd_ps(a42, b2, c8);
    }

    /* horizontal sum */

    memset(&accumulate[0], 0, 4 * sizeof(float));

    c1 = _mm256_add_ps(c1, c5);
    c2 = _mm256_add_ps(c2, c6);
    c3 = _mm256_add_ps(c3, c7);
    c4 = _mm256_add_ps(c4, c8);

    _mm256_store_ps(&fps[0], c1);
    _mm256_store_ps(&fps[8], c2);
    _mm256_store_ps(&fps[16], c3);
    _mm256_store_ps(&fps[24], c4);

    /* autovectorized */
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 8; ++j) {
            accumulate[i] += fps[i * 8 + j];
        }
    }

    /* stores */
    matData[(row + 0) * rowSpan + col] = accumulate[0];
    matData[(row + 1) * rowSpan + col] = accumulate[1];
    matData[(row + 2) * rowSpan + col] = accumulate[2];
    matData[(row + 3) * rowSpan + col] = accumulate[3];
}

/* Calculates a 4x3 block on output matrix C. (t,l,b,r)->(row,col,row+4,col+3) */
__declspec(noalias) void MMHelper_Mult4x3Blocks(float* __restrict const matData,
                                                const unsigned rowSpan, const Mat& matA,
                                                const Mat& matBT, const unsigned col,
                                                const unsigned row)
{
    /* aligned placeholders and accumulators */
    __declspec(align(32)) float fps[8 * 12];
    __declspec(align(32)) float accumulate[12];

    const unsigned matAoffset1 = (row + 0) * matA.rowSpan,
                   matAoffset2 = (row + 1) * matA.rowSpan,
                   matAoffset3 = (row + 2) * matA.rowSpan,
                   matAoffset4 = (row + 3) * matA.rowSpan,
                   matBToffset1 = (col + 0) * matBT.rowSpan,
                   matBToffset2 = (col + 1) * matBT.rowSpan,
                   matBToffset3 = (col + 2) * matBT.rowSpan;

    /* 
     * <-----A.w----> <-----A.w---->
     * [----[a1]----] [----[b1]----]
     * [----[a2]----] [----[b2]----]
     * [----[a3]----] [----[b3]----]
     * [----[a4]----]      ^col
     *      ^ row          
     *
     * we are now computing dot product of 3 rows and 3 columns
     * at the same time, 1x8f vectors at a time.
     *
     * 3 ymm registers for b1:3,
     * 4*3 = 12 registers for the accumulators
     * 1 register for the temporary ai value loaded.
     * All 16 registers are used.
     * High arithmetic density: 7 loads -> 12 fma instructions
     *
     */

    /* set up SIMD variables */
    __m256 a, b1, b2, b3;
    __m256 c1 = _mm256_setzero_ps();
    __m256 c2 = _mm256_setzero_ps();
    __m256 c3 = _mm256_setzero_ps();
    __m256 c4 = _mm256_setzero_ps();
    __m256 c5 = _mm256_setzero_ps();
    __m256 c6 = _mm256_setzero_ps();
    __m256 c7 = _mm256_setzero_ps();
    __m256 c8 = _mm256_setzero_ps();
    __m256 c9 = _mm256_setzero_ps();
    __m256 c10 = _mm256_setzero_ps();
    __m256 c11 = _mm256_setzero_ps();
    __m256 c12 = _mm256_setzero_ps();

    /* if prefetch switch is set, 
     * prefetch first sections, one cache line at a time */
    if constexpr (doL12Prefetch) {
        _mm_prefetch((const char*)&matA.mat[matAoffset1], _MM_HINT_T0);
        _mm_prefetch((const char*)&matA.mat[matAoffset2], _MM_HINT_T0);
        _mm_prefetch((const char*)&matA.mat[matAoffset3], _MM_HINT_T0);
        _mm_prefetch((const char*)&matA.mat[matAoffset4], _MM_HINT_T0);

        _mm_prefetch((const char*)&matBT.mat[matBToffset1], _MM_HINT_T0);
        _mm_prefetch((const char*)&matBT.mat[matBToffset2], _MM_HINT_T0);
        _mm_prefetch((const char*)&matBT.mat[matBToffset3], _MM_HINT_T0);
    }

    /* do the dot products */
    for (int pos = 0; pos < matA.width; pos += 8) {
        if constexpr (doL12Prefetch) {
            if ((pos & (unsigned)15)) {
                _mm_prefetch((const char*)&matA.mat[matAoffset1 + pos + 8],
                             _MM_HINT_T0);
            }
        }

        b1 = _mm256_load_ps(&matBT.mat[matBToffset1 + pos]);
        b2 = _mm256_load_ps(&matBT.mat[matBToffset2 + pos]);
        b3 = _mm256_load_ps(&matBT.mat[matBToffset3 + pos]);

        if constexpr (doL12Prefetch) {
            if ((pos & (unsigned)15)) {
                _mm_prefetch((const char*)&matA.mat[matAoffset2 + pos + 8],
                             _MM_HINT_T0);
            }
        }

        a = _mm256_load_ps(&matA.mat[matAoffset1 + pos]);
        c1 = _mm256_fmadd_ps(a, b1, c1);
        c2 = _mm256_fmadd_ps(a, b2, c2);
        c3 = _mm256_fmadd_ps(a, b3, c3);

        if constexpr (doL12Prefetch) {
            if ((pos & (unsigned)15)) {
                _mm_prefetch((const char*)&matA.mat[matAoffset3 + pos + 8],
                             _MM_HINT_T0);
            }
        }
        a = _mm256_load_ps(&matA.mat[matAoffset2 + pos]);
        c4 = _mm256_fmadd_ps(a, b1, c4);
        c5 = _mm256_fmadd_ps(a, b2, c5);
        c6 = _mm256_fmadd_ps(a, b3, c6);

        if constexpr (doL12Prefetch) {
            if ((pos & (unsigned)15)) {
                _mm_prefetch((const char*)&matA.mat[matAoffset4 + pos + 8],
                             _MM_HINT_T0);
            }
        }

        a = _mm256_load_ps(&matA.mat[matAoffset3 + pos]);
        c7 = _mm256_fmadd_ps(a, b1, c7);
        c8 = _mm256_fmadd_ps(a, b2, c8);
        c9 = _mm256_fmadd_ps(a, b3, c9);

        if constexpr (doL12Prefetch) {
            if ((pos & (unsigned)15)) {
                _mm_prefetch((const char*)&matBT.mat[matBToffset1 + pos + 8],
                             _MM_HINT_T0);
                _mm_prefetch((const char*)&matBT.mat[matBToffset2 + pos + 8],
                             _MM_HINT_T0);
                _mm_prefetch((const char*)&matBT.mat[matBToffset3 + pos + 8],
                             _MM_HINT_T0);
            }
        }

        a = _mm256_load_ps(&matA.mat[matAoffset4 + pos]);
        c10 = _mm256_fmadd_ps(a, b1, c10);
        c11 = _mm256_fmadd_ps(a, b2, c11);
        c12 = _mm256_fmadd_ps(a, b3, c12);
    }

    /* horizontal sum */
    memset(&accumulate[0], 0, 12 * sizeof(float));

    _mm256_store_ps(&fps[0], c1);
    _mm256_store_ps(&fps[8], c2);
    _mm256_store_ps(&fps[16], c3);
    _mm256_store_ps(&fps[24], c4);
    _mm256_store_ps(&fps[32], c5);
    _mm256_store_ps(&fps[40], c6);
    _mm256_store_ps(&fps[48], c7);
    _mm256_store_ps(&fps[56], c8);
    _mm256_store_ps(&fps[64], c9);
    _mm256_store_ps(&fps[72], c10);
    _mm256_store_ps(&fps[80], c11);
    _mm256_store_ps(&fps[88], c12);

    for (int i = 0; i < 12; ++i) {
        for (int j = 0; j < 8; ++j) {
            accumulate[i] += fps[i * 8 + j];
        }
    }

    /* stores */
    matData[(row + 0) * rowSpan + col + 0] = accumulate[0];
    matData[(row + 0) * rowSpan + col + 1] = accumulate[1];
    matData[(row + 0) * rowSpan + col + 2] = accumulate[2];

    matData[(row + 1) * rowSpan + col + 0] = accumulate[3];
    matData[(row + 1) * rowSpan + col + 1] = accumulate[4];
    matData[(row + 1) * rowSpan + col + 2] = accumulate[5];

    matData[(row + 2) * rowSpan + col + 0] = accumulate[6];
    matData[(row + 2) * rowSpan + col + 1] = accumulate[7];
    matData[(row + 2) * rowSpan + col + 2] = accumulate[8];

    matData[(row + 3) * rowSpan + col + 0] = accumulate[9];
    matData[(row + 3) * rowSpan + col + 1] = accumulate[10];
    matData[(row + 3) * rowSpan + col + 2] = accumulate[11];
}

/* 
 * Compute L2Y x L2X sized blocks from the output matrix C.
 * In order to keep this code nice and hot in instruction cache,
 * keep it restricted to full blocks of L2X x L2Y.
 */
__declspec(noalias) void MMHelper_MultL2Blocks(float* __restrict const matData,
                                               const unsigned rowSpan, const Mat& matA,
                                               const Mat& matBT, const unsigned col,
                                               const unsigned row,
                                               const unsigned L2BlockX,
                                               const unsigned L2BlockY)
{
    /* multiply 4x3 blocks, L2blockX == 3*k, L2blockY == 4*m */
    for (int blockRow = row; blockRow < row + L2BlockY; blockRow += 4) {
        for (int blockCol = col; blockCol < col + L2BlockX; blockCol += 3) {
            MMHelper_Mult4x3Blocks(matData, rowSpan, matA, matBT, blockCol, blockRow);
        }
    }
}

/* Compute K x K sized blocks from the output matrix C. see struct mmBlockInfo */
__declspec(noalias) void MMHelper_MultFullBlocks(float* __restrict const matData,
                                                 const unsigned rowSpan,
                                                 const Mat& matA, const Mat& matBT,
                                                 const unsigned colC,
                                                 const unsigned rowC,
                                                 const MMBlockInfo& mmBlockInfo)
{
    const unsigned L2BlockX = mmBlockInfo.L2BlockX, L2BlockY = mmBlockInfo.L2BlockY,
                   L3BlockX = mmBlockInfo.L3BlockX, L3BlockY = mmBlockInfo.L3BlockY,
                   issuedBlockSzX = mmBlockInfo.issuedBlockSzX,
                   issuedBlockSzY = mmBlockInfo.issuedBlockSzY;

    /* try to prefetch next bit of block into memory while still handling this one */
    {
        if constexpr (doL3Prefetch) {
            std::unique_lock<std::mutex> lock(prefetchMutex);
            int alreadyPrefetchedCol =
              prefetched[rowC / L3BlockY][colC / issuedBlockSzX];
            lock.unlock();
            if (!alreadyPrefetchedCol) {
                for (int c = colC + issuedBlockSzX; c < colC + issuedBlockSzX; ++c) {
                    for (int pos = 0; pos < matA.rowSpan;
                         pos += cacheLineSz / sizeof(float)) {
                        _mm_prefetch((const char*)&matBT.mat[c * matBT.rowSpan + pos],
                                     _MM_HINT_T2);
                    }
                }
                lock.lock();
                prefetched[rowC / L3BlockY][colC / issuedBlockSzX]++;
                lock.unlock();
            }
        }
    }

    /* multiply L2YxL2X blocks */
    for (int blockColC = colC; blockColC < colC + issuedBlockSzX;
         blockColC += L2BlockX) {
        for (int blockRowC = rowC; blockRowC < rowC + issuedBlockSzY;
             blockRowC += L2BlockY) {
            MMHelper_MultL2Blocks(matData, rowSpan, matA, matBT, blockColC, blockRowC,
                                  L2BlockX, L2BlockY);
        }
    }
}

/* 
 * This function divides the matrix multiplication into segments and
 * issues commands for a cache aware thread pool to handle them.
 * Uses the helper functions above. 
 */
__declspec(noalias) const Mat MTMatMul(const Mat& matA, const Mat& matB)
{
    /* if CPU information is not already queried, do so */
    if (!CPUInfoQueried) {
        int dCaches[3];
        int iCache;

        CPUUtil::GetCacheInfo(&dCaches[0], iCache);

        L2Size = dCaches[1];
        L3Size = dCaches[2];

        cacheLineSz = CPUUtil::GetCacheLineSize();

        CPUInfoQueried++;
    }

    /* allocate the aligned float array for our new matrix C */
    float* __restrict const matData =
      (float*)_aligned_malloc(matA.height * matB.rowSpan * sizeof(float), AVX_ALIGN);

    /* construct matrix C */
    Mat matC{matB.width, matA.height, matB.rowSpan, matData};

    /* for the sake of cache, we'll be working with transposed B */
    const Mat matBT = TransposeMat(matB);

    /* initialize the HWLocalThreadPool with 1 or 2 threads per physical core
    * for all physical cores. Number of threads per core depends on HTT status. */
    const int HTTEnabled = CPUUtil::GetHTTStatus();
    const int jobStride = (1 << HTTEnabled);
    HWLocalThreadPool tp(0, jobStride);

    /* decide the block sizes for the given matrix and CPU */
    const float invN = 1.0 / matA.rowSpan;

    int QL2 = invN * L2Size / sizeof(float);
    int QL3 = invN * L3Size / sizeof(float);
    int k = min(max(QL2 / 6, 1), 10);
    int m = min(max(QL2 / 8, 1), 10);
    int L2BlockX = 3 * k;
    int L2BlockY = 4 * m;
    int lcmMN = std::lcm(k, m);
    int L3BlockX = min(max(QL3 / 120 / lcmMN * lcmMN * 60, 12*L2BlockX), 360);
    int L3BlockY = L3BlockX;
    int issuedBlockSzX = L3BlockX / 4;
    int issuedBlockSzY = L3BlockY / 3;

    /*printf("%d %d\n%d %d %d %d %d %d\n", matC.height, matC.width, L2BlockX, L2BlockY, issuedBlockSzX, issuedBlockSzY,
           L3BlockX, L3BlockY);*/

    MMBlockInfo mmBlockInfo{L3BlockX, L3BlockY,       L2BlockX,
                            L2BlockY, issuedBlockSzX, issuedBlockSzY};

    /* before we begin, start prefetching the first L3 level block */
    /* reset the prefetched flags */
    memset(&prefetched[0][0], 0, 1024 * 1024 * sizeof(int));
    /* prefetch rows of A and columns of B, one cache line at a time */
    for (int r = 0; r < L3BlockY; ++r) {
        for (int pos = 0; pos < matA.rowSpan; pos += cacheLineSz / sizeof(float)) {
            _mm_prefetch((const char*)&matA.mat[r * matA.rowSpan + pos], _MM_HINT_T2);
        }
    }
    for (int c = 0; c < L3BlockX; ++c) {
        for (int pos = 0; pos < matA.rowSpan; pos += cacheLineSz / sizeof(float)) {
            _mm_prefetch((const char*)&matBT.mat[c * matBT.rowSpan + pos], _MM_HINT_T2);
        }
    }
    /* prefetch is called for the first block, mark it. */
    prefetched[0][0]++;

    /* start issuing jobs for the thread pool */

    /*
     * We incorporate multiple levels of tiling into our traversal.
     *
     * If we issue commands linearly, we'll have poor L3 cache utilization.
     * [ [C0T0 | C0T1] [C1T0 | C1T1] ... [C5T0 | C5T1] ] covering a rows, b columns,
     * (a+b)N floats of data is needed to compute a*b sized block.
     * So, instead, we issue commands in the blocked manner, like:
     * [ [C0T0 | C0T1] [C1T0 | C1T1] 
     *   [C2T0 | C5T1] [C2T0 | C2T1] ] 
     *
     * Traverse L3 sized blocks, 
     * inside each, issue issuedBlockSz sized blocks.
     */

    int rowC = 0;
    /* handle L3Y sized rows
     * cast unsigned dimensions to signed to avoid UB */
    for (; rowC <= (int)matA.height - L3BlockY; rowC += L3BlockY) {
        int colC = 0;
        /* handle L3Y x L3X sized blocks */
        for (; colC <= (int)matB.width - L3BlockX; colC += L3BlockX) {
            /* Issue issuedBlockSzY x issuedBlockSzX sized blocks */
            for (int blockRowC = rowC; blockRowC < rowC + L3BlockY;
                 blockRowC += issuedBlockSzY) {
                for (int blockColC = colC; blockColC < colC + L3BlockX;
                     blockColC += jobStride * issuedBlockSzX) {
                    tp.Add({
                        HWLocalThreadPool::WrapFunc(MMHelper_MultFullBlocks, matData,
                                                    matB.rowSpan, matA, matBT, blockColC,
                                                    blockRowC, mmBlockInfo),
                        HWLocalThreadPool::WrapFunc(MMHelper_MultFullBlocks, matData, 
                                                    matB.rowSpan, matA, matBT,
                                                    blockColC + issuedBlockSzX, 
                                                    blockRowC, mmBlockInfo)
                        });
                }
            }
        }
        /* handle the block w < L3X, h = L3Y at the end of the row */
        if (matB.width > colC) {
            const unsigned remSubX = (matB.width - colC) >> HTTEnabled;
            tp.Add({
                HWLocalThreadPool::WrapFunc(MMHelper_MultAnyBlocks, matData,
                                            matB.rowSpan, matA, matBT, colC, rowC,
                                            remSubX, L3BlockY, mmBlockInfo),
                HWLocalThreadPool::WrapFunc(MMHelper_MultAnyBlocks, matData, 
                                            matB.rowSpan, matA, matBT,
                                            colC + remSubX, rowC, 
                                            matB.width - colC - remSubX, L3BlockY,
                                            mmBlockInfo)
                });
        }
    }
    /* handle last row, h < L3Y */
    int colC = 0;
    /* first handle blocks of w = L3X, h < L3Y */
    for (; colC <= (int)matB.width - L3BlockX; colC += jobStride * issuedBlockSzX) {
        tp.Add({
            HWLocalThreadPool::WrapFunc(MMHelper_MultAnyBlocks, matData, 
                                        matB.rowSpan, matA, matBT, colC,
                                        rowC, issuedBlockSzX, matA.height - rowC, 
                                        mmBlockInfo),
            HWLocalThreadPool::WrapFunc(MMHelper_MultAnyBlocks, matData,
                                        matB.rowSpan, matA, matBT,
                                        colC + issuedBlockSzX, rowC, issuedBlockSzX,
                                        matA.height - rowC, mmBlockInfo)});
    }
    /* now handle the rightmost block of w < L3X, h < L3Y */
    tp.Add({HWLocalThreadPool::WrapFunc(MMHelper_MultAnyBlocks, matData, matB.rowSpan,
                                        matA, matBT, colC, rowC, matB.width - colC,
                                        matA.height - rowC, mmBlockInfo),
        []() {}});

    /* -- commands issued -- */

    /* wait for the thread pool to finish */
    tp.Close();
    /* free the temporary bT matrix */
    _aligned_free(matBT.mat);

    return matC;
}

/* MatMul function, a simple branch that calls the proper implementation
 * based on the complexity of the input matrix. */
const Mat MatMul(const Mat& matA, const Mat& matB)
{
    /* 
     * If complexity is low enough,
     * use the single threaded, transposed B method.
     * A(N, M) B(M, K) => # of ops ~= 2*N*K*M 
     */
    if (matA.height * matA.width * matB.width < 350 * 350 * 350) {
        return ST_TransposedBMatMul(matA, matB);
    }
    return MTMatMul(matA, matB);
}

int __cdecl main(int argc, char* argv[])
{
    if (argc < 4) {
        std::cout << "No args\n";
        return 0;
    }

    /* make sure the runtime system supports AVX and FMA ISAs */
    assert(CPUUtil::GetSIMDSupport());

    const char* inputMtxAFile = argv[1];
    const char* inputMtxBFile = argv[2];
    const char* outMtxABFile = argv[3];

    //const char* inputMtxAFile = "matrixAx.bin";
    //const char* inputMtxBFile = "matrixBx.bin";
    //const char* outMtxABFile = "matrixAB-out.bin";

    const Mat inputMtxA = LoadMat(inputMtxAFile);
    const Mat inputMtxB = LoadMat(inputMtxBFile);

    /*printf("%d %d %d %d\n", inputMtxA.height, inputMtxA.width, inputMtxB.height,
           inputMtxB.width);*/

    auto start = std::chrono::high_resolution_clock::now();
    const Mat outMtxAB = MatMul(inputMtxA, inputMtxB);
    auto end = std::chrono::high_resolution_clock::now();

    std::cout
      << "Matrix Multiplication: "
      << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
      << " microseconds.\n";

    DumpMat(outMtxABFile, outMtxAB);

    FreeMat(inputMtxA);
    FreeMat(inputMtxB);
    FreeMat(outMtxAB);

    return 0;
}
