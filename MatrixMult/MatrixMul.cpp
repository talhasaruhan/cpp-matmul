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
constexpr unsigned L2BlockX = 4, L2BlockY = 4;
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

__declspec(noalias)
void MMHelper_MultBlocks(float* __restrict const matData, const unsigned blockX, const unsigned blockY,
    const unsigned rowC, const unsigned colC,
    const Mat& matA, const Mat& matB, const Mat& matBT)
{
    __declspec(align(32)) float fps[8 * 8];
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

    /* 
    * assume L2BlockX = 4, L2BlockY % 2 == 0
    */
    for (int blockRowC = rowC; blockRowC < rowC + L3BlockY; blockRowC += L2BlockY) {
        for (int blockColC = colC; blockColC < colC + (L3BlockX >> 1); blockColC += L2BlockX) {
            for (int blockRow = 0; blockRow < L2BlockY; blockRow+=2) {
                const unsigned matAoffset1 = (blockRowC + blockRow + 0) * matA.rowSpan;
                const unsigned matAoffset2 = (blockRowC + blockRow + 1) * matA.rowSpan;
                const unsigned matBToffset1 = (blockColC + 0) * matBT.rowSpan;
                const unsigned matBToffset2 = (blockColC + 1) * matBT.rowSpan;
                const unsigned matBToffset3 = (blockColC + 2) * matBT.rowSpan;
                const unsigned matBToffset4 = (blockColC + 3) * matBT.rowSpan;

                __m256 a1, a2, a3, a4, b1, b2, b3, b4;
                __m256 c1 = _mm256_setzero_ps();
                __m256 c2 = _mm256_setzero_ps();
                __m256 c3 = _mm256_setzero_ps();
                __m256 c4 = _mm256_setzero_ps();
                __m256 c5 = _mm256_setzero_ps();
                __m256 c6 = _mm256_setzero_ps();
                __m256 c7 = _mm256_setzero_ps();
                __m256 c8 = _mm256_setzero_ps();
                
                for (int pos = 0; pos < matA.width; pos += 8) {
                    a1 = _mm256_load_ps(&matAmat[matAoffset1 + pos]);
                    a2 = _mm256_load_ps(&matAmat[matAoffset2 + pos]);

                    b1 = _mm256_load_ps(&matBTmat[matBToffset1 + pos]);
                    b2 = _mm256_load_ps(&matBTmat[matBToffset2 + pos]);
                    b3 = _mm256_load_ps(&matBTmat[matBToffset3 + pos]);
                    b4 = _mm256_load_ps(&matBTmat[matBToffset4 + pos]);

                    c1 = _mm256_fmadd_ps(a1, b1, c1);
                    c2 = _mm256_fmadd_ps(a1, b2, c2);
                    c3 = _mm256_fmadd_ps(a1, b3, c3);
                    c4 = _mm256_fmadd_ps(a1, b4, c4);

                    c5 = _mm256_fmadd_ps(a2, b1, c5);
                    c6 = _mm256_fmadd_ps(a2, b2, c6);
                    c7 = _mm256_fmadd_ps(a2, b3, c7);
                    c8 = _mm256_fmadd_ps(a2, b4, c8);
                }

                /* horizontal sum */

                float accumulate[8]; 
                memset(&accumulate[0], 0, 8*sizeof(float));

                _mm256_store_ps(&fps[0], c1);
                _mm256_store_ps(&fps[8], c2);
                _mm256_store_ps(&fps[16], c3);
                _mm256_store_ps(&fps[24], c4);
                _mm256_store_ps(&fps[32], c5);
                _mm256_store_ps(&fps[40], c6);
                _mm256_store_ps(&fps[48], c7);
                _mm256_store_ps(&fps[56], c8);

                for (int i = 0; i < 8; ++i) {
                    for (int j = 0; j < 8; ++j) {
                        accumulate[i] += fps[i*8+j];
                    }
                }

                matData[(blockRowC + blockRow + 0)*matB.rowSpan + blockColC + 0] = accumulate[0];
                matData[(blockRowC + blockRow + 0)*matB.rowSpan + blockColC + 1] = accumulate[1];
                matData[(blockRowC + blockRow + 0)*matB.rowSpan + blockColC + 2] = accumulate[2];
                matData[(blockRowC + blockRow + 0)*matB.rowSpan + blockColC + 3] = accumulate[3];

                matData[(blockRowC + blockRow + 1)*matB.rowSpan + blockColC + 0] = accumulate[4];
                matData[(blockRowC + blockRow + 1)*matB.rowSpan + blockColC + 1] = accumulate[5];
                matData[(blockRowC + blockRow + 1)*matB.rowSpan + blockColC + 2] = accumulate[6];
                matData[(blockRowC + blockRow + 1)*matB.rowSpan + blockColC + 3] = accumulate[7];
            }
        }
    }
}

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
