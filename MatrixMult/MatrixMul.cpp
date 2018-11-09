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
#include <thread>
#include "ThreadPool.h"

#define NOMINMAX

typedef struct Mat
{
    unsigned width;
    unsigned height;
    unsigned rowSpan;
    float *mat;
} Mat;

void FreeMat(const Mat &mat) {
    free(mat.mat);
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
    mat.mat = (float*)malloc(matSize);
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
const Mat TransposeMat(const Mat &mat) {
    const unsigned tRowSpan = RoundUpPwr2(mat.height, 64 / sizeof(float));
    float * const tData = (float*)malloc(mat.width*tRowSpan * sizeof(float));

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
    float * __restrict const matData = (float*)malloc(matA.height * matB.rowSpan * sizeof(float));

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
    float * __restrict const matData = (float*)malloc(matA.height * matB.rowSpan * sizeof(float));

    Mat matC{
        matB.width,
        matA.height,
        matB.rowSpan,
        matData
    };

    matC.rowSpan = matB.width;
    const Mat matBT = TransposeMat(matB);
    for (int rowC = 0; rowC < matA.height; ++rowC) {
        //if (rowC % 10 == 0)
        //    printf("row: %d of %d\n", rowC, matA.height);
        for (int colC = 0; colC < matB.width; ++colC) {
            float accumulate = 0;
            for (int pos=0; pos < matA.width; ++pos) {
                accumulate += matA.mat[rowC*matA.rowSpan + pos] * matBT.mat[colC*matBT.rowSpan + pos];
            }
            matData[rowC*matB.width + colC] = accumulate;
        }
    }

    free(matBT.mat);

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
    float * __restrict const matData = (float*)malloc(matA.height * matB.rowSpan * sizeof(float));

    Mat matC{
        matB.width,
        matA.height,
        matB.rowSpan,
        matData
    };

    const unsigned blockX = 16, blockY = 16;
    
    matC.rowSpan = matB.width;
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
                    matData[r*matB.width + c] = accumulate;
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
                matData[r*matB.width + c] = accumulate;
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
            matData[rowC*matB.width + colC] = accumulate;
        }
    }

    free(matBT.mat);
    
    return matC;
}

void MMHelper_MultBlocks(float* matData, const unsigned blockX, const unsigned blockY,
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

            matData[r*matB.width + c] = accumulate;
            //printf("%d %d, %f\n", r, c, accumulate);
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
const Mat MTMatMul(const Mat& matA, const Mat& matB) {
    /*
    * Now, let's parallelize the code!
    * We're already doing block by block multiplication, which are independent of each other.
    * I'm on an i7 system with HT, so I will assign 2 threads to process a block so they operate on same cache.
    *
    * I will use a thread-couple pool to process blocks.
    * Read HWLocalThreadPool.h for more details.
    */

    /* First, we need to query Win32 API to lookup the mapping between logical processors and physical cores. */

    float * __restrict const matData = (float*)malloc(matA.height * matB.rowSpan * sizeof(float));

    Mat matC{
        matB.width,
        matA.height,
        matB.rowSpan,
        matData
    };

    const unsigned blockX = 32, blockY = 32;
    const unsigned subX = blockX >> 1;

    matC.rowSpan = matB.width;
    const Mat matBT = TransposeMat(matB);

    HWLocalThreadPool<6, 2> tp;

    int rowC = 0;
    for (; rowC < matA.height - blockY; rowC += blockY) {
        int colC = 0;
        /* Process BlockX X BlockY blocks */
        for (; colC < matB.width - blockX; colC += blockX) {
            tp.Add({
                HWLocalThreadPool<>::WrapFunc(MMHelper_MultBlocks,
                    matData, subX, blockY, rowC, colC, matA, matB, matBT),
                HWLocalThreadPool<>::WrapFunc(MMHelper_MultBlocks,
                    matData, subX, blockY, rowC, colC + subX, matA, matB, matBT)                
            });
        }
        /* Process remainings at the end of the row, width < blockX */
        tp.Add({
            HWLocalThreadPool<>::WrapFunc(MMHelper_MultBlocks,
                matData, matB.width - colC, blockY, rowC, colC, matA, matB, matBT),
            []() {}
        });
    }

    /* Process last row, height < blockY, col+=blockX */
    int colC = 0;
    for (; colC < matB.width - blockX; colC += blockX) {
        tp.Add({
            HWLocalThreadPool<>::WrapFunc(MMHelper_MultBlocks,
                matData, subX, matA.height - rowC, rowC, colC, matA, matB, matBT) ,
            HWLocalThreadPool<>::WrapFunc(MMHelper_MultBlocks,
                matData, subX, matA.height - rowC, rowC, colC + subX, matA, matB, matBT)
        });
    }

    /* Process bottom right block, h < bY, w < bX */
    tp.Add({
            HWLocalThreadPool<>::WrapFunc(MMHelper_MultBlocks,
                matData, matB.width - colC, matA.height - rowC, rowC, colC, matA, matB, matBT),
            []() {}
        });

    std::cout << "Queued!\n";
    tp.Close();
    free(matBT.mat);
    std::cout << "Done!\n";

    return matC;
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

    const Mat outMtxAB = MTMatMul(inputMtxA, inputMtxB);

    //DumpMat(outMtxABFile, outMtxAB);

    const Mat outMtxAB2 = MTMatMul(inputMtxA, inputMtxB);

    while (1) {
        std::cout << " ";
    }





    return 0;
}
