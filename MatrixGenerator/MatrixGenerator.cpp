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
#include <fstream>
#include <cstdint>
#include <random>
#include <functional>
#include <iostream>
#include <chrono>
#include <cstdlib>
#include <cassert>

#define AVX_ALIGN 32

typedef struct Mat
{
	unsigned width;
	unsigned height;
	unsigned rowSpan;
	float *mat;
} Mat;

template<typename Rand>
static void RandInitMat(Mat *m, Rand &r)
{
	for(unsigned y=0; y<m->height; ++y)
		for(unsigned x=0; x<m->width; ++x)
			m->mat[y*m->rowSpan + x] = r();
}

const Mat LoadMat(const char * const filename) {
    Mat mat;
    uint32_t matSize;

    std::ifstream in(filename, std::ios::binary | std::ios::in);

    if (!in.is_open()) {
        std::cerr << "Err loading!\n";
        return {};
    }

    in.read((char*)&mat, 3 * sizeof(uint32_t));
    in.read((char*)&matSize, sizeof(uint32_t));
    in.seekg(12*sizeof(uint32_t), std::ios::cur);
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


/* Single threaded, do i need to multithread this as well? 
Honestly, I don't think it will have any significant effect. n^2 vs n^3 */
__declspec(noalias) const Mat TransposeMat(const Mat& mat)
{
    const unsigned tRowSpan = RoundUpPwr2(mat.height, 64 / sizeof(float));
    float* __restrict const tData =
        (float*)_aligned_malloc(mat.width * tRowSpan * sizeof(float), AVX_ALIGN);

    Mat T{ mat.height, mat.width, tRowSpan, tData };

    // hah, the loops are truly interchangable as we encounter a cache miss either ways
    for (int rowT = 0; rowT < T.height; ++rowT) {
        for (int colT = 0; colT < T.width; ++colT) {
            tData[rowT * tRowSpan + colT] = mat.mat[colT * mat.rowSpan + rowT];
        }
    }

    return T;
}

const Mat ST_TransposedBMatMul(const Mat& matA, const Mat& matB)
{
    /* Now, I thought transposing B and then traversing it row order would help and it does!
    * Also, note that, if we manually unrolled the loop here, compiler wouldn't vectorize the loop for some reason
    * (1301: Loop stride is not +1.) is the exact compiler message. */
    float* __restrict const matData =
        (float*)_aligned_malloc(matA.height * matB.rowSpan * sizeof(float), AVX_ALIGN);

    Mat matC{ matB.width, matA.height, matB.rowSpan, matData };

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

int _cdecl main(int argc, char *argv[])
{
	static const unsigned ALIGN = 64;
	static const unsigned FLT_ALIGN = ALIGN / sizeof(float);

	std::random_device rd;
	std::uniform_real_distribution<float> matValDist(-50.0f, 50.0f);
	auto matRand = std::bind(matValDist, std::ref(rd));
	Mat a, b;
    std::string suffix;

    if (argc == 1) {
        /* randomly generated */
	    std::uniform_int_distribution<unsigned> matSizeDist(100, 1000);
	    auto sizeRand = std::bind(matSizeDist, std::ref(rd));
        a.width = sizeRand();
	    a.height = sizeRand();
	    a.rowSpan = RoundUpPwr2(a.width, FLT_ALIGN);

        b.width = sizeRand();
	    b.height = a.width;

        suffix = "";
    }
    else if (argc == 2) {
        /* 2 NxN */
        const int N = atoi(argv[1]);
        assert(N > 0);
        a.width = N;
        a.height = N;
        b.width = N;
        b.height = N;

        suffix = "";
    }
    else if (argc == 3) {
        /* 2 NxN */
        const int N = atoi(argv[1]);
        assert(N > 0);
        a.width = N;
        a.height = N;
        b.width = N;
        b.height= N;
        
        suffix = std::string(argv[2]);
    }
    else if (argc == 4) {
        /* NxM, MxN */
        const int N = atoi(argv[1]);
        const int M = atoi(argv[2]);
        assert(N > 0 && M > 0);
        a.width = M;
        a.height = N;
        b.width = N;
        b.height = M;

        suffix = std::string(argv[3]);
    }
    else if (argc == 5) {
        /* NxM, MxK */
        const int N = atoi(argv[1]);
        const int M = atoi(argv[2]);
        const int K = atoi(argv[3]);
        assert(N > 0 && M > 0);
        a.width = M;
        a.height = N;
        b.width = K;
        b.height = M;    

        suffix = std::string(argv[4]);
    }
    else {
        std::cerr << "Invalid arguments!\n";
        return 2;
    }


    a.rowSpan = RoundUpPwr2(a.width, FLT_ALIGN);
    b.rowSpan = RoundUpPwr2(b.width, FLT_ALIGN);

	a.mat = new float[a.rowSpan*a.height];
	b.mat = new float[b.rowSpan*b.height];
	
	RandInitMat(&a, matRand);
	RandInitMat(&b, matRand);

    printf("a: [%d %d] | b: [%d %d]\n", a.width, a.height, b.width, b.height);

    auto start = std::chrono::high_resolution_clock::now();
    const Mat c = ST_TransposedBMatMul(a, b);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Generation w/ tranposed mult. took: " 
        << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() 
        << " microseconds.\n";

    DumpMat(("matrixA" + suffix + ".bin").c_str(), a);
    DumpMat(("matrixB" + suffix + ".bin").c_str(), b);
    DumpMat(("matrixAB" + suffix + ".bin").c_str(), c);

	delete[] a.mat;
	delete[] b.mat;
    _aligned_free(c.mat);

	return 0;
}
