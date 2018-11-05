#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include <chrono>
#include <sstream>
#include <iostream>
#include <fstream>

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
        std::cerr << "Err loading!\n";
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

static void PrintMat(const Mat &mat) {
    for (int i = 0; i < mat.height; i++) {
        for (int j = 0; j < mat.width; ++j) {
            printf("%f ", mat.mat[i*mat.rowSpan + j]);
        }
        printf("\n");
    }
}

int __cdecl main(int argc, char *argv[])
{
	if ( argc < 6 )
	{
		std::cout << "Matrix Multiplication Speed Tester:" << std::endl;
		std::cout << "Usage: multester.exe <known valid matrix file> <your program> <input A matrix> <input B matrix> <output AB matrix>" << std::endl;
		return 0;
	}

    static const int T = 100;

	static const float EPSILON = 1.0f;

	const char * const validMtxABFile = argv[1];
	const char * const testProgFile   = argv[2];
	const char * const inputMtxAFile  = argv[3];
	const char * const inputMtxBFile  = argv[4];
	const char * const outMtxABFile   = argv[5];
    STARTUPINFO si;
    PROCESS_INFORMATION proc;

    std::ostringstream cmdLineBuilder;
    std::string cmdLine;

    const Mat validAB = LoadMat(validMtxABFile);

    cmdLineBuilder.clear();
    cmdLineBuilder << "\"" << testProgFile << "\" \"" << inputMtxAFile << "\" \"" << inputMtxBFile << "\" \"" << outMtxABFile << "\"";
    cmdLine = cmdLineBuilder.str();

    std::memset(&si, 0, sizeof(si));
    si.cb = sizeof(si);

    auto start = std::chrono::high_resolution_clock::now();
    CreateProcessA(nullptr, const_cast<char*>(cmdLine.c_str()), nullptr, nullptr, false, 0, nullptr, nullptr, &si, &proc);
    WaitForSingleObject(proc.hProcess, INFINITE);
    auto end = std::chrono::high_resolution_clock::now();

    CloseHandle(proc.hThread);
    CloseHandle(proc.hProcess);
        
    std::cout << "Runtime: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " microseconds.\n";

    const Mat testAB = LoadMat(outMtxABFile);

    if ((testAB.width != validAB.width) || (testAB.height != validAB.height) || (testAB.rowSpan < testAB.width))
    {
        std::cout << "Bad size.\n";
        return -1;
    }

    for (unsigned row = 0; row < validAB.height; ++row)
    {
        for (unsigned col = 0; col < validAB.width; ++col)
        {
            const float valid = validAB.mat[row*validAB.rowSpan + col];
            const float test = testAB.mat[row*testAB.rowSpan + col];
            const float diff = std::fabs(valid - test);

            if (diff > EPSILON)
            {
                std::cout << "Variance greater than epsilon " << EPSILON << " at (" << col << ", " << row << "). Is " << test << ". Should be " << valid << ".\n";
                return -2;
            }
        }
    }

    std::cout << "Correct." << std::endl;

    FreeMat(testAB);
    FreeMat(validAB);

	return 0;
}
