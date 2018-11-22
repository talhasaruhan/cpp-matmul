#include <iostream>
#include <string.h>
#include <thread>
#include <chrono>
#include <random>
#include <sstream>

#define EIGEN_USE_MKL_ALL
#include <Eigen\Dense>

using namespace std;
using namespace Eigen;

int main(int argc, char* argv[])
{
    int K;
    if (argc == 1) {
        K = 5000;
    } else if (argc == 2) {
        /* 2 NxN */
        K = atoi(argv[1]);
        assert(K > 0);
    }

    mkl_set_num_threads(12);
    setNbThreads(12);

    MatrixXd matA = MatrixXd::Random(K, K);
    MatrixXd matB = MatrixXd::Random(K, K);

    auto start = std::chrono::high_resolution_clock::now();
    MatrixXd matC = matA * matB;
    auto end = std::chrono::high_resolution_clock::now();

    std::cout
        << "Matrix Multiplication: "
        << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
        << " microseconds.\n";
}
