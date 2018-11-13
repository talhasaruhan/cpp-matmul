//#include <iostream>
//#include <string.h>
//#include <thread>
//#include <chrono>
//#include <random>
//#include <sstream>
//
//#define EIGEN_USE_MKL_ALL
//#include <Eigen\Dense>
//
//using namespace std;
//using namespace Eigen;
//
//
//int main() {
//
//    const unsigned n = 10000;
//
//    mkl_set_num_threads(12);
//
//    MatrixXd matA = MatrixXd::Random(n, n);
//    MatrixXd matB = MatrixXd::Random(n, n);
//
//    setNbThreads(12);
//
//    auto start = std::chrono::high_resolution_clock::now();
//
//    MatrixXd matC = matA * matB;
//
//    auto end = std::chrono::high_resolution_clock::now();
//
//    std::cout << "Matrix Multiplication: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " microseconds.\n";
//
//
//    while (1) {}
//}