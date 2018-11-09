#include <iostream>
#include <string.h>
#include <thread>
#include <chrono>
#include <random>
#include <sstream>
#include <Eigen\Dense>

using namespace std;
using namespace Eigen;

int main() {

    const unsigned n = 5000;

    MatrixXd matA = MatrixXd::Random(n, n);
    MatrixXd matB = MatrixXd::Random(n, n);

    setNbThreads(12);

    auto start = std::chrono::high_resolution_clock::now();

    MatrixXd matC = matA * matB;

    auto end = std::chrono::high_resolution_clock::now();

    std::cout << "Matrix Multiplication: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " microseconds.\n";


    while (1) {}
}