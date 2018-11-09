# Project

[See CHANGELOG](#changelog)

In this project, I’ve implemented multiple methods for multiplying
matrices, and relevant utilities. My prime focuses were:

  - Cache locality, memory access patterns.

  - SIMD, ways to help compiler generate vectorized code without using
    intrinsics

  - Cache friendly multithreading

I didn’t implement the Strassen’s algorithm, but may do so later on.

# How to run

Note: The code is currently Win32 only (due to the calls to query CPU resources and to set processor affinity). 

Build the solution, then navigate to *x64\\Release\\* and run this command or call “run.bat”. If
you don’t have “tee” command, just delete the last part or install
GnuWin32 CoreUtils.

``` bash
for /l %x in (1, 1, 100) do echo %x && (MatrixGenerator.exe && printf "Generated valid output. Testing...\n" && MatrixMult.exe matrixA.bin matrixB.bin matrixAB-out.bin && printf \n\n ) | tee -a out.txt
```

# Benchmarks for large matrices

On my machine (6 core i7-8700K), I’ve compared my implementation against:
* Multithreaded python-numpy which uses C/C++ backend and Intel MKL BLAS
library. 
* Eigen library (with all the compiler optimizations turned on)

All benchmarks are for multiplying two 10,000 X 10,000 matrices.
Here are the benchmarks in a couple of lines, the full source code for tests can be found under Benchmark folder

### Numpy with C++/MKL backend
```
    >>> import numpy as np
    >>> a = np.random.randn(10000, 10000)
    >>> import time
    >>> start = time.time(); b=np.dot(a, a); end=time.time();
    >>> end-start
    8.877262115478516
```
### Eigen (O2, OMP, Opar, AVX2, fp:fast etc. fully optimized)
Setup:
```
    MatrixXd matA = MatrixXd::Random(10000, 10000);
    MatrixXd matB = MatrixXd::Random(10000, 10000);
    
    /* I found that 12 threads to work best (tried 4, 8, 12, 16), 
    which is expected as any less and cpu resources will be free, 
    any more and threads will compete with each other for no reason. */
    
    setNbThreads(12); 
    
    auto start = std::chrono::high_resolution_clock::now();
    MatrixXd matC = matA * matB;
    auto end = std::chrono::high_resolution_clock::now();

    std::cout << "Matrix Multiplication: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " microseconds.\n";
```
Output:
```
    Matrix Multiplication: 20327617 microseconds.
```
###  My implementation
Setup:
```
    /*  input matrices of 10Kx10K are generated beforehand using MatrixGenerator.exe */
    const Mat inputMtxA = LoadMat(inputMtxAFile);
    const Mat inputMtxB = LoadMat(inputMtxBFile);

    auto start = std::chrono::high_resolution_clock::now();
    const Mat outMtxAB = MTMatMul(inputMtxA, inputMtxB);
    auto end = std::chrono::high_resolution_clock::now();

    std::cout << "Matrix Multiplication: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " microseconds.\n";
```
Output:
```
    MatrixGenerator.exe && MatrixMult.exe matrixA.bin matrixB.bin matrixAB-out.bin
    a: [10000 10000] | b: [10000 10000]
    Generation w/ tranposed mult. took: 161384008 microseconds.
    generated
    Queued!
    Done!
    Matrix Multiplication: 18858824 microseconds.
    Correct.
```

### Comparison
Benchmark | Numpy(MKL)    | Eigen          | This impl. |
| ------------- | ------------- | ------------- | ------------- | 
(10Kx10K)(10Kx10K) | 8.88s  | 20.33s  | 18.86s  |

My multithreaded implementation is only 2.1 times slower than a
professional BLAS package (18.9 seconds vs 8.9 seconds) and is even slightly faster than the Eigen library. If I’d
implemented Strassen’s algorithm, assuming same constants, the program
would run (10^4)^(3-2.8) = 6.3 times faster. Obviously
Strassen’s constant is perceptibly larger, but I think it’s safe to
assume it would improve the overall performance to a level more comparable with numpy.

# Code details:

## Functions for multiplying matrices

``` c++
const Mat TransposeMat(const Mat& mat)
const Mat ST_NaiveMatMul(const Mat& matA, const Mat& matB)
const Mat ST_TransposedBMatMul(const Mat& matA, const Mat& matB)
const Mat ST_BlockMult(const Mat& matA, const Mat& matB)
const Mat MTMatMul(const Mat& matA, const Mat& matB) 
```

I’ve tried to address vectorization and cache locality in every
function, even algorithm wise naïve ones. For more details, each
function has a block of comment that explains how it’s designed and why
so.

## Multithreading utilities ([Repo](https://github.com/talhasaruhan/hwlocalthreadpool))

``` c++
Namespace QueryHWCores,
HWLocalThreadPool<NumOfCoresToUse, NumThreadsPerCore>   
```

I’ve also implemented a hardware local thread pool for multithreaded
*MTMatMul* function. The pool runs every thread corresponding to a job
on the same physical core. Idea is that, on hyperthreaded systems such
as mine, 2 threads that work on contiguous parts of memory should live
on the same core and share the same L1 and L2 cache.

  - Each job is described as an array of N functions. (N=2)

  - For each job, N threads (that were already created) are assigned respective
    functions.

  - For a given job, all threads are guaranteed to be on the same
    physical core.

  - No two threads from different jobs are allowed on the same physical
    core.

Basically, when traversing AB matrix in a block by block fashion, we
split each block in two and create two threads on a single physical
core, each handling half of the block.

## MSVC2017 Build options (over default x64 Release build settings)

  - Maximum optimization: /O2

  - Favor fast code /Ot

  - Enable function level linking: /Gy

  - Enable enhanced instruction set: /arch:AVX2

  - Floating point model: /fp:fast

  - Language: /std:c++17 (for several “if constexpr”s. otherwise can be
    compiled with C++ 11)
    
# Changelog

**Note:** Debugging builds will have arguments pre-set on the MatrixMul.cpp, you can ignore or revert those to accept argument from command line.

* 09/11/2018
* Fixed memory leaks

![no_leaks_f](https://user-images.githubusercontent.com/15991519/48237828-96127300-e3d9-11e8-9596-10e03797fc43.PNG)
(This is  the heap profile of the program after running C = AB, as can be seen here, all the previously leaked mess is now cleaned up nicely. Note: int[] is the CPU core to logical processor map,)

* Properly called destructors where CoreHandler objects are created using placement new into a malloc'ed buffer.
* Freed BT.mat (transpose of B) in the methods that use it to convert the problem into row-row dot product.
* ~~Changed Add function s.t it accepts std::shared_ptr<std::function<void()>[]>, this is only temporary.~~
* **Changed the Add() semantics**, now Add function accepts a std::vector<std::function<void()>>. Preferred way of using Add() function now is with initializer lists:

```
tp.Add({
    HWLocalThreadPool<>::WrapFunc(MMHelper_MultBlocks,
        matData, subX, matA.height - rowC, rowC, colC, matA, matB, matBT) ,
    HWLocalThreadPool<>::WrapFunc(MMHelper_MultBlocks,
        matData, subX, matA.height - rowC, rowC, colC + subX, matA, matB, matBT)
});
```
