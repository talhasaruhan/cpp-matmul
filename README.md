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

# Benchmarks

On my machine (6 core i7-8700K), I’ve compared my implementation against:
* Multithreaded python-numpy which uses C/C++ backend and Intel MKL BLAS
library. 
* Eigen library (with all the compiler optimizations turned on)

All example setups are for multiplying two 10,000 X 10,000 matrices, more tests with differently sized matrices can be found at [comparison](#comparison) section.
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

These results are averages of 10-100 runs depending on how small the matrices and how variant the figures are. Among all the benchmarks, numpy was by far the most consistent run to run. Least consistent was the Eigen, showing unreasonably high variance at smaller matrices. 

Benchmark | Numpy(MKL)    | Eigen          | This impl. (ST_TransposedBMatMul) | This impl. (MTMatMul) |
| ------------- | ------------- | ------------- | ------------- | ------------- |
(10Kx10K)(10Kx10K) | 8.88s  | 20.33s  |  161s | 18.86s  |
(5Kx5K)(5Kx5K) | 1.01s  | 2.58s  |  21.3s | 2.15s  |
(1Kx1K)(1Kx1K) | 10.0ms  | 28.7ms  |  97.5ms | 20.7ms  |
(500x500)(500x500) | 2.0ms  | 11.0ms  |  12.8ms | 6.5ms  |
(100x100)(100x100) | 1.0ms  | 400us-2.0ms  |  150us | 3.5ms  |


My multithreaded implementation is only 2.1 times slower than a
professional BLAS package (18.9 seconds vs 8.9 seconds) and is even slightly faster than the Eigen library. 

If I had implemented Strassen’s algorithm, for the 10K case, we could naively expect the program
to run (10^4)^(3-2.8) = 6.3 times faster. Obviously
Strassen’s constant for big O notation is much larger, so in real life the performance benefit would likely to be far less than that. But at the end, I think it’s safe to
assume that Strassen's would still improve the overall performance to a level more comparable with numpy for larger matrices (>10K?).

# Code details:

## Functions for multiplying matrices

``` c++
const Mat TransposeMat(const Mat& mat)
const Mat ST_NaiveMatMul(const Mat& matA, const Mat& matB)
const Mat ST_TransposedBMatMul(const Mat& matA, const Mat& matB)
const Mat ST_BlockMult(const Mat& matA, const Mat& matB)
const Mat MTMatMul(const Mat& matA, const Mat& matB) 

/* Selects between MTMatMul, ST_TransposedBMatMul */
const Mat MatMul(const Mat& matA, const Mat& matB)
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

![no_leaks_f2](https://user-images.githubusercontent.com/15991519/48242727-a0d70300-e3ed-11e8-80e9-01954f2ec6b9.PNG)

(This is  the heap profile of the program after running C1 = AB, freeing C1, then running C2=AB and freeing C2. As can be seen here, all the previously leaked mess (packed tasks, function pointers, CoreHandler member arrays etc. ) is now cleaned up nicely. Note: int[] is the static CPU core to logical processor map,)

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
* Added Eigen benchmarks
* Implemented MatMul which should be the general function exposed to outside. It simply selects betwen *MTMatMul* and *ST_TransposedBMatMul* depending on the sizes of the matrices. Current impl.: ```A.height*A.width*A.width*B.width < K : ST_TransposedBMatMul o.w : MTMatMul```
