#pragma once
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include <cassert>
#include <intrin.h>

namespace CPUUtil
{
    /* Utility, convert given bitmask to const char* */
    const char* BitmaskToStr(WORD bitmask);

    /* Get number of physical processors on the runtime system */
    int GetNumHWCores();

    /* Get number of logical processors on the runtime system */
    int GetNumLogicalProcessors();

    /* Get the logical processor mask corresponding to the Nth hardware core */
    int GetProcessorMask(unsigned n, ULONG_PTR& mask);

    /* Fill dCaches with L1,2,3 data cache sizes, 
     * and iCache with L1 dedicated instruction cache size. */
    void GetCacheInfo(int* dCaches, int& iCache);

    /* Query cache line size on the current system. */
    int GetCacheLineSize();

    /* Query whether or not the runtime system supports HTT */
    int GetHTTStatus();

    /* Query if the runtime system supports AVX and FMA instruction sets. */
    int GetSIMDSupport();

}; // namespace CPUUtil
