#pragma once
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include <cassert>
#include <intrin.h>

namespace CPUUtil
{
    /* Utility, convert given bitmask to const char* */
    const char* BitmaskToStr(WORD bitmask)
    {
        const unsigned N = sizeof(WORD) * 8;
        char* const str = new char[N + 1];
        str[N] = 0;
        for (int i = 0; i < N; ++i) {
            str[N - i - 1] = '0' + ((bitmask)&1);
            bitmask >>= 1;
        }
        return str;
    }

    /* Get number of physical processors on the system */
    int GetNumHWCores();

    /* Get the logical processor mask corresponding to the Nth hardware core */
    int GetProcessorMask(unsigned n, ULONG_PTR& mask);

    /* Fill dCaches with L1,2,3 data cache sizes, 
     * and iCache with L1 dedicated instruction cache size. */
    void GetCacheInfo(int* dCaches, int& iCache);

    /* Query cache line size on the current system. */
    int GetCacheLineSize();

}; // namespace CPUUtil
