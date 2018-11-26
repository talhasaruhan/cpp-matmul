#include "CPUUtil.h"
#include <cstdio>
#include <cstdint>

namespace CPUUtil
{
    namespace
    {
        static int logicalProcInfoCached = 0;
        static unsigned numHWCores, numLogicalProcessors;
        static ULONG_PTR* physLogicalProcessorMap = NULL;

        void PrintSysLPInfoArr(_SYSTEM_LOGICAL_PROCESSOR_INFORMATION* const sysLPInf,
                               const DWORD& retLen)
        {
            unsigned numPhysicalCores = 0;
            for (int i = 0; i * sizeof(_SYSTEM_LOGICAL_PROCESSOR_INFORMATION) <= retLen;
                 ++i) {
                if (sysLPInf[i].Relationship != RelationProcessorCore)
                    continue;

                printf(
                  "PHYSICAL CPU[%d]\n\t_SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX:\n",
                  numPhysicalCores);
                printf("\t\tProcessorMask:%s\n",
                       BitmaskToStr(sysLPInf[i].ProcessorMask));
                printf("\t\tRelationship:%u | RelationProcessorCore\n",
                       (uint8_t)sysLPInf[i].Relationship);
                printf("\t\tProcessorCore:\n");
                printf("\t\t\tFlags(HT?):%d\n",
                       (uint8_t)sysLPInf[i].ProcessorCore.Flags);
                ++numPhysicalCores;
            }
        }

        int TestPrintCPUCores()
        {
            const unsigned N = 30;
            _SYSTEM_LOGICAL_PROCESSOR_INFORMATION sysLPInf[N];
            DWORD retLen = N * sizeof(_SYSTEM_LOGICAL_PROCESSOR_INFORMATION);
            LOGICAL_PROCESSOR_RELATIONSHIP lpRel = RelationProcessorCore;

            BOOL retCode = GetLogicalProcessorInformation(&sysLPInf[0], &retLen);

            if (!retCode) {
                DWORD errCode = GetLastError();
                printf("ERR: %d\n", errCode);
                if (errCode == ERROR_INSUFFICIENT_BUFFER) {
                    printf("Buffer is not large enough! Buffer length required: %d\n",
                           retLen);
                } else {
                    printf("CHECK MSDN SYSTEM ERROR CODES LIST.\n");
                }
                return errCode;
            }

            PrintSysLPInfoArr(sysLPInf, retLen);

            return 0;
        }

        template <typename T>
        int NumSetBits(T n) {
            int count = 0;
            while (n) {
                count += (n & 1) > 0 ? 1 : 0;
                n >>= 1;
            }
            return count;
        }

        DWORD _GetSysLPMap(unsigned& numHWCores)
        {
            // These assumptions should never fail on desktop
            const unsigned N = 48, M = 48;

            _SYSTEM_LOGICAL_PROCESSOR_INFORMATION sysLPInf[N];
            DWORD retLen = N * sizeof(_SYSTEM_LOGICAL_PROCESSOR_INFORMATION);
            LOGICAL_PROCESSOR_RELATIONSHIP lpRel = RelationProcessorCore;

            static BOOL retCode = GetLogicalProcessorInformation(&sysLPInf[0], &retLen);

            if (!retCode) {
                return GetLastError();
            }

            ULONG_PTR* const lMap = (ULONG_PTR*)malloc(M * sizeof(ULONG_PTR));

            numHWCores = 0;
            for (int i = 0; i * sizeof(_SYSTEM_LOGICAL_PROCESSOR_INFORMATION) <= retLen;
                 ++i) {
                if (sysLPInf[i].Relationship != RelationProcessorCore)
                    continue;

                ULONG_PTR logicalProcessorMask = sysLPInf[i].ProcessorMask;
                lMap[numHWCores++] = logicalProcessorMask;
                numLogicalProcessors += NumSetBits(logicalProcessorMask);
            }

            physLogicalProcessorMap = (ULONG_PTR*)malloc(numHWCores * sizeof(ULONG_PTR));
            memcpy(physLogicalProcessorMap, lMap, numHWCores * sizeof(ULONG_PTR));
            free(lMap);

            return 0;
        }
    } // private namespace

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

    int GetNumHWCores()
    {
        if (!logicalProcInfoCached) {
            DWORD retCode = _GetSysLPMap(numHWCores);
            if (!retCode)
                logicalProcInfoCached = 1;
            else
                return -1;
        }
        return numHWCores;
    }

    int GetNumLogicalProcessors() {
        if (!logicalProcInfoCached) {
            DWORD retCode = _GetSysLPMap(numHWCores);
            if (!retCode)
                logicalProcInfoCached = 1;
            else
                return -1;
        }
        return numLogicalProcessors;
    }

    int GetProcessorMask(unsigned n, ULONG_PTR& mask)
    {
        if (!logicalProcInfoCached) {
            DWORD retCode = _GetSysLPMap(numHWCores);
            if (!retCode)
                logicalProcInfoCached = 1;
            else
                return retCode;
        }

        if (n >= numHWCores)
            return -1;

        mask = physLogicalProcessorMap[n];

        return 0;
    }

    /* Returns decimal value for a 32 bit mask at compile time, [i:j] set to 1, rest are 0. */
    constexpr int GenerateMask(int i, int j)
    {
        if (i > j)
            return (1 << (i + 1)) - (1 << j);
        else
            return (1 << (j + 1)) - (1 << i);
    }

    void GetCacheInfo(int* dCaches, int& iCache)
    {
        /*
        * From Intel's Processor Identification CPUID Instruction Notes:
        * EAX := 0x04, ECX := (0, 1, 2 .. until EAX[4:0]==0)
        * cpuid(memaddr, n, k) sets eax to n, ecx to k,
        * writes EAX, EBX, ECX, and EDX to memaddr[0:4] respectively.
        * Cache size in bytes = (Ways + 1) * (Partitions + 1)
        *                                  * (Line size + 1) * (Sets + 1)
        *                     = (EBX[31:22]+1) * (EBX[21:12]+1)
        *                                      * (EBX[11:0]+1) * (ECX+1)
        * For now, this function assumes we're on a modern Intel CPU
        * So we have L1,2,3 data caches and first level instruction cache
        */

        int cpui[4];

        for (int i = 0, dc = 0; i < 4; ++i) {
            __cpuidex(cpui, 4, i);
            int sz = (((cpui[1] & GenerateMask(31, 22)) >> 22) + 1) *
                     (((cpui[1] & GenerateMask(21, 12)) >> 12) + 1) *
                     ((cpui[1] & GenerateMask(11, 0)) + 1) * (cpui[2] + 1);
            int cacheType = (cpui[0] & 31);
            if (cacheType == 1 || cacheType == 3) {
                dCaches[dc++] = sz;
            } else if (cacheType == 2) {
                iCache = sz;
            }
        }
    }

    int GetCacheLineSize()
    {
        /*
        * From Intel's Processor Identification CPUID Instruction Notes:
        * Executing CPUID with EAX=1, fills EAX, EBX, ECX, EDX
        * EBX[15:8] : CLFLUSHSIZE, val*8 = cache line size
        */
        int cpui[4];
        __cpuid(cpui, 1);
        return (cpui[1] & GenerateMask(15, 8)) >> (8 - 3);
    }

    int GetHTTStatus() {
        int cpui[4];
        __cpuid(cpui, 1);
        return ((cpui[3] & (1<<28)) >> 28) ? 1 : 0;
    }

    int GetSIMDSupport() {
        int cpui[4];
        __cpuid(cpui, 1);
        int fma = (cpui[2] & (1 << 12)) >> 12;
        int avx = (cpui[2] & (1 << 28)) >> 28;
        return fma & avx;
    }

}; // namespace CPUUtil
