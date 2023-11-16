#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// Pageable memory
void GPUReductionSum(float* h_in, float* h_out, int size);
//// MultiGPUReductionSum incorporates loads passed from heterogenous sum
void MultiGPUReductionSum(float* h_in, float* h_out, int size);
void MultiGPUReductionSum2(float* h_in, float* h_out, int* loads, int size);
//
//// Page-locked memory
void GPUReductionSumPinned(float* h_in, float* h_out, int size);
void MultiGPUReductionSumPinned(float* h_in, float* h_out, int size);


// if porting to windows
//#ifdef GPUREDUCTIONSUM_EXPORTS
//#define DLLEXPORT __declspec(dllexport)
//#else
//#define DLLEXPORT __declspec(dllimport)
//#endif
//#if defined(_MSC_VER)
//#ifdef __cplusplus
//extern "C" {
//#endif
//DLLEXPORT void GPUReductionSum(float* h_in, float* h_out, int size);
//
//#ifdef __cplusplus
//}
//#endif
//
//#else
//    extern "C" void GPUReductionSum(float* h_in, float* h_out, int size);
//#endif