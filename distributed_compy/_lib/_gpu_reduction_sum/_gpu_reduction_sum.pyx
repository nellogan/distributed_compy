cimport numpy as np
import numpy as np
import cython


cdef extern from "GPUReductionSum.h":
    void GPUReductionSum(float* h_in, float* h_out, int size);
    void MultiGPUReductionSum(float* h_in, float* h_out, int size);
    void GPUReductionSumPinned(float* h_in, float* h_out, int size);
    void MultiGPUReductionSumPinned(float* h_in, float* h_out, int size);

def _gpu_reduction_sum(float[:] data):
    cdef int size = len(data)
    cdef float res = 0.0
    GPUReductionSum(&data[0], &res, size);
    return np.asarray(res, dtype=np.float32)

def _multigpu_reduction_sum(float[:] data):
    cdef int size = len(data)
    cdef float res = 0.0
    MultiGPUReductionSum(&data[0], &res, size);
    return np.asarray(res, dtype=np.float32)

def _gpu_reduction_sum_pinned(float[:] data):
    cdef int size = len(data)
    cdef float res = 0.0
    GPUReductionSumPinned(&data[0], &res, size);
    return np.asarray(res, dtype=np.float32)

def _multigpu_reduction_sum_pinned(float[:] data):
    cdef int size = len(data)
    cdef float res = 0.0
    MultiGPUReductionSumPinned(&data[0], &res, size);
    return np.asarray(res, dtype=np.float32)