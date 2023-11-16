cimport numpy as np
import numpy as np
import cython

cdef extern from "OMPReductionSum.h":
    void OMPReductionSum(float* data, float* out, int size)

def _omp_reduction_sum(float[:] data):
    cdef int size = len(data)
    cdef float res = 0.0
    OMPReductionSum(&data[0], &res, size);
    return np.asarray(res, dtype=np.float32)