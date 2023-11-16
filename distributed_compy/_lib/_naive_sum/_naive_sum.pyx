cimport numpy as np
import numpy as np
import cython

cdef extern from "NaiveSum.h":
    void NaiveSum(float* data, float* out, int size)

def _naive_sum(float[:] data):
    cdef int size = len(data)
    cdef float res = 0.0
    NaiveSum(&data[0], &res, size);
    return np.asarray(res, dtype=np.float32)