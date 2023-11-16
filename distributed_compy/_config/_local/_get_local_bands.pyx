cimport numpy as np
import numpy as np
import cython

cdef extern from "GetLocalBands.h":
    float* GetLocalBands(float* total_local_band, int* num_procs);

# see https://stackoverflow.com/questions/25102409/c-malloc-array-pointer-return-in-cython
# will automatically free dynamic array
cdef pointer_to_numpy_array_float32(void * ptr, np.npy_intp size):
    '''Convert c pointer to numpy array.
    The memory will be freed as soon as the ndarray is deallocated.
    '''
    cdef extern from "numpy/arrayobject.h":
        void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)
    cdef np.ndarray[np.float32_t, ndim=1] arr = \
            np.PyArray_SimpleNewFromData(1, &size, np.NPY_FLOAT32, ptr)
    PyArray_ENABLEFLAGS(arr, np.NPY_OWNDATA)
    return arr

#simple malloc array in cython: cdef float *local_bands_array = <float *> malloc(number * sizeof(double))
def _get_local_bands():
    cdef int num_procs;
    cdef float total_local_band;
    cdef float *local_bands_array = GetLocalBands(&total_local_band , &num_procs)
    return pointer_to_numpy_array_float32(local_bands_array, num_procs)

def _get_total_local_band():
    cdef int num_procs;
    cdef float total_local_band;
    cdef float *local_bands_array = GetLocalBands(&total_local_band, &num_procs)
    return total_local_band