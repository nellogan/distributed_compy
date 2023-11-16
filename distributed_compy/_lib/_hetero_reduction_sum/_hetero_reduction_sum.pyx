cimport numpy as np
import numpy as np
import cython

cdef extern from "NaiveSum.h":
    void NaiveSum(float* data, float* out, int size);

cdef extern from "HeteroReductionSum.h":
    void HeteroReductionSum(float* data, float* res, int size);
    void HeteroReductionSumBands(float* data, float* res, int size,
                             int num_procs, float* local_bands, float total_local_band);
    void HeteroReductionSumConfigured(float* data, float* res, int size,
                         const char *local_bands_file_name, const char *total_local_band_file_name);

def _hetero_reduction_sum(float[:] data):
    cdef int size = len(data)
    cdef float res = 0.0
    HeteroReductionSum(&data[0], &res, size)
    return np.asarray(res, dtype=np.float32)

def _hetero_reduction_sum_bands(float[:] data, float[:] local_bands):
    cdef int size = len(data)
    cdef int num_procs = len(local_bands)
    cdef float res = 0.0
    cdef float total_local_band;
    NaiveSum(&local_bands[0], &total_local_band, num_procs);
    HeteroReductionSumBands(&data[0], &res, size, num_procs, &local_bands[0], total_local_band)
    return np.asarray(res, dtype=np.float32)

def _hetero_reduction_sum_configured(float[:] data, str local_bands_filename, str total_local_band_filename):
    cdef int size = len(data)
    local_bands_file = local_bands_filename.encode()
    total_local_band_file = total_local_band_filename.encode()
    cdef float res = 0.0
    HeteroReductionSumConfigured(&data[0], &res, size, local_bands_file, total_local_band_file)
    return np.asarray(res, dtype=np.float32)