cimport numpy as np
import numpy as np
import cython

cdef extern from "HybridReductionSum.h":
    void HybridReductionSum(const char* data_file_name, const char* res_file_name);
    void HybridReductionSumConfigured(const char* data_file_name, const char* res_file_name,
                        const char* local_bands_file, const char* total_local_band_file,
                        const char* node_bands_file, const char* total_node_band_file);


def _hybrid_reduction_sum(str data_file_name, str res_file_name):
    data = data_file_name.encode()
    res = res_file_name.encode()
    HybridReductionSum(data, res)

def _hybrid_reduction_sum_configured(str data_file_name, str res_file_name, str local_bands_file_name,
                                     str total_local_file_name, str node_bands_file_name,
                                     str total_nodes_band_file_name):
    data = data_file_name.encode()
    res = res_file_name.encode()
    local_bands_file = local_bands_file_name.encode()
    total_local_file = total_local_file_name.encode()
    node_bands_file = node_bands_file_name.encode()
    total_nodes_band_file = total_nodes_band_file_name.encode()
    HybridReductionSumConfigured(data, res, local_bands_file, total_local_file,
                                     node_bands_file, total_nodes_band_file)