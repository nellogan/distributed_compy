cimport numpy as np
import numpy as np
import cython

cdef extern from "ConfigureLocal.h":
    void ConfigureLocal(const char *local_bands_file_name, const char *total_local_band_file_name);

def _configure_local(str local_bands_file_name, str total_local_band_file_name):
    local_bands_file = local_bands_file_name.encode()
    total_local_band_file = total_local_band_file_name.encode()
    ConfigureLocal(local_bands_file, total_local_band_file);
