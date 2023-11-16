cimport numpy as np
import numpy as np
import cython

cdef extern from "ConfigureNetwork.h":
    void ConfigureNetwork(const char *network_bands_file_name, const char *total_network_band_file_name);
    void ConfigureCluster(const char *local_bands_file_name, const char *total_local_band_file_name, const char *network_bands_file_name, const char *total_network_band_file_name);

def _configure_network(str network_bands_file, str total_networks_band_file):
    network_bands = network_bands_file.encode()
    total_networks_band = total_networks_band_file.encode()
    ConfigureNetwork(network_bands, total_networks_band)

def _configure_cluster(str local_bands_file, str total_local_band_file, str network_bands_file, str total_networks_band_file):
    local_bands = local_bands_file.encode()
    total_local_band = total_local_band_file.encode()
    network_bands = network_bands_file.encode()
    total_networks_band = total_networks_band_file.encode()
    ConfigureCluster(local_bands, total_local_band, network_bands, total_networks_band)