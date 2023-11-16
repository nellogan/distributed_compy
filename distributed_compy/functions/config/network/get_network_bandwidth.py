import numpy as np


def get_network_bandwidths_from_file(network_bands_file_name=None):
    """
    :param network_bands_file_name: file path of network_bands_file_name
    :return: Bandwidth of each node in cluster as a numpy array of dtype=np.float32
    """
    if not network_bands_file_name:
        print("Error: invalid network_bands_file_name.")
        return
    return np.fromfile(network_bands_file_name, dtype=np.float32)


def get_total_network_bandwidth_from_file(total_network_band_file_name=None):
    """
    :param total_network_band_file_name: file path of total_network_band_file_name
    :return: Total bandwidth of cluster
    """
    if not total_network_band_file_name:
        print("Error: invalid total_network_band_file_name.")
        return
    return np.fromfile(total_network_band_file_name, dtype=np.float32)
