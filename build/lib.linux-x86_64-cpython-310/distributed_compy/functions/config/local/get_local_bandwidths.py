import numpy as np
from distributed_compy._config._local.lib._get_local_bands import _get_local_bands, _get_total_local_band


def get_local_bandwidths():
    return _get_local_bands()


def get_total_local_bandwidth():
    return _get_total_local_band()


def get_local_bandwidths_from_file(local_bands_file_name=None):
    """
    :param local_bands_file_name: file path of local_bands_file_name
    :return: Bandwidth of each node in cluster as a numpy array of dtype=np.float32
    """
    if not local_bands_file_name:
        print("Error: invalid local_bands_file_name.")
        return
    return np.fromfile(local_bands_file_name, dtype=np.float32)


def get_total_local_bandwidth_from_file(total_local_band_file_name=None):
    """
    :param total_local_band_file_name: file path of total_local_band_file_name
    :return: Total bandwidth of cluster
    """
    if not total_local_band_file_name:
        print("Error: invalid total_local_band_file_name.")
        return
    return np.fromfile(total_local_band_file_name, dtype=np.float32)
