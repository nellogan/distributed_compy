import numpy as np


def set_network_bandwidth(arr, network_bands_file_path, total_network_band_file_path):
    """
    :param arr: Shall be a numpy array of dtype=np.float32
    :param local_bands_file_path:
    :param total_local_band_file_path:
    :return:
    """
    arr.tofile(network_bands_file_path)
    np.sum(arr).tofile(total_network_band_file_path)
