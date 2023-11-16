import numpy as np


def set_local_bandwidths(arr, local_bands_file_path, total_local_band_file_path):
    """
    :param arr: Shall be a numpy array of dtype=np.float32
    :param local_bands_file_path:
    :param total_local_band_file_path:
    :return:
    """
    arr.tofile(local_bands_file_path)
    np.sum(arr).tofile(total_local_band_file_path)
