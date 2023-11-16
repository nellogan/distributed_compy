from distributed_compy._config._local.lib._configure_local import _configure_local


def configure_local(local_bands_file_path, total_local_band_file_path):
    """
    Calculates both local_bands array and total local bandwidth then saves each as in local_bands_file_path and
    total_local_band_file_path respectfully
    :return:
    """
    _configure_local(str(local_bands_file_path), str(total_local_band_file_path))
