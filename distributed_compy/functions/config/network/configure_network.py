from distributed_compy._config._network.lib._configure_network import _configure_network
from distributed_compy.functions.util.network_prepper import network_prepper
from distributed_compy._config._network.lib._configure_network import _configure_cluster
from distributed_compy.functions.util.system_prepper import system_prepper


def configure_network(network_bands_file_path=None, total_network_file_path=None):
    """
    Calculates the bandwidth of CPU and GPUs per node, saves each node's individual bandwidth to network_bands_file_path
    and saves the total bandwidth of the cluster to total_network_file_path
    :param network_bands_file_path:
    :param total_network_file_path:
    :return:
    """
    if not network_bands_file_path or not total_network_file_path:
        print("Error both network_bands_file_path and total_network_file_path must be specified.")
        return

    network_prepper(pyx_wrapped_fn=_configure_network, network_bands_path=network_bands_file_path,
                    total_network_bands_path=total_network_file_path,
                    remove_fn=True, hostfile_path=None, network_dir=None)


def configure_cluster(local_bands_file_path=None, total_local_band_file_path=None,
                      network_bands_file_path=None, total_network_file_path=None):
    """
    Calculates the bandwidth of CPU and GPUs per node, saves each node's individual bandwidth to network_bands_file_path
    and saves the total bandwidth of the cluster to total_network_file_path
    :param network_bands_file_path:
    :param total_network_file_path:
    :return:
    """
    if not network_bands_file_path or not total_network_file_path:
        print("Error both network_bands_file_path and total_network_file_path must be specified.")
        return

    # system_prepper(pyx_wrapped_fn=None, local_bands_path=None, total_local_band_path=None,
    #                    network_bands_path=None, total_network_bands_path=None,
    #                    remove_fn=True, hostfile_path=None, local_dir=None, network_dir=None):

    system_prepper(pyx_wrapped_fn=_configure_cluster,
                   local_bands_path=local_bands_file_path,
                   total_local_band_path=total_local_band_file_path,
                   network_bands_path=network_bands_file_path,
                   total_network_bands_path=total_network_file_path,
                   remove_fn=True, hostfile_path=None, network_dir=None)
