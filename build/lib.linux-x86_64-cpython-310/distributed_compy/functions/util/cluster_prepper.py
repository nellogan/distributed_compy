import os
from pathlib import Path
from distributed_compy.functions.util.executor import executor


def temp_py_writer(temp_file_name, pyx_wrapped_fn_name, local_bands_path, total_local_band_path,
                   node_bands_file_path, total_node_band_path):
    file = open(f'{temp_file_name}', 'w')
    file.write(f'import numpy as np\n'
               f'from compy._config._network.lib._configure_network_band import {pyx_wrapped_fn_name}\n'
               f'{pyx_wrapped_fn_name}("{local_bands_path}", "{total_local_band_path}", {node_bands_file_path}, {total_node_band_path})\n'
               )
    file.close()


# TODO: type and error checking, alternate prepper for c/c++
# computing directly from data_file_name rather than from data (numpy arr)
# maybe use templates to all Sum Cpp functions to handle half precision floats, doubles, long doubles or integer types.

# _configure_cluster(str local_bands_file_name, str total_local_band_file_name,
#                        str node_bands_file_name, str total_network_band_file_name)
def cluster_prepper(pyx_wrapped_fn=None, local_bands_path=None, total_local_band_path=None,
                    node_bands_path=None, total_node_bands_path=None,
                    remove_fn=True, hostfile_path=None, local_dir=None, network_dir=None):
    if not pyx_wrapped_fn:
        return

    pyx_wrapped_fn_name = pyx_wrapped_fn.__name__

    if not local_dir:
        curr_dir = os.getcwd()
        local_dir = Path(curr_dir).joinpath("./local_tmp")
        local_dir.mkdir(parents=True, exist_ok=True)
    else:
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)

    if not network_dir:
        curr_dir = os.getcwd()
        network_dir = Path(curr_dir).joinpath("./network_tmp")
        network_dir.mkdir(parents=True, exist_ok=True)
    else:
        network_dir = Path(network_dir)
        network_dir.mkdir(parents=True, exist_ok=True)

    if not local_bands_path:
        local_bands_path = local_dir.joinpath("local_bands.bin")
    if not total_local_band_path:
        total_local_band_path = local_dir.joinpath("total_local_band.bin")
    if not node_bands_path:
        node_bands_path = network_dir.joinpath("network_bands.bin")
    if not total_node_bands_path:
        total_node_bands_path = network_dir.joinpath("total_network_band.bin")

    temp_file_name = local_dir.joinpath(pyx_wrapped_fn_name + ".py")

    temp_py_writer(temp_file_name, pyx_wrapped_fn_name, local_bands_path, total_local_band_path,
                   node_bands_path, total_node_bands_path)

    executor(temp_file_name=temp_file_name, hostfile_path=hostfile_path)

    if remove_fn:
        os.remove(f'{temp_file_name}')
