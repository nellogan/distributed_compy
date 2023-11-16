import os
from pathlib import Path
from distributed_compy.functions.util.executor import executor


# TODO: generalize temp_py_writer -> separate function: to accept temp_file_name, pyx_wrapper_module_name and **kwargs
def temp_py_writer(temp_file_name, pyx_wrapped_fn_name,
                   network_bands_path, total_network_bands_path):
    file = open(f'{temp_file_name}', 'w')
    print("temp_file_name", temp_file_name)
    file.write(f'import numpy as np\n'
               f'from distributed_compy._config._network.lib._configure_network import {pyx_wrapped_fn_name}\n'
               f'{pyx_wrapped_fn_name}("{network_bands_path}", "{total_network_bands_path}")\n'
               )
    file.close()


# TODO: type and error checking, alternate prepper for c/c++
# computing directly from data_file_name rather than from data (numpy arr)
# maybe use templates to all Sum Cpp functions to handle half precision floats, doubles, long doubles or integer types.
def network_prepper(pyx_wrapped_fn=None, network_bands_path=None, total_network_bands_path=None,
                    remove_fn=True, hostfile_path=None, network_dir=None):
    if not pyx_wrapped_fn:
        return

    pyx_wrapped_fn_name = pyx_wrapped_fn.__name__

    if not network_dir:
        curr_dir = os.getcwd()
        network_dir = Path(curr_dir).joinpath("./network_tmp")
        network_dir.mkdir(parents=True, exist_ok=True)
    else:
        network_dir = Path(network_dir)
        network_dir.mkdir(parents=True, exist_ok=True)

    if not network_bands_path:
        network_bands_path = network_dir.joinpath("network_bands.bin")
    if not total_network_bands_path:
        total_network_bands_path = network_dir.joinpath("total_network_band.bin")

    temp_file_name = network_dir.joinpath(pyx_wrapped_fn_name + ".py")
    temp_py_writer(temp_file_name, pyx_wrapped_fn_name,
                   network_bands_path, total_network_bands_path)
    executor(temp_file_name=temp_file_name, hostfile_path=hostfile_path)

    if remove_fn:
        os.remove(f'{temp_file_name}')
