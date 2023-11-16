import numpy as np
import os
from pathlib import Path
from distributed_compy.functions.util.executor import executor


# TODO: generalize temp_py_writer -> separate function: to accept temp_file_name, pyx_wrapper_module_name and **kwargs
def temp_py_writer(temp_file_name, pyx_wrapper_module_name, pyx_wrapped_fn_name,
                   data_file_path, res_file_path,
                   local_bands_path, total_local_band_path,
                   node_bands_file_path, total_node_band_path, configured=False):
    file = open(f'{temp_file_name}', 'w')

    if configured:
        file.write(f'import numpy as np\n'
                   f'from compy._lib.{pyx_wrapper_module_name}.lib.{pyx_wrapper_module_name} import {pyx_wrapped_fn_name}\n'
                   f'{pyx_wrapped_fn_name}("{data_file_path}", "{res_file_path}", "{local_bands_path}", "{total_local_band_path}", "{node_bands_file_path}", "{total_node_band_path}")\n'
                   )
        file.close()
    else:
        file.write(f'import numpy as np\n'
                   f'from distributed_compy._lib.{pyx_wrapper_module_name}.lib.{pyx_wrapper_module_name} import {pyx_wrapped_fn_name}\n'
                   f'{pyx_wrapped_fn_name}("{data_file_path}", "{res_file_path}")\n'
                   )
        file.close()


# TODO: type and error checking, alternate prepper for c/c++
# computing directly from data_file_name rather than from data (numpy arr)
# maybe use templates to all Sum Cpp functions to handle half precision floats, doubles, long doubles or integer types.
def hybrid_sum_prepper(data=None, pyx_wrapped_fn=None, pyx_wrapper_module_name=None,
                       local_bands_path=None, total_local_band_path=None,
                       node_bands_path=None, total_node_bands_path=None,
                       data_file_path=None, res_file_path=None, configured=False,
                       remove_fn=True, remove_tmp_data=True, remove_tmp_res=False,
                       hostfile_path=None, local_dir=None, network_dir=None):
    if not pyx_wrapped_fn:
        return

    pyx_wrapped_fn_name = pyx_wrapped_fn.__name__

    if not local_dir:
        curr_dir = os.getcwd()
        local_dir = Path(curr_dir).joinpath("./local_tmp")
        local_dir.mkdir(parents=True, exist_ok=True)
    else:  # creates any parent dirs for tmp_dir specified
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)

    if not network_dir:
        curr_dir = os.getcwd()
        network_dir = Path(curr_dir).joinpath("./network_tmp")
        network_dir.mkdir(parents=True, exist_ok=True)
    else:  # creates any parent dirs for tmp_dir specified
        network_dir = Path(network_dir)
        network_dir.mkdir(parents=True, exist_ok=True)

    # if data -> will be saved to file, then read from each node
    if data is not None:
        data_file_path = network_dir.joinpath(f"{pyx_wrapped_fn_name}" + "_data.bin")
        data.tofile(data_file_path)

    # else load data from supplied data_file_path

    if not res_file_path:
        res_file_path = network_dir.joinpath(f"{pyx_wrapped_fn_name}" + "_res.bin")

    if not local_bands_path:
        local_bands_path = local_dir.joinpath("local_bands.bin")
    if not total_local_band_path:
        total_local_band_path = local_dir.joinpath("total_local_band.bin")
    if not node_bands_path:
        node_bands_path = network_dir.joinpath("network_bands.bin")
    if not total_node_bands_path:
        total_node_bands_path = network_dir.joinpath("total_network_band.bin")

    temp_file_name = local_dir.joinpath(pyx_wrapped_fn_name + ".py")

    if not pyx_wrapper_module_name:
        pyx_wrapper_module_name = pyx_wrapped_fn_name

    temp_py_writer(temp_file_name, pyx_wrapper_module_name, pyx_wrapped_fn_name,
                   data_file_path, res_file_path,
                   local_bands_path, total_local_band_path,
                   node_bands_path, total_node_bands_path, configured=configured)

    executor(temp_file_name=temp_file_name, hostfile_path=hostfile_path)

    if remove_fn:
        os.remove(f'{temp_file_name}')

    res = np.fromfile(res_file_path, np.float32)

    if remove_tmp_data:
        os.remove(f'{data_file_path}')
    if remove_tmp_res:
        os.remove(f'{res_file_path}')

    return res
