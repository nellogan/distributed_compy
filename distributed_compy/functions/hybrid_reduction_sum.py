from distributed_compy._lib._hybrid_reduction_sum.lib._hybrid_reduction_sum import (_hybrid_reduction_sum,
                                                                                    _hybrid_reduction_sum_configured)
from distributed_compy.functions.util.hybrid_sum_prepper import hybrid_sum_prepper


# TODO: Refactor _hetero_reduction_sum with templates to support half float, double, long double
# TODO: Include type checking e.g. :param arr: must be a numpy arr of dtype float32
# TODO: Add config function to set data/res file path -- for now saves to tmp
# TODO: Add param to set specific nodes from either hostfile or tmp_hostfile
def hybrid_reduction_sum(arr, res_file_path=None):
    '''
    :param arr: Must be a numpy arr of dtype float32
    :return: Hybrid reduction sum, utilizing all available CPU cores and GPUs on number of nodes specified
    '''
    if len(arr) < 1:
        return 0
    res = hybrid_sum_prepper(data=arr, pyx_wrapped_fn=_hybrid_reduction_sum, res_file_path=res_file_path)
    return res


def hybrid_reduction_sum_from_data_file(data_file_path, res_file_path=None):
    '''
    :param data_file_path: Absolute path to data_file. data_file must be in binary format holding float32 elements.
    Should also be a network file.
    :return: Hybrid reduction sum, utilizing all available CPU cores and GPUs on number of nodes specified
    '''
    res = hybrid_sum_prepper(data_file_path=data_file_path, pyx_wrapped_fn=_hybrid_reduction_sum,
                             res_file_path=res_file_path)
    return res


def hybrid_reduction_sum_configured(arr, local_bands_file_name, lotal_local_file_name, node_bands_file_name,
                                    total_nodes_band_file_name, res_file_path=None):
    '''
    :param arr: Must be a numpy arr of dtype float32
    :return: Hybrid reduction sum, utilizing all available CPU cores and GPUs on number of nodes specified
    '''
    if len(arr) < 1:
        return 0
    res = hybrid_sum_prepper(data=arr, pyx_wrapped_fn=_hybrid_reduction_sum_configured,
                             pyx_wrapper_module_name="_hybrid_reduction_sum",
                             local_bands_path=local_bands_file_name, total_local_band_path=lotal_local_file_name,
                             node_bands_path=node_bands_file_name, total_node_bands_path=total_nodes_band_file_name,
                             configured=True, res_file_path=res_file_path)
    return res
