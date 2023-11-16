from distributed_compy._lib._hetero_reduction_sum.lib._hetero_reduction_sum import _hetero_reduction_sum


# TODO: Refactor _hetero_reduction_sum with templates to support half float, double, long double
# TODO: Include type checking e.g. :param arr: must be a numpy arr of dtype float32
def hetero_reduction_sum(arr):
    """
    :param arr: Must be a numpy arr of dtype float32
    :return: Heterogeneous reduction sum, utilizing all available CPU cores and GPUs on current node
    """
    if len(arr) < 1:
        return 0
    return _hetero_reduction_sum(arr)
