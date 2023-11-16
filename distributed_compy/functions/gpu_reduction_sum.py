from distributed_compy._lib._gpu_reduction_sum.lib._gpu_reduction_sum import _gpu_reduction_sum, _multigpu_reduction_sum, \
    _gpu_reduction_sum_pinned, _multigpu_reduction_sum_pinned


# TODO: Refactor _hetero_reduction_sum with templates to support half float, double, long double
# TODO: Include type checking e.g. :param arr: must be a numpy arr of dtype float32
def gpu_reduction_sum(arr, multi_gpu=False):
    """
    :param arr: Must be a numpy arr of dtype float32
    :param multi_gpu: if True, all available GPUs are called. Assumes NVIDIA CUDA capable GPUs.
    :return: GPU reduction sum -- allocates and uses pageable memory, and GPUs on current node
    """
    if len(arr) < 0:
        return 0
    if not multi_gpu:
        return _gpu_reduction_sum(arr)
    else:
        return _multigpu_reduction_sum(arr)


# TODO: Refactor _hetero_reduction_sum with templates to support half float, double, long double
# TODO: Include type checking e.g. :param arr: must be a numpy arr of dtype float32
def gpu_reduction_sum_pinned(arr, multi_gpu=False):
    """
    :param arr: Must be a numpy arr of dtype float32
    :param multi_gpu: if True, all available GPUs are called. Assumes NVIDIA CUDA capable GPUs.
    :return: GPU reduction sum -- allocates and uses page-locked(pinned) memory, and GPUs on current node
    """
    if len(arr) < 0:
        return 0
    if not multi_gpu:
        return _gpu_reduction_sum_pinned(arr)
    else:
        return _multigpu_reduction_sum_pinned(arr)
