from distributed_compy._lib._naive_sum.lib._naive_sum import _naive_sum
from distributed_compy._lib._omp_reduction_sum.lib._omp_reduction_sum import _omp_reduction_sum


# TODO: Refactor _naive_sum _omp_reduction_sum and with templates to support half float, double, long double
# TODO: Include type checking e.g. :param arr: must be a numpy arr of dtype float32
def reduction_sum(arr, multithreaded=False):
    """
    :param arr: Must be a numpy array of dtype float32
    :param multithreaded: If multithreaded == True, OMPNaiveSum with all available CPU cores is called,
    else NaiveSum.cpp is called
    :return: If threads == 1, NaiveSum.cpp is called, else OMPNaiveSum with all available CPU cores is called
    """
    if len(arr) < 0:
        return 0
    if not multithreaded:
        return _naive_sum(arr)
    else:
        return _omp_reduction_sum(arr)
