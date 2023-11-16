def py_naive_sum(arr):
    """
    :param arr: Can be any python type array
    :return: Naive sum of arr
    """
    if len(arr) < 1:
        return 0
    res = 0
    for i in range(len(arr)):
        res += arr[i]
    return res
