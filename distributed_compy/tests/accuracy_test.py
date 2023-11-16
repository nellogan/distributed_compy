import unittest
import numpy as np
from distributed_compy.functions.py_naive_sum import py_naive_sum
from distributed_compy.functions.reduction_sum import reduction_sum
from distributed_compy.functions.gpu_reduction_sum import gpu_reduction_sum, gpu_reduction_sum_pinned
from distributed_compy.functions.hetero_reduction_sum import hetero_reduction_sum
from distributed_compy.functions.hybrid_reduction_sum import hybrid_reduction_sum, hybrid_reduction_sum_from_data_file
import os
from pathlib import Path


class TestFunctionAccuracy(unittest.TestCase):
    def setUp(self):
        self.size = 1 << 16
        self.nodes = 3
        self.rng = np.random.default_rng(seed=42)
        self.arr = self.rng.random(self.size, dtype=np.float32)
        self.check = np.sum(self.arr)
        self.rtol = 1e-5
        self.curr_path = os.getcwd()
        self.tmp_data_file_path = Path(self.curr_path).joinpath("tmp_data_file.bin")
        self.arr.tofile(self.tmp_data_file_path)

    def tearDown(self):
        if os.path.isfile(self.tmp_data_file_path):
            os.remove(self.tmp_data_file_path)

    # np.testing.assert_allclose returns None if True else raises AssertionError
    def test_py_naive_sum(self):
        message = "py_reduction_sum function failed."
        self.assertIsNone(np.testing.assert_allclose(py_naive_sum(self.arr), self.check, rtol=self.rtol), message)

    def test_reduction_sum_singlethreaded(self):
        message = "Single threaded reduction_sum function failed."
        self.assertIsNone(np.testing.assert_allclose(reduction_sum(self.arr), self.check, rtol=self.rtol), message)

    def test_reduction_sum_multithreaded(self):
        message = "Multithreaded reduction_sum function failed."
        self.assertIsNone(np.testing.assert_allclose(reduction_sum(self.arr, multithreaded=True), self.check,
                                                     rtol=self.rtol), message)

    def test_gpu_reduction_sum(self):
        message = "Single-GPU gpu_reduction_sum function failed."
        self.assertIsNone(np.testing.assert_allclose(gpu_reduction_sum(self.arr, multi_gpu=False), self.check,
                                                     rtol=self.rtol), message)

    def test_multigpu_reduction_sum(self):
        message = "Multi-GPU gpu_reduction_sum function failed."
        self.assertIsNone(np.testing.assert_allclose(gpu_reduction_sum(self.arr, multi_gpu=True), self.check,
                                                     rtol=self.rtol), message)

    def test_gpu_reduction_sum_pinned(self):
        message = "Single-GPU gpu_reduction_sum_pinned function failed."
        self.assertIsNone(np.testing.assert_allclose(gpu_reduction_sum_pinned(self.arr, multi_gpu=False), self.check,
                                                     rtol=self.rtol), message)

    def test_multigpu_reduction_sum(self):
        message = "Multi-GPU gpu_reduction_sum_pinned function failed."
        self.assertIsNone(np.testing.assert_allclose(gpu_reduction_sum_pinned(self.arr, multi_gpu=True), self.check,
                                                     rtol=self.rtol), message)

    def test_hetero_reduction_sum(self):
        message = "hetero_reduction_sum function failed."
        self.assertIsNone(np.testing.assert_allclose(hetero_reduction_sum(self.arr), self.check, rtol=self.rtol),
                          message)

    def test_hybrid_reduction_sum(self):
        message = "hybrid_reduction_sum function failed."
        self.assertIsNone(np.testing.assert_allclose(hybrid_reduction_sum(self.arr), self.check, rtol=self.rtol),
                          message)

    def test_hybrid_reduction_sum_from_data_file(self):
        message = "hybrid_reduction_sum function failed."
        self.assertIsNone(np.testing.assert_allclose(hybrid_reduction_sum_from_data_file(self.tmp_data_file_path),
                                                     self.check, rtol=self.rtol),
                          message)


if __name__ == '__main__':
    unittest.main()
