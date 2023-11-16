import unittest
import numpy as np
import os
from pathlib import Path

from distributed_compy.functions.config.local.get_local_bandwidths import get_local_bandwidths, \
    get_total_local_bandwidth
from distributed_compy.functions.config.local.set_local_bandwidths import set_local_bandwidths
from distributed_compy.functions.config.local.configure_local import configure_local
from distributed_compy.functions.config.network.get_network_bandwidth import (get_network_bandwidths_from_file,
                                                                              get_total_network_bandwidth_from_file)
from distributed_compy.functions.config.network.set_network_bandwidth import set_network_bandwidth
from distributed_compy.functions.config.network.configure_network import configure_network, configure_cluster


# could use builtin python tempfile instead of creating / deleting files manually
class TestLocalUtilFunctions(unittest.TestCase):
    def setUp(self):
        self.curr_path = os.getcwd()
        self.tmp_local_bands_file_path = Path(self.curr_path).joinpath("tmp_local_bands_file_path.bin")
        self.tmp_total_local_band_file_path = Path(self.curr_path).joinpath("tmp_total_local_band_file_path.bin")
        self.tmp_network_bands_file_path = Path(self.curr_path).joinpath("tmp_network_bands_file_path.bin")
        self.tmp_total_network_band_file_path = Path(self.curr_path).joinpath("tmp_total_network_band_file_path.bin")
        self.proc_arr = np.asarray([345, 777, 888], dtype=np.float32)
        self.total_proc_arr = np.sum(self.proc_arr)

    def tearDown(self):
        if os.path.isfile(self.tmp_local_bands_file_path):
            os.remove(self.tmp_local_bands_file_path)

        if os.path.isfile(self.tmp_total_local_band_file_path):
            os.remove(self.tmp_total_local_band_file_path)

        if os.path.isfile(self.tmp_network_bands_file_path):
            os.remove(self.tmp_network_bands_file_path)

        if os.path.isfile(self.tmp_total_network_band_file_path):
            os.remove(self.tmp_total_network_band_file_path)

    # get
    def test_get_local_bandwidths(self):
        message = "Error: get_local_bandwidths function failed."
        res = all(get_local_bandwidths() > 0)
        self.assertTrue(res, message)

    def test_get_total_local_bandwidth(self):
        message = "Error: get_total_local_bandwidth function failed."
        res = get_total_local_bandwidth() > 0
        self.assertTrue(res, message)

    # set
    def test_set_local_bandwidths(self):
        message = "Error: set_local_bandwidths function failed."
        set_local_bandwidths(self.proc_arr, self.tmp_local_bands_file_path, self.tmp_total_local_band_file_path)
        set_local_bands_res = np.fromfile(self.tmp_local_bands_file_path, dtype=np.float32)
        set_total_local_band_res = np.fromfile(self.tmp_total_local_band_file_path, dtype=np.float32)
        res_1 = np.allclose(set_local_bands_res, self.proc_arr)
        res_2 = np.allclose(set_total_local_band_res, np.sum(self.proc_arr))
        os.remove(self.tmp_local_bands_file_path)
        os.remove(self.tmp_total_local_band_file_path)
        final_res = all([res_1, res_2])
        self.assertTrue(final_res, message)

    # configure
    def test_configure_local_bandwidths(self):
        message = "Error: configure_local_bandwidths function failed."
        configure_local(self.tmp_local_bands_file_path, self.tmp_total_local_band_file_path)
        config_local_bands = np.fromfile(self.tmp_local_bands_file_path, dtype=np.float32)
        config_total_local_band = np.fromfile(self.tmp_total_local_band_file_path, dtype=np.float32)
        res_1 = np.allclose(config_local_bands, get_local_bandwidths())
        res_2 = np.allclose(config_total_local_band, get_total_local_bandwidth())
        os.remove(self.tmp_local_bands_file_path)
        os.remove(self.tmp_total_local_band_file_path)
        final_res = all([res_1, res_2])
        self.assertTrue(final_res, message)


class TestNetworkUtilFunctions(unittest.TestCase):
    def setUp(self):
        self.curr_path = os.getcwd()
        self.tmp_local_bands_file_path = Path(self.curr_path).joinpath("tmp_local_bands_file_path.bin")
        self.tmp_total_local_band_file_path = Path(self.curr_path).joinpath("tmp_total_local_band_file_path.bin")
        self.tmp_network_bands_file_path = Path(self.curr_path).joinpath("tmp_network_bands_file_path.bin")
        self.tmp_total_network_band_file_path = Path(self.curr_path).joinpath("tmp_total_network_band_file_path.bin")
        self.proc_arr = np.asarray([345, 777, 888], dtype=np.float32)
        self.total_proc_arr = np.asarray([2010], dtype=np.float32)

    def tearDown(self):
        if os.path.isfile(self.tmp_local_bands_file_path):
            os.remove(self.tmp_local_bands_file_path)

        if os.path.isfile(self.tmp_total_local_band_file_path):
            os.remove(self.tmp_total_local_band_file_path)

        if os.path.isfile(self.tmp_network_bands_file_path):
            os.remove(self.tmp_network_bands_file_path)

        if os.path.isfile(self.tmp_total_network_band_file_path):
            os.remove(self.tmp_total_network_band_file_path)

    def test_get_network_bandwidths(self):
        message = "Error: get_network_bandwidths function failed."

        # supply data
        self.proc_arr.tofile(self.tmp_network_bands_file_path)

        get_network_bandwidths_res = get_network_bandwidths_from_file(
            network_bands_file_name=self.tmp_network_bands_file_path)
        res_1 = np.allclose(self.proc_arr, get_network_bandwidths_res)

        # delete supplied data
        os.remove(self.tmp_network_bands_file_path)

        self.assertTrue(res_1, message)

    def test_get_total_network_bandwidth(self):
        message = "Error: get_total_network_bandwidth function failed."

        # supply data
        np.sum(self.proc_arr).tofile(self.tmp_total_network_band_file_path)

        get_total_network_bandwidth_from_file_res = get_total_network_bandwidth_from_file(
            total_network_band_file_name=self.tmp_total_network_band_file_path)
        res_1 = np.allclose(np.sum(self.proc_arr), get_total_network_bandwidth_from_file_res)

        # delete supplied data
        os.remove(self.tmp_total_network_band_file_path)

        self.assertTrue(res_1, message)

    def test_set_network_bandwidths(self):
        message = "Error: set_network_bandwidths function failed."
        set_network_bandwidth(self.proc_arr, self.tmp_network_bands_file_path, self.tmp_total_network_band_file_path)
        set_network_bands = np.fromfile(self.tmp_network_bands_file_path, dtype=np.float32)
        set_total_network_band = np.fromfile(self.tmp_total_network_band_file_path, dtype=np.float32)
        res_1 = np.allclose(set_network_bands, self.proc_arr)
        res_2 = np.allclose(set_total_network_band, np.sum(self.proc_arr))
        final_res = all([res_1, res_2])
        os.remove(self.tmp_network_bands_file_path)
        os.remove(self.tmp_total_network_band_file_path)
        self.assertTrue(final_res, message)

    def test_configure_network(self):
        message = "Error: configure_network function failed."
        configure_network(self.tmp_network_bands_file_path, self.tmp_total_network_band_file_path)
        network_bands_res = np.fromfile(self.tmp_network_bands_file_path, dtype=np.float32)
        total_network_band_res = np.fromfile(self.tmp_total_network_band_file_path, dtype=np.float32)
        res_1 = np.all(network_bands_res > 0)
        res_2 = total_network_band_res > 0
        final_res = all([res_1, res_2])
        os.remove(self.tmp_network_bands_file_path)
        os.remove(self.tmp_total_network_band_file_path)
        self.assertTrue(final_res, message)

    def test_configure_cluster(self):
        message = "Error: configure_cluster function failed."
        configure_cluster(self.tmp_local_bands_file_path, self.tmp_total_local_band_file_path,
                          self.tmp_network_bands_file_path, self.tmp_total_network_band_file_path)
        local_bands_res = np.fromfile(self.tmp_local_bands_file_path, dtype=np.float32)
        total_local_band_res = np.fromfile(self.tmp_total_local_band_file_path, dtype=np.float32)
        network_bands_res = np.fromfile(self.tmp_network_bands_file_path, dtype=np.float32)
        total_network_band_res = np.fromfile(self.tmp_total_network_band_file_path, dtype=np.float32)

        res_1 = np.all(local_bands_res > 0)
        res_2 = total_local_band_res > 0
        res_3 = np.all(network_bands_res > 0)
        res_4 = total_network_band_res > 0

        os.remove(self.tmp_local_bands_file_path)
        os.remove(self.tmp_total_local_band_file_path)
        os.remove(self.tmp_network_bands_file_path)
        os.remove(self.tmp_total_network_band_file_path)
        final_res = all([res_1, res_2, res_3, res_4])

        self.assertTrue(final_res, message)


if __name__ == '__main__':
    unittest.main()
