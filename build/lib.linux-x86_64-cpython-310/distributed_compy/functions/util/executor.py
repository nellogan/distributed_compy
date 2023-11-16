import subprocess


def executor(temp_file_name=None, hostfile_path=None):
    """
        NOTE: mpiexec -rank-by node switch -> ranks are performed on nodes in round-robin fashion,
        meaning rank0 -> node0, rank1 -> node1, rank2 -> node0, ...
        Only need one process per node for these calculations possible TODO for other fns.

        Alternative: mpiexec -n 1 --hostfile hostfile_path -rf rankfile.txt executable.exe

        $cat rankfile.txt
        rank 0=node0 slot=0
        rank 1=node1 slot=0

        This alternative allows explicit mapping of rank X=nodeY
    """
    if not temp_file_name:
        return print("Error, invalid invalid py_wrapper / temp_file_name")

    if not hostfile_path:
        subprocess.run([f"mpiexec -rank-by node -N 1 python {temp_file_name}"], shell=True)
    else:
        # hostfile_path should be the absolute NFS file path
        subprocess.run([f"mpiexec -rank-by node -N 1 --hostfile {hostfile_path} {temp_file_name}"],
                       shell=True)
