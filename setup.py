from setuptools import setup, Extension, find_packages
import numpy as np
import os
from os.path import join as pjoin
from pathlib import Path

try:
    from Cython.Distutils import build_ext
    USE_CYTHON = True
    wrapper_file_ext = '.pyx'

except ImportError:
    from setuptools.command.build_ext import build_ext
    USE_CYTHON = False
    wrapper_file_ext = '.cpp'


def find_in_path(name, path):
    """Find a file in a search path"""

    for dir in path.split(os.pathsep):
        binpath = pjoin(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None


# Locate CUDA
def locate_cuda():
    """
    Locate the CUDA environment on the system
    Returns a dict with keys 'home', 'nvcc', 'include', and 'lib64'
    and values giving the absolute path to each directory.
    Starts by looking for the CUDAHOME env variable. If not found,
    everything is based on finding 'nvcc' in the PATH.
    """

    # First check if the CUDAHOME env variable is in use
    if 'CUDAHOME' in os.environ:
        home = os.environ['CUDAHOME']
        nvcc = pjoin(home, 'bin')
    else:
        # Otherwise, search the PATH for NVCC
        nvcc = find_in_path('nvcc', os.environ['PATH'])
        if nvcc is None:
            raise EnvironmentError('The nvcc binary could not be '
                                   'located in your $PATH. Either add it to your path, '
                                   'or set $CUDAHOME')
        home = os.path.dirname(os.path.dirname(nvcc))

    cudaconfig = {'home': home, 'nvcc': nvcc,
                  'include': pjoin(home, 'include'),
                  'lib64': pjoin(home, 'lib64')}

    for k, v in iter(cudaconfig.items()):
        if not os.path.exists(v):
            raise EnvironmentError('The CUDA %s path could not be '
                                   'located in %s' % (k, v))

    return cudaconfig


CUDA = locate_cuda()


# Locate MPI
def locate_mpi():
    """
    Locate the MPI environment on the system
    Returns a dict with key 'include'.
    Gives the absolute path of the openmpi include directory.
    """

    # First check if the openmpi env variable is in use
    if 'openmpi' in os.environ:
        home = os.environ['openmpi']
        openmpi = pjoin(home, 'include')
    else:
        # Otherwise, search /usr/include/ for either openmpi or x86_64-linux-gnu/openmpi (Debian-based distros)
        openmpi = find_in_path('openmpi', '/usr/include/')
        if not openmpi:
            openmpi = find_in_path('openmpi', '/usr/include/x86_64-linux-gnu/')
        if openmpi is None:
            raise EnvironmentError('The openmpi binary could not be '
                                   'located in your $PATH. Either add it to your path, '
                                   'or set $openmpi')

    openmpi_config = {'include': openmpi}
    if not os.path.exists(openmpi):
        raise EnvironmentError('The openmpi path could not be located in %s' % openmpi)
    return openmpi_config


MPI = locate_mpi()

# Include paths
_config_include_path = "distributed_compy/_include/_config/"
_lib_include_path = "distributed_compy/_include/_lib/"
_util_include_path = "distributed_compy/_include/_util/"

# Manually set compiler options -> enables compilation of MPI
os.environ["CC"] = "mpicc"
os.environ["CXX"] = "mpic++"

# Utility source file paths
getfilelen_cpp_source = "distributed_compy/_util/GetFileLen.cpp"
dataextractor_cpp_source = "distributed_compy/_util/DataExtractor.cpp"
loadbalancer_cpp_source = "distributed_compy/_util/LoadBalancer.cpp"

naive_sum_pyx_source = "distributed_compy/_lib/_naive_sum/_naive_sum"+wrapper_file_ext
naive_sum_cpp_source = "distributed_compy/_lib/_naive_sum/NaiveSum.cpp"
naive_sum_extension = Extension(name="distributed_compy._lib._naive_sum.lib._naive_sum",
                                sources=[naive_sum_pyx_source, naive_sum_cpp_source],
                                extra_compile_args={
                                    'gcc': ['-ffast-math', '-O3', '-march=native', '-mtune=native',
                                                    '-mfpmath=sse'],
                                    'nvcc': [],
                                },
                                include_dirs=[_lib_include_path, np.get_include()],
                                language='c++'
                                )

omp_reduction_sum_pyx_source = "distributed_compy/_lib/_omp_reduction_sum/_omp_reduction_sum"+wrapper_file_ext
omp_reduction_sum_cpp_source = "distributed_compy/_lib/_omp_reduction_sum/OMPReductionSum.cpp"
omp_reduction_extension = Extension(name="distributed_compy._lib._omp_reduction_sum.lib._omp_reduction_sum",
                                    sources=[omp_reduction_sum_pyx_source, omp_reduction_sum_cpp_source],
                                    extra_compile_args={
                                        'gcc': ['-fopenmp', '-ffast-math', '-O3', '-march=native', '-mtune=native',
                                                '-mfpmath=sse'],
                                        'nvcc': [],
                                    },
                                    extra_link_args=['-fopenmp'],
                                    include_dirs=[_lib_include_path, np.get_include()],
                                    language='c++'
                                    )

gpu_reduction_sum_pyx_source = "distributed_compy/_lib/_gpu_reduction_sum/_gpu_reduction_sum"+wrapper_file_ext
gpu_reduction_sum_cu_source = "distributed_compy/_lib/_gpu_reduction_sum/GPUReductionSum.cu"
gpu_reduction_extension = Extension(name="distributed_compy._lib._gpu_reduction_sum.lib._gpu_reduction_sum",
                                    sources=[gpu_reduction_sum_pyx_source, gpu_reduction_sum_cu_source],
                                    extra_compile_args={
                                        'gcc': ['-fopenmp', '-ffast-math', '-O3', '-march=native', '-mtune=native',
                                                '-mfpmath=sse'],
                                        'nvcc': ['-Xcompiler', '-fopenmp,' '--shared', '-lgomp', '-O3'],
                                    },
                                    extra_link_args=['-fopenmp'],
                                    include_dirs=[_lib_include_path, CUDA['include']],
                                    library_dirs=[CUDA['lib64']],
                                    runtime_library_dirs=[CUDA['lib64']],
                                    libraries=['gomp', 'cudart'],
                                    language='c++'
                                    )

get_local_bands_pyx_source = "distributed_compy/_config/_local/_get_local_bands"+wrapper_file_ext
get_local_bands_cpp_source = "distributed_compy/_config/_local/GetLocalBands.cpp"
get_gpu_bands_cu_source = "distributed_compy/_config/_local/GetGPUBands.cu"
get_local_bands_extension = Extension(name="distributed_compy._config._local.lib._get_local_bands",
                                      sources=[get_local_bands_pyx_source, get_local_bands_cpp_source,
                                               get_gpu_bands_cu_source],
                                      extra_compile_args={
                                          'gcc': ['-ffast-math', '-O3', '-march=native', '-mtune=native',
                                                  '-mfpmath=sse'],
                                          'nvcc': ['-Xcompiler', '--shared', '-O3'],
                                      },
                                      include_dirs=[_config_include_path, CUDA['include']],
                                      library_dirs=[CUDA['lib64']],
                                      libraries=['cudart'],
                                      language='c++'
                                      )

configure_local_pyx_source = "distributed_compy/_config/_local/_configure_local"+wrapper_file_ext
configure_local_cpp_source = "distributed_compy/_config/_local/ConfigureLocal.cpp"
configure_local_extension = Extension(name="distributed_compy._config._local.lib._configure_local",
                                      sources=[configure_local_pyx_source, get_local_bands_cpp_source,
                                               configure_local_cpp_source, get_gpu_bands_cu_source],
                                      extra_compile_args={
                                          'gcc': ['-ffast-math', '-O3', '-march=native', '-mtune=native',
                                                  '-mfpmath=sse'],
                                          'nvcc': ['-Xcompiler', '--shared', '-O3'],
                                      },
                                      include_dirs=[_config_include_path, CUDA['include']],
                                      library_dirs=[CUDA['lib64']],
                                      libraries=['cudart'],
                                      language='c++'
                                      )

configure_network_pyx_source = "distributed_compy/_config/_network/_configure_network"+wrapper_file_ext
configure_network_cpp_source = "distributed_compy/_config/_network/ConfigureNetwork.cpp"
set_local_bands_cpp_source = "distributed_compy/_config/_local/SetLocalBands.cpp"
configure_network_extension = Extension(name="distributed_compy._config._network.lib._configure_network",
                                        sources=[set_local_bands_cpp_source, configure_network_pyx_source,
                                                 get_local_bands_cpp_source, configure_network_cpp_source,
                                                 get_gpu_bands_cu_source],
                                        extra_compile_args={
                                            'gcc': ['-ffast-math', '-O3', '-march=native', '-mtune=native',
                                                    '-mfpmath=sse'],
                                            'nvcc': ['-Xcompiler', '--shared', '-O3'],
                                        },
                                        include_dirs=[_config_include_path, CUDA['include'], MPI['include']],
                                        library_dirs=[CUDA['lib64']],
                                        libraries=['cudart'],
                                        language='c++'
                                        )

hetero_reduction_sum_pyx_source = "distributed_compy/_lib/_hetero_reduction_sum/_hetero_reduction_sum"+wrapper_file_ext
hetero_reduction_sum_cpp_source = "distributed_compy/_lib/_hetero_reduction_sum/HeteroReductionSum.cpp"
get_local_bands_cpp_source = "distributed_compy/_config/_local/GetLocalBands.cpp"
hetero_reduction_extension = Extension(name="distributed_compy._lib._hetero_reduction_sum.lib._hetero_reduction_sum",
                                       sources=[get_local_bands_cpp_source, hetero_reduction_sum_pyx_source,
                                                hetero_reduction_sum_cpp_source, naive_sum_cpp_source,
                                                omp_reduction_sum_cpp_source, getfilelen_cpp_source,
                                                dataextractor_cpp_source, loadbalancer_cpp_source,
                                                get_gpu_bands_cu_source, gpu_reduction_sum_cu_source],
                                       extra_compile_args={
                                           'gcc': ['-fopenmp', '-ffast-math', '-O3', '-march=native', '-mtune=native',
                                                   '-mfpmath=sse'],
                                           'nvcc': ['-Xcompiler', '-fopenmp', '--shared', '-O3'],
                                       },
                                       extra_link_args=['-fopenmp'],
                                       include_dirs=[_lib_include_path, _util_include_path, _config_include_path,
                                                     CUDA['include']],
                                       library_dirs=[CUDA['lib64']],
                                       libraries=['gomp', 'cudart'],
                                       language='c++'
                                       )

hybrid_reduction_sum_pyx_source = "distributed_compy/_lib/_hybrid_reduction_sum/_hybrid_reduction_sum"+wrapper_file_ext
hybrid_reduction_sum_cpp_source = "distributed_compy/_lib/_hybrid_reduction_sum/HybridReductionSum.cpp"
hybrid_reduction_extension = Extension(name="distributed_compy._lib._hybrid_reduction_sum.lib._hybrid_reduction_sum",
                                       sources=[hybrid_reduction_sum_cpp_source, get_local_bands_cpp_source,
                                                hybrid_reduction_sum_pyx_source, hetero_reduction_sum_cpp_source,
                                                omp_reduction_sum_cpp_source, getfilelen_cpp_source,
                                                dataextractor_cpp_source, loadbalancer_cpp_source,
                                                get_gpu_bands_cu_source, gpu_reduction_sum_cu_source],
                                       extra_compile_args={
                                           'gcc': ['-fopenmp', '-ffast-math', '-O3', '-march=native', '-mtune=native',
                                                   '-mfpmath=sse'],
                                           'nvcc': ['-Xcompiler', '-fopenmp', '--shared', '-O3'],
                                       },
                                       extra_link_args=['-fopenmp'],
                                       include_dirs=[_lib_include_path, _util_include_path, _config_include_path,
                                                     CUDA['include'], MPI['include']],
                                       library_dirs=[CUDA['lib64']],
                                       libraries=['gomp', 'cudart'],
                                       language='c++'
                                       )


_extensions = [naive_sum_extension, omp_reduction_extension, gpu_reduction_extension, get_local_bands_extension,
               configure_local_extension, configure_network_extension, hetero_reduction_extension,
               hybrid_reduction_extension]


def customize_compiler_for_nvcc(self):
    # Tell the compiler it can process .cu
    self.src_extensions.append('.cu')

    # Save references to the default compiler_so and _comple methods
    default_compiler_so = self.compiler_so
    super = self._compile

    # Now redefine the _compile method. This gets executed for each
    # object but distutils doesn't have the ability to change compilers
    # based on source extension: we add it
    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        if os.path.splitext(src)[1] == '.cu':
            # Use the cuda for .cu files
            self.set_executable('compiler_so', CUDA['nvcc'])
            # Use only a subset of the extra_postargs, which are 1-1 translated
            # from the extra_compile_args in the Extension class
            postargs = extra_postargs['nvcc']
        else:
            postargs = extra_postargs['gcc']

        super(obj, src, ext, cc_args, postargs, pp_opts)
        # Reset the default compiler_so, which we might have changed for cuda
        self.compiler_so = default_compiler_so

    # Inject our redefined _compile method into the class
    self._compile = _compile


# Run customize_compiler
class CustomBuildExt(build_ext):
    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)



# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()


setup(
    name="distributed_compy",
    version="1.0.50",
    description=('Distributed hybrid (multi-node) heterogeneous (CPU + multi-GPU) computing library. Utilizes and '
                 'requires CUDA toolkit, OpenMP, and OpenMPI.'),
    author="nellogan",
    author_email='1nel.logan1@gmail.com',
    url='https://github.com/nellogan/distributed_compy',
    packages=find_packages(),
    setup_requires=[
        'numpy'
    ],
    install_requires=['numpy'],
    ext_modules=_extensions,
    include_dirs=[np.get_include()],
    cmdclass={'build_ext': CustomBuildExt},
    package_data={'': ['*.h', '*.pyx'],
                  'distributed_compy._config._local': ["_get_local_bands.cpp", "_configure_local.cpp"],
                  'distributed_compy._config._network': ["_configure_network.cpp"],
                  'distributed_compy._lib._naive_sum': ["_naive_sum.cpp"],
                  'distributed_compy._lib._omp_reduction_sum': ["_omp_reduction_sum.cpp"],
                  'distributed_compy._lib._gpu_reduction_sum': ["_gpu_reduction_sum.cpp"],
                  'distributed_compy._lib._hetero_reduction_sum': ["_hetero_reduction_sum.cpp"],
                  'distributed_compy._lib._hybrid_reduction_sum': ["_hybrid_reduction_sum.cpp"],
                  },
    license='MIT',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Environment :: GPU :: NVIDIA CUDA',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Intended Audience :: System Administrators',
        'Intended Audience :: End Users/Desktop',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX',
        'Programming Language :: C++',
        'Programming Language :: Cython',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering',
        'Topic :: System :: Clustering',
        'Topic :: System :: Distributed Computing',
        'Topic :: Software Development',
        'Topic :: Software Development :: Code Generators',
        'Topic :: Utilities'
    ],
    zip_safe=False,
    long_description=long_description,
    long_description_content_type='text/markdown'
)
