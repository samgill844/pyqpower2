#from future.utils import iteritems
from __future__ import print_function
import  os
from os.path import join as pjoin
from setuptools import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import subprocess
import numpy
from distutils.core import setup
from distutils.extension import Extension
import numpy as np
from distutils.ccompiler import new_compiler
import os
import sys
import tempfile




################################################################################################
# ALL THIS FROM
# https://github.com/rmcgibbo/npcuda-example/blob/master/cython/src/manager.hh
################################################################################################


def find_in_path(name, path):
    "Find a file in a search path"
    #adapted fom http://code.activestate.com/recipes/52224-find-a-file-given-a-search-path/
    for dir in path.split(os.pathsep):
        binpath = pjoin(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None


def locate_cuda():
    """Locate the CUDA environment on the system

    Returns a dict with keys 'home', 'nvcc', 'include', and 'lib64'
    and values giving the absolute path to each directory.

    Starts by looking for the CUDAHOME env variable. If not found, everything
    is based on finding 'nvcc' in the PATH.
    """

    # first check if the CUDAHOME env variable is in use
    if 'CUDAHOME' in os.environ:
        home = os.environ['CUDAHOME']
        nvcc = pjoin(home, 'bin', 'nvcc')
    else:
        # otherwise, search the PATH for NVCC
        nvcc = find_in_path('nvcc', os.environ['PATH'])
        if nvcc is None:
            raise EnvironmentError('The nvcc binary could not be '
                'located in your $PATH. Either add it to your path, or set $CUDAHOME')
        home = os.path.dirname(os.path.dirname(nvcc))

    cudaconfig = {'home':home, 'nvcc':nvcc,
                  'include': pjoin(home, 'include'),
                  'lib64': pjoin(home, 'lib64')}
    for k, v in iter(cudaconfig.items()):
        if not os.path.exists(v):
            raise EnvironmentError('The CUDA %s path could not be located in %s' % (k, v))

    return cudaconfig


def customize_compiler_for_nvcc(self):
    """inject deep into distutils to customize how the dispatch
    to gcc/nvcc works.

    If you subclass UnixCCompiler, it's not trivial to get your subclass
    injected in, and still have the right customizations (i.e.
    distutils.sysconfig.customize_compiler) run on it. So instead of going
    the OO route, I have this. Note, it's kindof like a wierd functional
    subclassing going on."""

    # tell the compiler it can processes .cu
    self.src_extensions.append('.cu')

    # save references to the default compiler_so and _comple methods
    default_compiler_so = self.compiler_so
    super = self._compile

    # now redefine the _compile method. This gets executed for each
    # object but distutils doesn't have the ability to change compilers
    # based on source extension: we add it.
    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        if os.path.splitext(src)[1] == '.cu':
            # use the cuda for .cu files
            self.set_executable('compiler_so', CUDA['nvcc'])
            # use only a subset of the extra_postargs, which are 1-1 translated
            # from the extra_compile_args in the Extension class
            postargs = extra_postargs['nvcc']
            postargs.append('-lgsl')
            postargs.append('-lgslcblas')
        else:
            postargs = extra_postargs['gcc']

        super(obj, src, ext, cc_args, postargs, pp_opts)
        # reset the default compiler_so, which we might have changed for cuda
        self.compiler_so = default_compiler_so

    # inject our redefined _compile method into the class
    self._compile = _compile


# run the customize_compiler
class custom_build_ext(build_ext):
    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)



CUDA = locate_cuda()

# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()







########################################
# NOW FOR OPEN MP
#######################################
def detect_openmp():
    """Does this compiler support OpenMP parallelization?"""
    compiler = new_compiler()
    print("Checking for OpenMP support... ")
    hasopenmp = hasfunction(compiler, 'omp_get_num_threads()')
    needs_gomp = hasopenmp
    if not hasopenmp:
        compiler.add_library('gomp')
    hasopenmp = hasfunction(compiler, 'omp_get_num_threads()')
    needs_gomp = hasopenmp
    if hasopenmp: print("Compiler supports OpenMP")
    else: print( "Did not detect OpenMP support.")
    return hasopenmp, needs_gomp

def hasfunction(cc, funcname, include=None, extra_postargs=None):
    # From http://stackoverflow.com/questions/
    #            7018879/disabling-output-when-compiling-with-distutils
    tmpdir = tempfile.mkdtemp(prefix='hasfunction-')
    devnull = oldstderr = None
    try:
        try:
            fname = os.path.join(tmpdir, 'funcname.c')
            f = open(fname, 'w')
            if include is not None:
                f.write('#include %s\n' % include)
            f.write('int main(void) {\n')
            f.write('    %s;\n' % funcname)
            f.write('}\n')
            f.close()
            # Redirect stderr to /dev/null to hide any error messages
            # from the compiler.
            # This will have to be changed if we ever have to check
            # for a function on Windows.
            devnull = open('/dev/null', 'w')
            oldstderr = os.dup(sys.stderr.fileno())
            os.dup2(devnull.fileno(), sys.stderr.fileno())
            objects = cc.compile([fname], output_dir=tmpdir, extra_postargs=extra_postargs)
            cc.link_executable(objects, os.path.join(tmpdir, "a.out"))
        except Exception as e:
            return False
        return True
    finally:
        if oldstderr is not None:
            os.dup2(oldstderr, sys.stderr.fileno())
        if devnull is not None:
            devnull.close()

#checks whether OpenMP is supported
has_openmp, needs_gomp = detect_openmp()
parallel_args = ['-fopenmp', '-std=c99'] if has_openmp else ['-std=c99']
parallel_libraries = ['gomp'] if needs_gomp else []



nvcc_extra_compile_args = ['-arch=sm_35', '-O3', '--ptxas-options=-v', '-c', '--compiler-options', "'-fPIC'", '--expt-relaxed-constexpr', '-Xcompiler', '-fopenmp'] if has_openmp else ['-arch=sm_35', '-O3', '-arch=sm_30', '--ptxas-options=-v', '-c', '--compiler-options', "'-fPIC'", '--expt-relaxed-constexpr']
nvcc_libraries=['cudart', 'm', 'gsl', 'gslcblas', 'gomp'] if has_openmp else ['cudart', 'm', 'gsl', 'gslcblas']

ext = Extension('qpower2',
                sources=['c_src/qpower2.cu', 'gpuastro.pyx'],
                library_dirs=[CUDA['lib64'], "c_src/"],
                #libraries=['cudart', 'm', 'gsl', 'gslcblas', 'gomp'],
                libraries=nvcc_libraries,
                language='c++',
                runtime_library_dirs=[CUDA['lib64']],
                # this syntax is specific to this build system
                # we're only going to use certain compiler args with nvcc and not with gcc
                # the implementation of this trick is in customize_compiler() below
                extra_compile_args={'gcc': [],
                                    'nvcc': nvcc_extra_compile_args},
                include_dirs = [numpy_include, CUDA['include'], 'src'])




setup(name='qpower2',
      # random metadata. there's more you can supploy
      author='Samuel Gill',
      version='0.1',

      ext_modules = [ext],

      # inject our custom trigger
      cmdclass={'build_ext': custom_build_ext},

      # since the package has c code, the egg cannot be zipped
      zip_safe=False)
