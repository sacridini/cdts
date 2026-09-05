import os
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import setuptools

class get_pybind_include(object):
    """Helper class to determine the pybind11 include path"""
    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)

ext_modules = [
    Extension(
        'cdts._core',
        ['src/main.cpp', 'src/landtrendr.cpp', 'src/ccdc.cpp', 'src/utils.cpp'],
        include_dirs=[
            get_pybind_include(),
            get_pybind_include(user=True),
            'third_party/eigen'
        ],
        language='c++'
    ),
]

def has_flag(compiler, flagname):
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.cpp') as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True

class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    c_opts = {
        'msvc': ['/EHsc'],
        'unix': [],
    }

    if sys.platform == 'darwin':
        c_opts['unix'] += ['-stdlib=libc++', '-mmacosx-version-min=10.7']

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        if ct == 'unix':
            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
            opts.append('-std=c++14')
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')
            if sys.platform != 'darwin' and has_flag(self.compiler, '-fopenmp'):
                opts.append('-fopenmp')
        elif ct == 'msvc':
            opts.append('/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())
            opts.append('/std:c++14')
            opts.append('/openmp')
        for ext in self.extensions:
            ext.extra_compile_args = opts
            if ct == 'unix' and sys.platform != 'darwin':
                ext.extra_link_args = ['-fopenmp']
        build_ext.build_extensions(self)

setup(
    name='cdts',
    version='0.2.3',
    packages=['cdts'],
    ext_modules=ext_modules,
    setup_requires=['pybind11>=2.10.0'],
    cmdclass={'build_ext': BuildExt},
    zip_safe=False,
)
