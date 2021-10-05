from setuptools import setup
from Cython.Build import cythonize

setup(
    name='small utils',
    ext_modules=cythonize("processing_utils.pyx"),
    zip_safe=False,
)
