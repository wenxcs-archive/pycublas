import os
import shutil
from setuptools import setup, find_packages

setup(
    name="pycublas",
    version="0.1",
    author="Wenxiang@Microsoft Research",
    packages=find_packages(exclude=["test"]),
)
