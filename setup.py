import os
from setuptools import setup, find_packages
import subprocess
import logging
from dcarte_transform import __version__, __doc__, __author__, __title__

setup(
    name=__title__,
    version=__version__,
    description=__doc__,
    author=__author__,
    url='',
    packages=find_packages(),
    long_description=open('README.txt').read(),
    install_requires=[
        "numpy>=1.22",
        "pandas>=1.4",
        "matplotlib>=3.5",
        "dcarte>=0.3",
        "scikit-learn>=1.0",
        "tqdm>=4.64",
    ]
)