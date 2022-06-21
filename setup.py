import os
from setuptools import setup, find_packages
import subprocess
import logging

PACKAGE_NAME = 'dcarte_transform'

setup(
    name=PACKAGE_NAME,
    version='0.0.1',
    description="A package that can be used alongside DCARTE"\
                "that adds extra functionality for engineered features and machine learning.",
    author='Alex Capstick and Francesca Palermo',
    url='',
    packages=[PACKAGE_NAME,],
    long_description=open('README.txt').read(),
    install_requires=[
                        "numpy>=1.22",
                        "pandas>=1.4",
                        "matplotlib>=3.5",
                        "dcarte>=0.3",
                        "scikit-learn>=1.0",
                        "pydtmc>=6.10",
                        "tqdm>=4.64",
                        "tensorboard>=2.9",
                        "pytorch-lightning>=1.6",
                        "pandarallel>=1.6",
                        "aml @ git+https://github.com/alexcapstick/AML",
                        "sku @ git+https://github.com/alexcapstick/SKPipelineUtils"
    ]
)