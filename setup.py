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
    author='Francesca Palermo Alex Capstick',
    url='',
    packages=[PACKAGE_NAME,],
    long_description=open('README.txt').read(),
    install_requires=[
                        "numpy>=1.22",
                        "pandas>=1.4",
                        "matplotlib>=3.5.1",
                        "dcarte>=0.3",
                        "scikit-learn>=1.1.1",
    ]
)