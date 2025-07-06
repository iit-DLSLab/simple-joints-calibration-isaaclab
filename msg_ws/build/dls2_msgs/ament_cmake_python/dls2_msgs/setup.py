from setuptools import find_packages
from setuptools import setup

setup(
    name='dls2_msgs',
    version='0.0.0',
    packages=find_packages(
        include=('dls2_msgs', 'dls2_msgs.*')),
)
