from setuptools import setup, find_packages
from opt_greeks import __version__

setup(
    name='opt_greeks',
    version=__version__,
    packages=find_packages(),
    install_requires=['scipy', 'numpy'],  # Add any dependencies here
)

