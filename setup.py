from setuptools import setup, find_packages

setup(
    name='gs_interpolation',
    version='0.1.0',
    description='A Gauss-Seidel interpolation method for 3D arrays with NaN values.',
    author='Jack Atkinson',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
    ],
)