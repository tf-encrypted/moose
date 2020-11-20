from setuptools import find_packages, setup
from setuptools_rust import RustExtension


setup_requires = ['setuptools-rust~=0.11.5']
install_requires = ['numpy']
test_requires = install_requires + ['pytest']

setup(
    name='moose',
    version='0.1.0',
    description='Example of python extension using rust-numpy',
    rust_extensions=[RustExtension(
        'moose.moose',
        './Cargo.toml',
    )],
    install_requires=install_requires,
    setup_requires=setup_requires,
    test_requires=test_requires,
    packages=find_packages(),
    zip_safe=False,
)
