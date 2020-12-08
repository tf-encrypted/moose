from setuptools import find_packages
from setuptools import setup
from setuptools_rust import RustExtension

setup_requires = ["setuptools-rust~=0.11.5"]
install_requires = ["numpy"]
test_requires = install_requires + ["pytest", "absl-py"]

setup(
    name="moose_kernels",
    version="0.1.0",
    description="Example of python extension using rust-numpy",
    rust_extensions=[RustExtension("moose_kernels.moose_kernels", "./Cargo.toml",)],
    install_requires=install_requires,
    setup_requires=setup_requires,
    test_requires=test_requires,
    packages=find_packages(),
    zip_safe=False,
)
