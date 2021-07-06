from setuptools import find_packages
from setuptools import setup
from setuptools_rust import RustExtension

setup_requires = ["setuptools-rust~=0.11.5"]
install_requires = ["numpy"]
test_requires = install_requires + ["pytest", "absl-py"]

setup(
    name="pymoose",
    version="0.1.2-alpha.0",  # NOTE: auto-updated during release
    description="Python-bindings for Moose",
    rust_extensions=[
        RustExtension("pymoose.moose_kernels", "./Cargo.toml"),
        RustExtension("pymoose.moose_runtime", "./Cargo.toml"),
        RustExtension("pymoose.moose_compiler", "./Cargo.toml"),
    ],
    install_requires=install_requires,
    setup_requires=setup_requires,
    test_requires=test_requires,
    packages=find_packages(),
    zip_safe=False,
)
