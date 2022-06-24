"""Installing with setuptools."""
import setuptools
import setuptools_rust

with open("README.md", "r", encoding="utf8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pymoose",
    version="0.2.2",  # NOTE: auto-updated during release
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
    install_requires=[],
    setup_requires=["setuptools-rust~=0.11.5"],
    rust_extensions=[
        setuptools_rust.RustExtension(
            "pymoose._rust", "./Cargo.toml", features=["extension-module"]
        ),
    ],
    zip_safe=False,
    license="Apache License 2.0",
    url="https://github.com/tf-encrypted/runtime",
    description="A Secure Runtime for Federated Learning & Encrypted Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="The TF Encrypted Authors",
    author_email="contact@tf-encrypted.io",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Development Status :: 2 - Pre-Alpha",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security :: Cryptography",
    ],
)
