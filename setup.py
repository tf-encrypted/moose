"""Installing with setuptools."""
import setuptools

with open("README.md", "r", encoding="utf8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="moose",
    version="0.1.1",  # NOTE: auto-updated during release
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
    install_requires=[],
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
