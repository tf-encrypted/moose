# Changelog

## v0.1.6
- Memory usage optimizations and reduced cloning
- Wellformed compiler pass to statically verify a computation with elk: elk compile -p wellformed comp.moose
- TLS support for gRCP networking
- PyTorch neural networks support
- Elk now supports all three computation formats (textual, msgpack, and bincode)
- Ensures tensors are in “standard layout” (i.e. contiguous & row-major)
- Added local file storage to moose modules (reading csv and numpy data files supported)
- start_server convenance method added to gRPC network plugin

## v0.1.5

- Significant RAM usage improvements throughout the codebase
- Add Multilayer Perceptron (MLP), a type of neural network to pymoose predictors
- `PrimDeriveSeed` operator renamed to `DeriveSeed`
- `PrimPrfKeyGen` operator renamed to `PrfKeyGen`
- Added `Host` prefix to `Seed`, `PrfKey` and `Unit`
- Unified `MeanOp` and fuse `RepFixedpointMeanOp` with `RingFixedpointMeanOp`
- Using BitVec as the data storage for HostBitTensor
- Replace sodium oxide with thread_rng and blake3
- Fix binary LinearClassifier logic

### Other changes

- Uniform textual format for rendezvous and sync keys

## v0.1.4

- A bugfix to produce less verbose errors

## v0.1.3

- Upgraded nom to 7.1 to deal with the floating point parser problems
- Switched base array from ndarray::Array to ndarray:ArcArray
- Deprecated `to_bytes/from_bytes` in favour fo `to_msgpack/from_msgpack`.

## v0.1.2

### Release notes

This is the first "stable" release with a complete set of public APIs.
Future releases are expected to contain instructions for migration from the previous "stable" version.

### Changed

- Symbolic compilation support
- Sufficient primitives to execute AES decryption
- Sufficient primitives to execute custom models, including linear regression and XGBoost decision trees.

### Fixed

- NA
