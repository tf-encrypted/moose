# Changelog

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
