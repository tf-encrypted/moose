Pymoose: Python bindings to the Elk compiler and Moose Runtime
===============

### Installation & Testing

```
pip install -r requirements-dev.txt
pip install -e .
pytest .
```

Note that to run tests with `cargo` you should specify `--no-default-features` due to PyO3:

```
cargo test --no-default-features
```

### Usage

Usage examples live in the `examples/` directory.
