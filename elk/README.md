# Elk

Elk is a thin CLI wrapper over the Moose.

### Compiling computations

To use Elk as a compiler run it like this:

```
cargo run --bin elk -- compile --passes=typing,networking,toposort input.moose out.moose
```

The list of passes to execute is optional.

### Computation stats

To use Elk to examine a computation use the stats command like this:

```
# Histogram by operation kinds
cargo run --bin elk -- stats op_hist input.moose

# The same, but by placement
cargo run --bin elk -- stats op_hist input.moose --by-placement

# Total count of operations
cargo run --bin elk -- stats op_count input.moose
cargo run --bin elk -- stats -b op_count input.moose
```
=======
## Installation

Run the following to install as the `elk` CLI binary:

```
cargo install --path elk
```
