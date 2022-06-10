# Elk

Elk is a thin CLI wrapper over the Moose.

## Installation

Elk may be installed as a binary `elk` as follows:

```sh
cargo install --bin elk
```

Alternatively, Elk may also be invoked using `cargo run`, which is particularly useful during development:

```sh
cargo run --bin elk --
```

## Commands

To use Elk as a compiler:

```sh
elk compile in.moose out.moose
```

To use Elk to collect (static) statistics about a computation:

```sh
# Histogram by operation kinds
elk stats op_hist input.moose

# The same, but by placement
elk stats op_hist input.moose --by-placement

# Total count of operations
elk stats op_count input.moose
elk stats -b op_count input.moose
```
