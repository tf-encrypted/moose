#!/bin/bash

set -e

cargo fmt
cargo run -- \
    --hosts "$(cat hosts.json)" \
    --data data.csv \
    --placement "player0" \
    --comp "comp.bytes"
