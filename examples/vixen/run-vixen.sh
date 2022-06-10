#!/bin/bash

set -e

if [[ -z $1 ]]; then
    echo "first argument must be placement, e.g., {player0, player1, player2}"
    exit 1
fi

cargo run --bin vixen --release -- \
    --hosts "$(cat examples/vixen/hosts.json)" \
    --placement $1 \
    --comp "examples/vixen/comp.bytes" \
    --role-assignment "$(cat examples/vixen/role_assignment.json)" \
    --session-id 1234
