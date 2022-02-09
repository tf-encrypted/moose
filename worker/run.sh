#!/bin/bash

set -e

if [[ -z $1 ]]; then
    echo "first argument must be placement, e.g., {player0, player1, player2}"
    exit 1
fi

cargo run -- \
    --hosts "$(cat hosts.json)" \
    --data data.csv \
    --placement $1 \
    --comp "comp.bytes" \
    --role-assignment "$(cat role_assignment.json)" \
    --session-id 1234
