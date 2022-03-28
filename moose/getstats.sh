#!usr/bin/env bash

file=$(cat modules.txt)
    for line in $file
    do
        printf "$line " >> log.txt && cargo expand "$line" | wc -l >> log.txt
        printf "\n"
    done
