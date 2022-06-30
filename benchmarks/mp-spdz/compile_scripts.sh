#!/usr/bin/env bash

./compile.py -R 128 test_dot_seq 1 1
./compile.py -R 128 test_dot_seq 1 10
./compile.py -R 128 test_dot_seq 1 100


./compile.py -R 128 test_dot_seq 10 1
./compile.py -R 128 test_dot_seq 10 10
./compile.py -R 128 test_dot_seq 10 100


./compile.py -R 128 test_dot_seq 100 1
./compile.py -R 128 test_dot_seq 100 10
./compile.py -R 128 test_dot_seq 100 100


./compile.py -R 128 test_dot_seq 1000 1
./compile.py -R 128 test_dot_seq 1000 10
./compile.py -R 128 test_dot_seq 1000 100


./compile.py -R 128 test_dot_thread 1 1 1 1
./compile.py -R 128 test_dot_thread 1 10 1 10
./compile.py -R 128 test_dot_thread 1 100 1 100


./compile.py -R 128 test_dot_thread 10 1 1 1
./compile.py -R 128 test_dot_thread 10 10 1 10
./compile.py -R 128 test_dot_thread 10 100 1 100


./compile.py -R 128 test_dot_thread 100 1 1 1
./compile.py -R 128 test_dot_thread 100 10 1 10
./compile.py -R 128 test_dot_thread 100 100 1 100

./compile.py -R 128 test_dot_thread 1000 1 1 1
./compile.py -R 128 test_dot_thread 1000 10 1 10
./compile.py -R 128 test_dot_thread 1000 100 1 100
