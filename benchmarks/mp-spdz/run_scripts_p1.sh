#!/usr/bin/env bash

echo "test_dot_seq_1-1" >> log 2>&1
./bin/Linux-amd64/replicated-ring-party.x -p 1 test_dot_seq-1-1 -pn 50001 -ip HOSTS >> log 2>&1

echo "test_dot_seq_1-10" >> log 2>&1
./bin/Linux-amd64/replicated-ring-party.x -p 1 test_dot_seq-1-10 -pn 50002 -ip HOSTS >> log 2>&1

echo "test_dot_seq_1-100" >> log 2>&1
./bin/Linux-amd64/replicated-ring-party.x -p 1 test_dot_seq-1-100 -pn 50003 -ip HOSTS >> log 2>&1

echo "test_dot_seq_10-1" >> log 2>&1
./bin/Linux-amd64/replicated-ring-party.x -p 1 test_dot_seq-10-1 -pn 50004 -ip HOSTS >> log 2>&1

echo "test_dot_seq_10-10" >> log 2>&1
./bin/Linux-amd64/replicated-ring-party.x -p 1 test_dot_seq-10-10 -pn 50005 -ip HOSTS >> log 2>&1

echo "test_dot_seq_10-100" >> log 2>&1
./bin/Linux-amd64/replicated-ring-party.x -p 1 test_dot_seq-10-100 -pn 50006 -ip HOSTS >> log 2>&1

echo "test_dot_seq_100-1" >> log 2>&1
./bin/Linux-amd64/replicated-ring-party.x -p 1 test_dot_seq-100-1 -pn 50007 -ip HOSTS >> log 2>&1

echo "test_dot_seq_100-10" >> log 2>&1
./bin/Linux-amd64/replicated-ring-party.x -p 1 test_dot_seq-100-10 -pn 50008 -ip HOSTS >> log 2>&1

echo "test_dot_seq_100-100" >> log 2>&1
./bin/Linux-amd64/replicated-ring-party.x -p 1 test_dot_seq-100-100 -pn 50009 -ip HOSTS >> log 2>&1

echo "test_dot_seq_1000-1" >> log 2>&1
./bin/Linux-amd64/replicated-ring-party.x -p 1 test_dot_seq-1000-1 -pn 50010 -ip HOSTS >> log 2>&1

echo "test_dot_seq_1000-10" >> log 2>&1
./bin/Linux-amd64/replicated-ring-party.x -p 1 test_dot_seq-1000-10 -pn 50011 -ip HOSTS >> log 2>&1

echo "test_dot_seq_1000-100" >> log 2>&1
./bin/Linux-amd64/replicated-ring-party.x -p 1 test_dot_seq-1000-100 -pn 50012 -ip HOSTS >> log 2>&1

echo "test_dot_thread-1-1-1-1" >> log 2>&1
./bin/Linux-amd64/replicated-ring-party.x -p 1 test_dot_thread-1-1-1-1 -pn 50013 -ip HOSTS >> log 2>&1

echo "test_dot_thread-1-10-1-10" >> log 2>&1
./bin/Linux-amd64/replicated-ring-party.x -p 1 test_dot_thread-1-10-1-10 -pn 50014 -ip HOSTS >> log 2>&1

echo "test_dot_thread-1-100-1-100" >> log 2>&1
./bin/Linux-amd64/replicated-ring-party.x -p 1 test_dot_thread-1-100-1-100 -pn 50015 -ip HOSTS >> log 2>&1

echo "test_dot_thread-10-1-1-1" >> log 2>&1
./bin/Linux-amd64/replicated-ring-party.x -p 1 test_dot_thread-10-1-1-1 -pn 50016 -ip HOSTS >> log 2>&1

echo "test_dot_thread-10-10-1-10" >> log 2>&1
./bin/Linux-amd64/replicated-ring-party.x -p 1 test_dot_thread-10-10-1-10 -pn 50017 -ip HOSTS >> log 2>&1

echo "test_dot_thread-10-100-1-100" >> log 2>&1
./bin/Linux-amd64/replicated-ring-party.x -p 1 test_dot_thread-10-100-1-100 -pn 50018 -ip HOSTS >> log 2>&1

echo "test_dot_thread-100-1-1-1" >> log 2>&1
./bin/Linux-amd64/replicated-ring-party.x -p 1 test_dot_thread-100-1-1-1 -pn 50019 -ip HOSTS >> log 2>&1

echo "test_dot_thread-100-10-1-10" >> log 2>&1
./bin/Linux-amd64/replicated-ring-party.x -p 1 test_dot_thread-100-10-1-10 -pn 50020 -ip HOSTS >> log 2>&1

echo "test_dot_thread-100-100-1-100" >> log 2>&1
./bin/Linux-amd64/replicated-ring-party.x -p 1 test_dot_thread-100-100-1-100 -pn 50021 -ip HOSTS >> log 2>&1

echo "test_dot_thread-1000-1-1-1" >> log 2>&1
./bin/Linux-amd64/replicated-ring-party.x -p 1 test_dot_thread-1000-1-1-1 -pn 50022 -ip HOSTS >> log 2>&1

echo "test_dot_thread-1000-10-1-10" >> log 2>&1
./bin/Linux-amd64/replicated-ring-party.x -p 1 test_dot_thread-1000-10-1-10 -pn 50023 -ip HOSTS >> log 2>&1

echo "test_dot_thread-1000-100-1-100" >> log 2>&1
./bin/Linux-amd64/replicated-ring-party.x -p 1 test_dot_thread-1000-100-1-100 -pn 50024 -ip HOSTS >> log 2>&1






