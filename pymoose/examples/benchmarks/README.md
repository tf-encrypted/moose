# Setup

We benchmark `moose` against `MP-SPDZ` using the arithmetic replicated secret
sharing protocol for three parties where all the computations are done in a
`128` bit ring.  Moose runs by default with replicated secret sharing whereas in
`MP-SPDZ` this can be found as the `replicated-ring-party.x` executable.

All tests use 4 machines - one that acts as a coordinator, i.e. compiling and
sending the computation to the others while the rest of the three machines
execute the protocol. Each machine is an `c5.9xlarge` AWS instance with 36 vCPUs
connected in the same region.

# Performance

Latency for matrix multiplication - time in seconds for evaluating a single dot product
node with various tensor sizes.

|         | 1x1      | 10x10    | 100x100  | 1000x1000 |
| ------- | -------- | -------- | -------- | --------- |
| MP-SPDZ | 0.0005      | 0.0006      | 0.031    | 43.00     |
| `moose` | 0.004      | 0.004     | 0.007    | 5.444     |


Throughput (matrix multiplications per second). This number is computed by
varying the `n_parallel` argument for the `dot_product` computation which adds
`n_parallel` dot product nodes to the graph that can be evaluated in parallel.
In the table below we compute the maximum throughput where the number of dot
products in parallel vary through {1, 10, 100, 1000, 10000}.

Note that as the tensor size grows then the maximum throughput is achieved
with a smaller number of nodes evaluated in parallel. This is because the
physical computation grows and the tokio tasks spawned have a larger memory and
compute footprint depleting the AWS instances from available resources.

|         | 1x1   | 10x10    | 100x100  | 1000x1000 |
| ------- | -------- | -------- | -------- | --------- |
| MP-SPDZ | 226598      | 137107      | 774    | 0.63     |
| `moose` | 3047      | 1434      | 729    | 0.71     |


# How to reproduce the benchmarks

The software versions used throughout the benchmarks are [moose v0.2.2](https://github.com/tf-encrypted/moose/releases/tag/v0.2.2) and [MP-SPDZ v0.3.2](https://github.com/data61/MP-SPDZ/releases/tag/v0.3.2).


## MP-SPDZ

First create a file `test_dot.mpc` which contains the following

```
from Compiler.library import print_ln_to

n = int(program.args[1])

a = sint.Matrix(n, n)
b = sint.Matrix(n, n)

a.input_from(0)
b.input_from(1)

c = a * b

for i in range(n):
    print_ln_to(2, "%s", c[i].reveal_to(2))
```

python pymoose/examples/benchmarks/dot_product.py --shape 1000 1000 --n 10 --n_parallel 10  13973.8578000 ms
python pymoose/examples/benchmarks/dot_product.py --shape 1000 1000 --n 3 --n_parallel 100 208302.7673333 ms

python pymoose/examples/benchmarks/dot_product.py --shape 100 100 --n 10 --n_parallel 100 159.8916 ms
python pymoose/examples/benchmarks/dot_product.py --shape 100 100 --n 10 --n_parallel 1000 1332.007 ms
python pymoose/examples/benchmarks/dot_product.py --shape 100 100 --n 10 --n_parallel 1000 751.93 ops/s
python pymoose/examples/benchmarks/dot_product.py --shape 100 100 --n 10 --n_parallel 1000 629.26 ops/s
python pymoose/examples/benchmarks/dot_product.py --shape 100 100 --n 5 --n_parallel 10000 556.117 ops/s
python pymoose/examples/benchmarks/dot_product.py --shape 100 100 --n 10 --n_parallel 1000 729.13 ops/s


python pymoose/examples/benchmarks/dot_product.py --shape 10 10 --n 10 --n_parallel 1000 1271.97 ops/s
python pymoose/examples/benchmarks/dot_product.py --shape 10 10 --n 10 --n_parallel 10000 1434.58 ops/s
python pymoose/examples/benchmarks/dot_product.py --shape 1 1 --n 10 --n_parallel 10000  3047.16 ops/s

/bin/Linux-amd64/replicated-ring-party.x -p 0 test_dot_thread-1-36-360000 -ip HOSTS 1.58871
./bin/Linux-amd64/replicated-ring-party.x -p 0 test_dot_thread-10-36-360000 -ip HOSTS 2.62567
./bin/Linux-amd64/replicated-ring-party.x -p 0 test_dot_thread-100-36-972 -ip HOSTS 1.32536
./bin/Linux-amd64/replicated-ring-party.x -p 0 test_dot_thread-100-36-9972 -ip HOSTS 12.879s
/bin/Linux-amd64/replicated-ring-party.x -p 0 test_dot_thread-1000-36-360 -ip HOSTS 566.479

/bin/Linux-amd64/replicated-ring-party.x -p 0 test_dot-100 -ip HOSTS
Time = 0.0317824 seconds

./bin/Linux-amd64/replicated-ring-party.x -p 0 test_dot-10 -ip HOSTS
Time = 0.0006443 seconds

./bin/Linux-amd64/replicated-ring-party.x -p 0 test_dot-1 -ip HOSTS
Time = 0.000551438 second

TODO: add instructions needed to run moose, version number (same with MP-SPDZ).
Add mpc file that were needed for MP-SPDZ.