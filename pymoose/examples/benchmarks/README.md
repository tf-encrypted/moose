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

## Sequential computations

Timings given in s.

| MP-SPDZ/`moose` | 1x1      | 10x10    | 100x100  | 1000x1000 |
| ------- | -------- | -------- | -------- | --------- |
| 1       | 0.0006<br/>0.0039  | 0.0005<br/>0.0027     | 0.0322<br/>0.102    | 43.710<br/>5.910     |
| 10      | 0.001<br/>0.017    | 0.0016<br/>0.0180     | 0.248<br/>0.717   | 430.590<br/>54.588   |
| 100     | 0.006<br/>0.099    | 0.0116<br/>0.1232     | 2.422<br/>0.675  | 4305.900<br/>545.675|



## Parallel computations

Timings given in s.

| MP-SPDZ/`moose` | 1x1    | 10x10    | 100x100  | 1000x1000 |
| ------- | -------- | -------- | -------- | --------- |
| 1       | 0.001<br/>0.039    | 0.0005<br/>0.004    | 0.031<br/>0.006  | 43.158<br/>5.844      |
| 10      | 0.003<br/>0.010    | 3.152<br/>0.0107    | 0.051<br/>0.016  | 43.679<br/>11.110     |
| 100     | 0.032<br/>0.041    | 33.06<br/>0.066     | 0.219<br/>0.135  | 175.369<br>163.098    |



Logistic Regression
=====

| MP-SPDZ/`moose` | 128    | 512    | 1024  | 2048 |
| ------- | -------- | -------- | -------- | --------- |
| 10      | 3.429<br/>1.363    | 133.844<br/>1.993     | -<br/>-  | -<br/>4.773   |
| 50      | 3.534<br/>2.466    | 134.387<br/>1.989     | -<br/>-  | -<br/>4.713    |
| 100     | 3.585<br/>2.478    | 136.117<br/>3.066     | 264.812<br/>2.922  | -<br/>4.749    |

batch 128, features 100, 1280 instances, 10 epochs (i.e. 100 iterations)
- mp-spdz: 35.828s (35.758, 35.5988, 35.6241)
- moose: 14.435s (37.068, 38.875, 14.492 )

TODO

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

TODO: add instructions needed to run moose, version number (same with MP-SPDZ).
Add mpc file that were needed for MP-SPDZ.