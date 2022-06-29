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

Timings given in seconds.

| MP-SPDZ/`moose` | 1x1 (tensor size)      | 10x10    | 100x100  | 1000x1000 |
| ------- | -------- | -------- | -------- | --------- |
| 1 (# dot products)       | 0.0006<br/>0.0039  | 0.0005<br/>0.0027     | 0.0322<br/>0.102    | 43.710<br/>5.910     |
| 10      | 0.001<br/>0.017    | 0.0016<br/>0.0180     | 0.248<br/>0.717   | 430.590<br/>54.588   |
| 100     | 0.006<br/>0.099    | 0.0116<br/>0.1232     | 2.422<br/>0.675  | 4305.900<br/>545.675|



## Parallel computations

Timings given in seconds.

| MP-SPDZ/`moose` | 1x1 (tensor size)    | 10x10    | 100x100  | 1000x1000 |
| ------- | -------- | -------- | -------- | --------- |
| 1 (#dot products)       | 0.001<br/>0.039    | 0.0005<br/>0.004    | 0.031<br/>0.006  | 43.158<br/>5.844      |
| 10      | 0.003<br/>0.010    | 3.152<br/>0.0107    | 0.051<br/>0.016  | 43.679<br/>11.110     |
| 100     | 0.032<br/>0.041    | 33.06<br/>0.066     | 0.219<br/>0.135  | 175.369<br>163.098    |



Logistic Regression
=====

Timings given in seconds.

| MP-SPDZ/`moose` | 128 (batch size)    | 512    | 1024  | 2048 |
| ------- | -------- | -------- | -------- | --------- |
| 10 (iterations) | 0.450<br/>1.316    | 0.902<br/>1.981    | 1.509<br/>2.963  | 2.817<br/>4.730   |
| 50              | 2.155<br/>7.091    | 4.395<br/>10.134   | 7.318<br/>15.033  | 13.207<br/>24.266    |
| 100             | 4.255<br/>14.385   | 8.795<br/>20.819   | 14.645<br/>31.017  | 26.333<br/>63.100    |

When the execution time is relatively low (less than 5 minutes) we noticed a
relatively high variance between experiments for `moose`. For completeness we
add a table for the logistic regression where we give the minimum, maximum,
variance and mean across 3 experiments on various batch sizes and number of
iterations:


| `moose` | 128 (batch size) Min/Max/Variance/Mean | 512    | 1024  | 2048 |
| ------- | -------- | -------- | -------- | --------- |
| 10 (iterations) |  1.316/3.642/1.034/2.424   | 1.981/4.175/1.567/3.426    | 2.963/4.087/0.412/3.346 | 4.730/7.393/2.333/5.629   |
| 50              | 7.091/19.882/51.646/15.379      | 10.134/22.109/46.996/14.194     | 15.033/28.497/59.992/19.554  | 24.266/30.518/12.867/26.376 |
| 100             | 14.385/37.705/135.967/26.000    | 20.641/48.555/259.720/29.946    | 31.105/58.850/253.585/49.492 | 63.100/75.775/47.454/67.879 |



# How to reproduce the benchmarks

The software versions used throughout the benchmarks
are [moosev0.2.2](https://github.com/tf-encrypted/moose/releases/tag/v0.2.2)
and [MP-SPDZv0.3.2](https://github.com/data61/MP-SPDZ/releases/tag/v0.3.2).

## Moose

To benchmark the sequential dot products benchmarked in `moose` we used
`dot_product.py` file that is in the `pymoose` directory. For this we need to
fire up three instances of `comet` (one on each computer). For simplicity consider we do this on localhost.

```
comet --identity localhost:50000 --port 50000
comet --identity localhost:50001 --port 50001
comet --identity localhost:50002 --port 50002
```

In order to run
100 parallel dot products whith matrices of size `1000x1000` we used the following command:

```
python benchmarks/pymoose/dot_product.py -c parallel --s 1000 --c_arg 100
````

To run 100 dot products in sequence we used the following command:
```
python benchmarks/pymoose/dot_product.py -c seq --s 1000 --c_arg 100
```

To run 50 iterations of logistic regression training with a batch size of `128`
and iterations we used the following command:

```
python pymoose/examples/benchmarks/logreg.py --batch_size 128 --n_iter 50
```

## MP-SPDZ

First download MP-SPDZ from [here](https://github.com/data61/MP-SPDZ/releases/tag/v0.3.2). Suppose this was downloaded inside a folder called `MP-SPDZ`.
Then copy the contents of the mp-spdz folder in the following manner:
```
cd ~/MP-SPDZ
cp ~/moose/benchmarks/mp-spdz/*.sh .
cp ~/moose/benchmarks/mp-spdz/*.mpc Programs/Source/
```
First compile the programs using `compile_scripts.sh` on each machine
and then execute the `run_scripts_p0.sh` on party 0, `run_scripts_p1.sh` on party
1 and `run_scripts_p2.sh` on party 2 and IP addresses accordingly in the HOSTS file inside MP-SPDZ.

To compile logistic regression in MP-SPDZ[https://github.com/data61/MP-SPDZ/blob/master/Programs/Source/logreg.mpc] with feature size of `100`, batch size
of `128`, `50` iterations and `36` threads we ran the following command:

```
/compile.py -R 128 logreg 100 128 50 36
```