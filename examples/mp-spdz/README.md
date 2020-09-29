# Calling MP-SPDZ from Moose

This example shows you how you can use MP-SPDZ as an external system for secure computation directly from Moose.

We recommend running the example using the included docker-compose.yaml file to start up the needed workers, and docker for running `main.py`. To do so, first run the following from this directory to start up the required workers:

```
docker-compose up --build
```

Then run the following to execute `main.py` against there:

```
make run
```

To stop the workers again run `docker-compose down`.
