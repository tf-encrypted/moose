# Calling Python functions from Moose

This example shows you how you can call python functions on cleartext within Moose.

We recommend running the example using the included docker-compose.yaml file to start up the needed workers, and docker for running `main.py`. To do so, first run the following from this directory to start up the required workers:

```
docker-compose up --build
```

Then run the following to execute `main.py` against there:

```
make run
```

To stop the workers again run `docker-compose down`.
