# Calling Python Functions from Moose

This example shows you how you can call Python functions within Moose.

We recommend running the example using the included Docker Compose file to start up the needed workers, and docker for running `main.py`. To do so, first run the following from this directory to start up the required workers:

TODO can be run very simply with `python main.py --runtime test --verbose` without networking


TODO `certstrap` and `make certs`

```
docker-compose up --build
```

Then run the following to execute `main.py` against there:

```
make run
```

To stop the workers again run `docker-compose down`.
