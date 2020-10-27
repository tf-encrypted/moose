# Calling Python Functions from Moose

This example shows how you can use Moose to call Python functions on workers.

We recommend running the example using Docker Compose by executing the following commands from this directory.

First we must generate certificates for all participants (note that you must have [certstrap](https://github.com/square/certstrap) installed):

```sh
make certs
```

Then we can spin up a local cluster of containers using:

```sh
make up
```

This will block the current terminal, so launch a new one to execute the remaining commands.

To execute a computation on the cluster we can finally run the following, which may be done repeatedly without re-running the steps above:

```sh
make run
```

Once we a done we can shut down the cluster again:

```
make down
```
