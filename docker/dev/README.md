# Local cluster with Docker Compose

While not intended for unit testing nor deployment, in some cases it can be useful to explore the runtime in a local cluster. This directory contains tools for making that easy using [Docker Compose](https://docs.docker.com/compose/). To start the cluster simply running the following from a console in this directory:

```
docker-compose up
```

Note that changes to any Python files in the project will automatically restart the workers in the cluster, although in some cases this must be done manually by restarting the process launched above.

Note also that a pre-compiled Docker image is available on Docker Hub under `tfencrypted/runtime-dev-worker:latest`, yet some changes require this image to be rebuild. One example of such is an updates to the `requirements.txt` file.

Alternatively, you can use `docker-compose up --detach` to run the cluster in detached mode, allowing you to reuse the same terminal for other things. In this case, logs from the cluster can be viewed using `docker-compose logs --follow`, and the cluster can be shut down again using `docker-compose down`.

## Rebuilding the image

Run `make build-worker-image` from this directory to rebuild the image used by the cluster. For testing you can use `make run-worker-image` to launch the image without running the worker.
