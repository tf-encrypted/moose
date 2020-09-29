
WORKER_IMAGE?=tfencrypted/runtime-dev-worker:latest

build-worker-image:
	docker build \
		--file Dockerfile.worker \
		--tag ${WORKER_IMAGE} .

run-worker-image:
	docker run -it \
		--env PORT=50000 \
		--volume $(realpath ../..):/runtime \
		${WORKER_IMAGE} \
		bash

.PHONY: build-worker-image run-worker-image
