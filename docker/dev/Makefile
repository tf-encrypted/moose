build-worker-image:
	docker build --file Dockerfile.worker --tag tfencrypted/runtime-dev-worker ../../

run-worker-image:
	docker run -it --env PORT=50000 --volume $(realpath ../..):/runtime tfencrypted/runtime-dev-worker:latest

.PHONY: build-worker-image run-worker-image

