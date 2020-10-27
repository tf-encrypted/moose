BASE_WORKER_IMAGE?=tfencrypted/runtime-base-worker:latest

build-base-worker-image:
	docker build \
		--file ./base/Dockerfile.worker \
		--tag ${BASE_WORKER_IMAGE} \
		--no-cache \
		..

push-base-worker-image:
	docker push ${BASE_WORKER_IMAGE}

.PHONY: build-base-worker-image push-base-worker-image


MPSPDZ_WORKER_IMAGE?=tfencrypted/runtime-mpspdz-worker:latest

build-mpspdz-worker-image:
	docker build \
		--file ./mpspdz/Dockerfile.worker \
		--tag ${MPSPDZ_WORKER_IMAGE} \
		--no-cache \
		..

push-mpspdz-worker-image:
	docker push ${MPSPDZ_WORKER_IMAGE}

.PHONY: build-mpspdz-worker-image push-mpspdz-worker-image


build: build-base-worker-image build-mpspdz-worker-image

push: push-base-worker-image push-mpspdz-worker-image

.PHONY: build push
