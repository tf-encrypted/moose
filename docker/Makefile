build-base-worker-image:
	docker build \
		--file ./base/Dockerfile.worker \
		--tag tfencrypted/runtime-base-worker:latest \
		--no-cache \
		..

.PHONY: build-base-worker-image

build-mpspdz-worker-image:
	docker build \
		--file ./mpspdz/Dockerfile.worker \
		--tag tfencrypted/runtime-mpspdz-worker:latest \
		--no-cache \
		..

push-mpspdz-worker-image:
	docker push tfencrypted/runtime-mpspdz-worker:latest

.PHONY: build-mpspdz-worker-image push-mpspdz-worker-image

build: build-base-worker-image build-mpspdz-worker-image

.PHONY: build
