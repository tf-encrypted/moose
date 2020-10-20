build-base-worker-image:
	docker build \
		--file Dockerfile.base-worker \
		--tag tfencrypted/runtime-base-worker:latest --no-cache \
		..

build-mpspdz-worker-image:
	docker build \
		--file Dockerfile.mpspdz-worker \
		--tag tfencrypted/runtime-mpspdz-worker:latest --no-cache \
		..

push-mpspdz-worker-image:
	docker push tfencrypted/runtime-mpspdz-worker


.PHONY: build-base-worker-image build-mpspdz-worker-image
