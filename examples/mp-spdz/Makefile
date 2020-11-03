MAIN_IMAGE?=tfencrypted/runtime-mpspdz-worker:latest

run:
	docker run \
		--tty \
		--network mp-spdz_default \
		--volume $(realpath ../../moose):/moose \
		--volume $(realpath .):/example \
		${MAIN_IMAGE} \
		sh -c "cd /example; python3 main.py --runtime remote --verbose"

test-localhost:
	docker run \
		--tty \
		--volume $(realpath ../../moose):/moose \
		--volume $(realpath .):/example \
		${MAIN_IMAGE} \
		sh -c "cd /example; python3 main.py --runtime test --verbose"

.PHONY: run test-localhost
