MAIN_IMAGE?=tfencrypted/runtime-base-worker:latest

run:
	docker run \
		--tty \
		--network mp-spdz_default \
		--volume $(realpath ../../moose):/moose \
		--volume $(realpath .):/examples/mp-spdz \
		${MAIN_IMAGE} \
		sh -c "python3 /examples/mp-spdz/main.py --verbose --cluster-spec /examples/mp-spdz/cluster-spec.yaml"

test-localhost:
	docker run \
		--tty \
		--volume $(realpath ../../moose):/moose \
		--volume $(realpath .):/examples/mp-spdz \
		${MAIN_IMAGE} \
		sh -c "python3 /examples/mp-spdz/main.py --runtime test --verbose "

.PHONY: run
