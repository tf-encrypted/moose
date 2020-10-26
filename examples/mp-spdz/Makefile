MAIN_IMAGE?=tfencrypted/runtime-mpspdz-worker:latest

run:
	docker run \
		--tty \
		--network mp-spdz_default \
		--volume $(realpath ../../moose):/moose \
		--volume $(realpath .):/examples/mp-spdz \
		${MAIN_IMAGE} \
		sh -c "python3 /examples/mp-spdz/main.py --runtime remote --verbose --cluster-spec /examples/mp-spdz/cluster-spec.yaml"

test-localhost:
	docker run \
		--tty \
		--volume $(realpath ../../moose):/moose \
		--volume $(realpath .):/examples/mp-spdz \
		${MAIN_IMAGE} \
		sh -c "cp /examples/mp-spdz/elk-to-mpc . && python3 /examples/mp-spdz/main.py --runtime test --verbose "

.PHONY: run test-localhost
