
MAIN_IMAGE?=tfencrypted/runtime-base-worker:latest

run:
	docker run \
		--tty \
		--volume $(realpath ../../moose):/moose \
		--volume $(realpath .):/examples/python-functions \
		${MAIN_IMAGE} \
		sh -c "python3 /examples/python-functions/main.py --verbose --cluster-spec /examples/python-functions/cluster-spec.yaml"

test-localhost:
	docker run \
		--tty \
		--volume $(realpath ../../moose):/moose \
		--volume $(realpath .):/examples/python-functions \
		${MAIN_IMAGE} \
		sh -c "python3 /examples/python-functions/main.py --runtime test --verbose "



.PHONY: run test-localhost
