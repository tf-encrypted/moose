up:
	docker-compose up --build --remove-orphans

down:
	docker-compose down --remove-orphans

run:
	docker-compose exec choreographer \
		sh -c "python3 main.py --runtime remote --verbose"

test-localhost:
	docker run \
		--tty \
		--volume $(realpath ../..):/worker/moose \
		tfencrypted/runtime-mpspdz-worker:latest \
		sh -c "cd /worker/moose/examples/mp-spdz; python3 main.py --runtime test --verbose"

.PHONY: up down run test-localhost
