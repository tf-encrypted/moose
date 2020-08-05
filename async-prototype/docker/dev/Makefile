build:
	docker build \
		--file Dockerfile.worker \
		--tag tfencrypted/runtime-dev-worker \
		../..

run:
	docker run -it \
		-v $(shell realpath ../..):/runtime \
		tfencrypted/runtime-dev-worker:latest

up:
	docker-compose up

down:
	docker-compose down

.PHONY: build run up down
