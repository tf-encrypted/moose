up:
	docker-compose --file docker-compose.yaml \
		up --build --remove-orphans

up-dev:
	docker-compose --file docker-compose-dev.yaml \
		up --build --remove-orphans

down:
	docker-compose --file docker-compose.yaml \
		down --remove-orphans

down-dev:
	docker-compose --file docker-compose-dev.yaml \
		down --remove-orphans

.PHONY: up up-dev down down-dev
