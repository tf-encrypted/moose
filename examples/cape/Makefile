up:
	docker-compose up --build --remove-orphans

up-dev:
	docker-compose --file docker-compose-dev.yaml up --build --remove-orphans

down:
	docker-compose down --remove-orphans

down-dev:
	docker-compose --file docker-compose-dev.yaml down --remove-orphans

run:
	docker-compose exec choreographer \
		sh -c "python3 choreographer.py --verbose"

.PHONY: up up-dev down down-dev run
