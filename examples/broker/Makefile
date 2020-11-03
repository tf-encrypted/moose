up:
	docker-compose up --build

down:
	docker-compose down --remove-orphans

run:
	docker-compose exec coordinator sh -c "python3 main.py --verbose --runtime remote"

.PHONY: up down run
