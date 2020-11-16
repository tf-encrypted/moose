up:
	docker-compose up --build --remove-orphans

down:
	docker-compose down --remove-orphans

run:
	docker-compose exec choreographer \
		sh -c "python3 choreographer.py --verbose"

.PHONY: certs up down run
