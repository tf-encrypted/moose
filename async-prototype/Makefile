fmt:
	isort --atomic --recursive --skip protos/*.py .
	black .

lint:
	flake8 .

run:
	python main.py

test: run

.PHONY: fmt lint run test
