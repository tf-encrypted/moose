fmt:
	isort --atomic --recursive --skip protos/*.py .
	black .

lint:
	flake8 .

test: lint

.PHONY: fmt lint test
