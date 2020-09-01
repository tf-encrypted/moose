fmt:
	isort --atomic --recursive --skip protos/*.py .
	black .

lint:
	flake8 .

test:
	pytest .

.PHONY: fmt lint test
