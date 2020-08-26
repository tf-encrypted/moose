fmt:
	isort --atomic --recursive --skip protos/*.py .
	black .

lint:
	flake8 .

test: lint 
		pytest .

.PHONY: fmt lint test
