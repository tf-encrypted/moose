pydep:
	pip install -r requirements-dev.txt

protolib:
	python -m grpc_tools.protoc \
      --proto_path=. \
      --python_out=. \
      --grpc_python_out=. \
      protos/*.proto

pylib:
	pip install -e .

install: pydep protolib pylib

fmt:
	isort --atomic --recursive --skip protos/*.py .
	black .

lint:
	flake8 . --exclude=protos,venv

test:
	pytest .

ci: lint test

.PHONY: install fmt lint test ci
