build:
	python -m grpc_tools.protoc \
			--proto_path=. \
			--python_out=. \
			--grpc_python_out=. \
			protos/*.proto

pydep:
	pip install -r requirements-dev.txt

pylib:
	pip install -e .

install: pydep pylib

fmt:
	isort --atomic --recursive --skip protos/*.py .
	black .

lint:
	flake8 . --exclude=protos,venv

test: build
	pytest .

ci: fmt lint test

.PHONY: build pydep pylib install fmt lint test ci
