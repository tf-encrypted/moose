build:
	python3 -m grpc_tools.protoc \
			--proto_path=. \
			--python_out=. \
			--grpc_python_out=. \
			moose/protos/*.proto

pydep:
	pip install -r requirements-dev.txt

pylib:
	pip install -e .

install: pydep pylib

fmt:
	isort .
	black .

lint:
	flake8 --extend-exclude=venv* .

test:
	pytest .

ci: fmt lint test

.PHONY: build pydep pylib install fmt lint test ci
