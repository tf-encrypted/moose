build:
	python3 -m grpc_tools.protoc \
			--proto_path=. \
			--python_out=. \
			--grpc_python_out=. \
			moose/protos/*.proto
	cd rust && cargo build

pydep:
	pip install -r requirements-dev.txt
	pip install -r rust/python-bindings/requirements-dev.txt

pylib:
	pip install -e .
	pip install -e rust/python-bindings

install: pydep pylib

fmt:
	isort .
	black .

lint:
	flake8 .

test:
	pytest .

clean:
	find ./moose -depth -type d -name '__pycache__' -prune -print -exec rm -rf {} +
	rm -rf .pytest_cache
	rm -Rf .hypothesis

ci: fmt lint test

.PHONY: build pydep pylib install fmt lint test ci
