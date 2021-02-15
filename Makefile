build:
	python3 -m grpc_tools.protoc \
			--proto_path=. \
			--python_out=. \
			--grpc_python_out=. \
			moose/protos/*.proto
	cd rust && cargo build

pydep:
	pip install -r requirements-dev.txt
	pip install -r rust/pymoose/requirements-dev.txt

pylib:
	pip install -e .
	pip install -e rust/pymoose

install: pydep pylib

fmt:
	isort .
	black .
	cd rust; cargo fmt

fmt-check:
	cd rust && cargo fmt --all -- --check

lint:
	flake8 .
	cd rust && cargo clippy

lint-check:
	cd rust && cargo clippy -- -D warnings

test:
	pytest .
	cd rust && cargo test

clean:
	find ./moose -depth -type d -name '__pycache__' -prune -print -exec rm -rf {} +
	rm -rf .pytest_cache
	rm -Rf .hypothesis
	cargo clean

ci:
	make fmt
	make lint
	HYPOTHESIS_PROFILE='ci' $(MAKE) test

ci-long:
	make fmt
	make lint
	HYPOTHESIS_PROFILE='ci-long' $(MAKE) test


release: ci
	cd rust && cargo release --workspace --skip-publish

.PHONY: build pydep pylib install fmt lint test ci release
