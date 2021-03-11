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
	cd rust && cargo fmt

lint:
	flake8 .
	cd rust && cargo fmt --all -- --check
	cd rust && cargo clippy -- -D warnings

test:
	pytest .
	cd rust && cargo test

test-long:
	HYPOTHESIS_PROFILE='test-long' $(MAKE) test

test-ci:
	HYPOTHESIS_PROFILE='ci' $(MAKE) test

clean:
	find ./moose -depth -type d -name '__pycache__' -prune -print -exec rm -rf {} +
	rm -rf .pytest_cache
	rm -Rf .hypothesis
	cd rust && cargo clean

ci-ready:
	cd rust && cargo clean
	$(MAKE) fmt
	$(MAKE) lint
	$(MAKE) test-ci

release: ci-ready
	cd rust && cargo release --workspace --skip-publish

.PHONY: build pydep pylib install fmt lint test test-long test-ci clean ci-ready release
