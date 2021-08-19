build:
	python3 -m grpc_tools.protoc \
			--proto_path=. \
			--python_out=. \
			--grpc_python_out=. \
			pymoose/pymoose/protos/*.proto
	cargo build

pydep:
	pip install -r pymoose/requirements-dev.txt

pylib:
	pip install -e pymoose

install: pydep pylib

fmt:
	cd pymoose && isort .
	cd pymoose && black .
	cargo fmt

lint:
	cd pymoose && flake8 .
	cargo fmt --all -- --check
	cargo clippy --all-targets -- -D warnings

test:
	cd pymoose && pytest -m "not slow"
	cargo test --no-default-features

test-long:
	HYPOTHESIS_PROFILE='test-long' $(MAKE) test
	cd pymoose && pytest -m "slow"

test-ci:
	HYPOTHESIS_PROFILE='ci' $(MAKE) test

clean:
	find ./ -depth -type d -name '__pycache__' -prune -print -exec rm -rf {} +
	rm -rf ./pymoose/.pytest_cache
	rm -Rf .hypothesis
	cargo clean

ci-ready:
	cargo clean
	$(MAKE) fmt
	$(MAKE) lint
	$(MAKE) test-ci

release: ci-ready
	cargo release --workspace --skip-publish

.PHONY: build pydep pylib install fmt lint test test-long test-ci clean ci-ready release
