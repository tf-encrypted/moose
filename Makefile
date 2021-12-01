build:
	cargo build

pydep:
	pip install -r pymoose/requirements-dev.txt

pylib:
	pip install -e pymoose

install: pydep pylib

fmt:
	cargo fmt
	cd pymoose && isort .
	cd pymoose && black .

lint:
	cargo fmt --all -- --check
	cargo clippy --all-targets -- -D warnings --no-deps
	cd pymoose && flake8 .

test:
	cargo test
	cd pymoose && pytest -m "not slow"

test-long:
	HYPOTHESIS_PROFILE='test-long' $(MAKE) test
	cd pymoose && pytest -m "slow"

test-ci:
	HYPOTHESIS_PROFILE='ci' $(MAKE) test

clean:
	cargo clean
	find ./ -depth -type d -name '__pycache__' -prune -print -exec rm -rf {} +
	rm -rf ./pymoose/.pytest_cache
	rm -Rf .hypothesis

ci-ready:
	cargo clean
	$(MAKE) fmt
	$(MAKE) lint
	$(MAKE) test-ci

release: ci-ready
	cargo release --workspace --skip-publish

.PHONY: build pydep pylib install fmt lint test test-long test-ci clean ci-ready release
