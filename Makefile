build:
	cargo build

pydep:
	pip install -r pymoose/requirements-dev.txt

pylib-release:
	cd pymoose && python setup.py install

install-release: pydep pylib-release

pylib:
	cd pymoose && python setup.py develop

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
	pytest -m "not slow" ./pymoose

test-long:
	$(MAKE) test
	pytest -m "slow" ./pymoose

test-ci:
	$(MAKE) test

clean:
	cargo clean
	find ./ -depth -type d -name '__pycache__' -prune -print -exec rm -rf {} +
	rm -rf ./pymoose/.pytest_cache

ci-ready:
	cargo clean
	$(MAKE) fmt
	$(MAKE) lint
	$(MAKE) test-ci

release: ci-ready
	cargo release --workspace --skip-publish

.PHONY: build pydep pylib install fmt lint test test-long test-ci clean ci-ready release
