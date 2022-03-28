.PHONY: build
build:
	cargo build

.PHONY: pydep
pydep:
	pip install -r pymoose/requirements-dev.txt

.PHONY: pylib-release
pylib-release:
	cd pymoose && python setup.py install
	cd pymoose && python setup.py bdist_wheel

.PHONY: install-release
install-release: pydep pylib-release

.PHONY: pylib
pylib:
	cd pymoose && python setup.py develop

.PHONY: install
install: pydep pylib

.PHONY: fmt
fmt:
	cargo fmt
	cd pymoose && isort .
	cd pymoose && black .

.PHONY: lint
lint:
	cargo fmt --all -- --check
	cargo clippy --all-targets -- -D warnings --no-deps
	cd pymoose && flake8 .

.PHONY: test
test:
	cargo test
	pytest -m "not slow" ./pymoose

.PHONY: test-long
test-long: test
	pytest -m "slow" ./pymoose

.PHONY: audit
audit:
	cargo audit

.PHONY: deny
deny:
	cargo deny check

.PHONY: test-ci
test-ci: test

.PHONY: clean
clean:
	cargo clean
	find ./ -depth -type d -name '__pycache__' -prune -print -exec rm -rf {} +
	rm -rf ./pymoose/.pytest_cache

.PHONY: ci-ready
ci-ready: fmt lint test-ci

.PHONY: ci-clean-check
ci-clean-check: clean ci-ready

# Cargo Release docs:
# https://github.com/crate-ci/cargo-release/blob/master/docs/reference.md
.PHONY: release
release: ci-ready
	cargo release --workspace --no-publish --execute

.PHONY: release
release-minor: ci-ready
	cargo release --workspace --no-publish --execute minor
