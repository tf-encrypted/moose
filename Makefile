.PHONY: build
build:
	cargo build

.PHONY: pydep
pydep:
	pip install -r pymoose/requirements/base.txt -r pymoose/requirements/dev.txt

.PHONY: pydep-upgrade
pydep-upgrade:
	pip install -U pip-tools
	CUSTOM_COMPILE_COMMAND="make pydep-upgrade" pip-compile --output-file=pymoose/requirements/base.txt pymoose/requirements/base.in
	CUSTOM_COMPILE_COMMAND="make pydep-upgrade" pip-compile --output-file=pymoose/requirements/dev.txt pymoose/requirements/dev.in
	pip install -r pymoose/requirements/base.txt -r pymoose/requirements/dev.txt

.PHONY: pylib
pylib:
	cd pymoose && maturin develop

.PHONY: install
install: pydep pylib

.PHONY: fmt
fmt:
	cargo fmt
	cd pymoose && isort .
	cd pymoose && black .
	cd benchmarks/pymoose && isort .
	cd benchmarks/pymoose && black .


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

.PHONY: release
release: ci-ready
	cargo release --workspace --no-publish --execute


# PyMoose Docs

.PHONY: install-docs
install-docs:
	pip install -r pymoose/docs/requirements.txt

.PHONY: docs-clean
docs-clean:
	find pymoose/docs/source -name "*.moose" -exec rm {} \+
	find pymoose/docs/source -name "*.ipynb" -exec rm {} \+
	find pymoose/docs/source -name "*.md" -exec rm {} \+
	find pymoose/docs/source/_static -name "*.png" -exec rm {} \+
	cd pymoose/docs && \
	make clean && \
	cd ../../

.PHONY: docs-prep
docs-prep:
	cp README.md pymoose/docs/source/moose-readme.md && \
	cp tutorials/README.md pymoose/docs/source/tutorial-readme.md && \
	cp tutorials/scientific-computing-multiple-players.ipynb pymoose/docs/source/ && \
	cp tutorials/ml-inference-with-onnx.ipynb pymoose/docs/source/ && \
	cp tutorials/interfacing-moose-with-pymoose.ipynb pymoose/docs/source/ && \
	cp -r tutorials/_static/ pymoose/docs/source/

.PHONY: docs
docs: docs-prep
	cd pymoose/docs && \
	make html && \
	cd ../../