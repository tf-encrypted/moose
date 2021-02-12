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

lint:
	flake8 .

test:
	pytest .

clean:
	find ./moose -depth -type d -name '__pycache__' -prune -print -exec rm -rf {} +
	rm -rf .pytest_cache
	rm -Rf .hypothesis

ci-routines: fmt lint

ci:
	$(MAKE) ci-routines
	HYPOTHESIS_PROFILE='ci' $(MAKE) test

ci-long:
	$(MAKE) ci-routines
	HYPOTHESIS_PROFILE='ci-long' $(MAKE) test

rust-ci:
	cd rust && cargo fmt --all -- --check && cargo clippy -- -D warnings

release: ci
	cd rust && cargo release --workspace --skip-publish

.PHONY: build pydep pylib install fmt lint test ci release
