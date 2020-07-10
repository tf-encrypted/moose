fmt:
	black .

lint:
	flake8 . --max-line-length 88
	black . --check

run:
	python main.py

test: run

.PHONY: fmt lint run test
