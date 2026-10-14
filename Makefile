.PHONY: check test build

check:
	uv run --locked ruff check src tests benchmarks tools
	uv run --locked python -m compileall -q src tests benchmarks tools

test:
	uv run --locked python -m unittest discover -s tests -p 'test_*.py' < /dev/null

build:
	uv build
