all: help

install-linters: ## install the linters
	pip install ruff==0.6.9

install: ## install TetherFlow in editable mode
	python3 -m pip install -e .

lint: ## check formatting and lint rules with ruff
	python3 -m ruff format --check
	python3 -m ruff check

format: ## format the code and auto-fix lint rules with ruff
	python3 -m ruff format
	python3 -m ruff check --fix

test: ## run the unit tests
	python3 -m pytest tests

.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
