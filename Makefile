.PHONY: lint format format-check typecheck test privacy-check check install-hooks

lint:
	uv run ruff check .

format:
	uv run ruff format .

format-check:
	uv run ruff format --check .

typecheck:
	uv run pyrefly check

test:
	uv run pytest

privacy-check:
	uv run python scripts/check_private.py --all-tracked

check: lint format-check typecheck test privacy-check

install-hooks:
	@hook="$$(git rev-parse --git-path hooks/pre-commit)"; \
	mkdir -p "$$(dirname "$$hook")"; \
	printf '%s\n' '#!/bin/sh' 'exec uv run python scripts/check_private.py --staged' > "$$hook"; \
	chmod +x "$$hook"; \
	echo "installed $$hook"
