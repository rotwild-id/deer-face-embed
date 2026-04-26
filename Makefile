.DEFAULT_GOAL := help

# ─── Configuration ────────────────────────────────────────────────────────────

SRC_DIR   := src/deer_face_embed
PYTHON    := python

# ─── Help ─────────────────────────────────────────────────────────────────────

.PHONY: help
help:
	@echo ""
	@echo "Usage: make <target>"
	@echo ""
	@echo "  Code quality"
	@echo "    lint          Run ruff linter"
	@echo "    format        Run ruff formatter"
	@echo "    format-check  Check formatting without modifying files"
	@echo "    typecheck     Run ty type checker"
	@echo "    check         Run lint + format-check + typecheck (CI-safe)"
	@echo "    fix           Run ruff linter + formatter with auto-fixes"
	@echo ""
	@echo "  Testing"
	@echo "    test          Run pytest"
	@echo "    test-cov      Run pytest with coverage report"
	@echo ""
	@echo "  Utilities"
	@echo "    clean         Remove cache and build artifacts"
	@echo ""

# ─── Code quality ─────────────────────────────────────────────────────────────

.PHONY: lint
lint:
	uv run ruff check $(SRC_DIR)
# uv run ruff check $(SRC_DIR) $(TEST_DIR)

.PHONY: format
format:
	uv run ruff format $(SRC_DIR)

.PHONY: format-check
format-check:
	uv run ruff format --check $(SRC_DIR)

.PHONY: typecheck
typecheck:
	uv run ty check $(SRC_DIR)

.PHONY: check
check: lint format-check typecheck
	@echo ""
	@echo "✅ All checks passed."

.PHONY: fix
fix:
	uv run ruff check --fix $(SRC_DIR)
	uv run ruff format $(SRC_DIR)

# ─── Testing ──────────────────────────────────────────────────────────────────

.PHONY: test
test:
	uv run pytest

.PHONY: test-cov
test-cov:
	uv run pytest --cov=$(SRC_DIR) --cov-report=term-missing

# ─── Utilities ────────────────────────────────────────────────────────────────

.PHONY: clean
clean:
	find . -type d -name "__pycache__"  -exec rm -rf {} +
	find . -type d -name ".ty_cache"    -exec rm -rf {} +
	find . -type d -name ".ruff_cache"  -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name "*.egg-info"   -exec rm -rf {} +
	find . -type f -name "*.pyc"        -delete
	@echo "🧹 Cleaned."