# Makefile for Prismatron LED Display Software
# Provides convenient commands for development, testing, and deployment

.PHONY: help install install-dev test test-cov lint format type-check clean docs serve-docs build docker-build docker-run pre-commit

# Default target
help: ## Show this help message
	@echo "Prismatron LED Display Software - Development Commands"
	@echo "======================================================"
	@echo ""
	@echo "Available commands:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "Environment:"
	@echo "  Python: $$(python --version 2>/dev/null || echo 'Not found')"
	@echo "  Virtual env: $$(if [ -n "$$VIRTUAL_ENV" ]; then echo "$$VIRTUAL_ENV"; else echo "Not activated"; fi)"
	@echo ""

# Installation and setup
install: ## Install production dependencies
	pip install -r requirements.txt

install-dev: ## Install development dependencies including pre-commit hooks
	pip install -r requirements.txt
	pip install -e ".[dev]"
	pre-commit install
	@echo "✅ Development environment setup complete"

# Code quality and formatting
format: ## Format code with Black and isort
	@echo "🎨 Formatting Python code..."
	black --line-length 88 src/ tests/ tools/
	isort --profile black --line-length 88 src/ tests/ tools/
	@echo "✅ Code formatting complete"

format-check: ## Check if code formatting is correct (CI mode)
	@echo "🔍 Checking code formatting..."
	black --check --line-length 88 src/ tests/ tools/
	isort --check-only --profile black --line-length 88 src/ tests/ tools/
	@echo "✅ Code formatting check passed"

lint: ## Run flake8 linting
	@echo "🔍 Running linting checks..."
	flake8 src/ tests/ tools/
	@echo "✅ Linting passed"

type-check: ## Run mypy type checking
	@echo "🔍 Running type checks..."
	mypy src/ tests/ tools/
	@echo "✅ Type checking passed"

# Testing
test: ## Run pytest tests
	@echo "🧪 Running tests..."
	pytest tests/ -v --tb=short
	@echo "✅ Tests completed"

test-cov: ## Run tests with coverage reporting
	@echo "🧪 Running tests with coverage..."
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term
	@echo "✅ Coverage report generated in htmlcov/"

test-hardware: ## Run hardware-specific tests (requires actual hardware)
	@echo "🔌 Running hardware tests..."
	pytest tests/ -v -m hardware
	@echo "✅ Hardware tests completed"

test-integration: ## Run integration tests
	@echo "🔗 Running integration tests..."
	pytest tests/ -v -m integration
	@echo "✅ Integration tests completed"

test-all: ## Run all tests including slow and hardware tests
	@echo "🧪 Running all tests..."
	pytest tests/ -v --cov=src --cov-report=html
	@echo "✅ All tests completed"

# Pre-commit and CI
pre-commit: ## Run all pre-commit hooks on all files
	@echo "🔍 Running pre-commit hooks..."
	pre-commit run --all-files
	@echo "✅ Pre-commit hooks completed"

ci-check: format-check lint type-check test-cov ## Run all CI checks locally
	@echo "✅ All CI checks passed"

# Cleaning
clean: ## Clean up build artifacts and cache files
	@echo "🧹 Cleaning up..."
	rm -rf build/ dist/ *.egg-info/
	rm -rf htmlcov/ .coverage .pytest_cache/
	rm -rf .mypy_cache/ .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*~" -delete
	@echo "✅ Cleanup complete"

clean-env: ## Remove and recreate virtual environment
	@echo "🗑️  Removing virtual environment..."
	rm -rf env/
	python -m venv env
	@echo "✅ Fresh virtual environment created"
	@echo "   Run 'source env/bin/activate && make install-dev' to set up"

# Documentation
docs: ## Build documentation
	@echo "📚 Building documentation..."
	mkdocs build
	@echo "✅ Documentation built in site/"

serve-docs: ## Serve documentation locally for development
	@echo "📚 Serving documentation at http://localhost:8000"
	mkdocs serve

# Package building
build: ## Build Python package
	@echo "📦 Building package..."
	python -m build
	@echo "✅ Package built in dist/"

# Docker commands
docker-build: ## Build Docker image
	@echo "🐳 Building Docker image..."
	docker build -t prismatron:latest .
	@echo "✅ Docker image built"

docker-run: ## Run application in Docker container
	@echo "🐳 Starting Docker container..."
	docker run -it --rm -p 8000:8000 prismatron:latest

# Development utilities
dev-server: ## Start development server with hot reload
	@echo "🚀 Starting development server..."
	python -m uvicorn src.web.api_server:app --reload --host 0.0.0.0 --port 8000

wled-test: ## Run WLED test patterns (requires WLED controller)
	@echo "🌈 Testing WLED connection..."
	python tools/wled_test_patterns.py test --verbose

wled-rainbow: ## Start rainbow pattern on WLED (requires WLED controller)
	@echo "🌈 Starting rainbow pattern..."
	python tools/wled_test_patterns.py rainbow-cycle --speed 1.0

# Project initialization for new clones
init: ## Initialize project after cloning (one-time setup)
	@echo "🚀 Initializing Prismatron development environment..."
	python -m venv env
	@echo "   Virtual environment created in env/"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Activate virtual environment: source env/bin/activate"
	@echo "  2. Install dependencies: make install-dev"
	@echo "  3. Run tests: make test"
	@echo "  4. Start developing! 🎉"

# Environment info
info: ## Show environment and dependency information
	@echo "Prismatron Development Environment Information"
	@echo "============================================="
	@echo ""
	@echo "Python Environment:"
	@echo "  Python version: $$(python --version 2>/dev/null || echo 'Not found')"
	@echo "  Python executable: $$(which python 2>/dev/null || echo 'Not found')"
	@echo "  Virtual environment: $$(if [ -n "$$VIRTUAL_ENV" ]; then echo "$$VIRTUAL_ENV"; else echo "Not activated"; fi)"
	@echo "  pip version: $$(pip --version 2>/dev/null || echo 'Not found')"
	@echo ""
	@echo "Project Structure:"
	@echo "  Source code: src/"
	@echo "  Tests: tests/"
	@echo "  Tools: tools/"
	@echo "  Documentation: docs/"
	@echo ""
	@echo "Development Tools:"
	@echo "  Black: $$(black --version 2>/dev/null || echo 'Not installed')"
	@echo "  Flake8: $$(flake8 --version 2>/dev/null || echo 'Not installed')"
	@echo "  MyPy: $$(mypy --version 2>/dev/null || echo 'Not installed')"
	@echo "  Pytest: $$(pytest --version 2>/dev/null || echo 'Not installed')"
	@echo "  Pre-commit: $$(pre-commit --version 2>/dev/null || echo 'Not installed')"
	@echo ""

# Quick commands for common workflows
quick-test: ## Quick test run (fast tests only)
	pytest tests/ -v -x -m "not slow" --tb=short

quick-format: ## Quick format and lint check
	black src/ tests/ tools/ && flake8 src/ tests/ tools/

# Release preparation
prepare-release: ## Prepare for release (run all checks and build)
	@echo "🚀 Preparing release..."
	make clean
	make ci-check
	make build
	@echo "✅ Release preparation complete"
	@echo "   - All checks passed"
	@echo "   - Package built in dist/"
	@echo "   - Ready for release!"
