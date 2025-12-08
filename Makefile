# Makefile for Prismatron LED Display System
# Administrative operations for setup, calibration, and deployment

.PHONY: help frontend calibrate capture matrices setup install-service uninstall-service test pre-commit clean

# Project paths
PROJECT_DIR := $(shell pwd)
FRONTEND_DIR := $(PROJECT_DIR)/src/web/frontend
DATA_DIR := /mnt/prismatron
SCRIPTS_DIR := $(PROJECT_DIR)/scripts
TOOLS_DIR := $(PROJECT_DIR)/tools
VENV := $(PROJECT_DIR)/env/bin

# Default target
help: ## Show this help message
	@echo "Prismatron LED Display System - Admin Commands"
	@echo "==============================================="
	@echo ""
	@echo "Setup & Installation:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "Calibration Pipeline:"
	@echo "  1. make calibrate   - Calibrate camera region of interest"
	@echo "  2. make capture     - Capture LED diffusion patterns"
	@echo "  3. make matrices    - Compute optimization matrices"
	@echo ""
	@echo "Environment:"
	@echo "  Python: $$($(VENV)/python --version 2>/dev/null || echo 'Not found')"
	@echo "  Data dir: $(DATA_DIR)"
	@echo ""

# =============================================================================
# Frontend Build
# =============================================================================

frontend: ## Build the web frontend (React/Vite)
	@echo "Building frontend..."
	cd $(FRONTEND_DIR) && npm install && npm run build
	@echo "Frontend built to $(FRONTEND_DIR)/dist/"

frontend-dev: ## Start frontend dev server with hot reload
	cd $(FRONTEND_DIR) && npm run dev

# =============================================================================
# Calibration Pipeline
# =============================================================================

calibrate: ## Run camera calibration (interactive)
	@echo "Starting camera calibration..."
	@echo "This will open an interactive window to select the display region."
	$(VENV)/python $(TOOLS_DIR)/camera_calibration.py --camera-device 0

capture: ## Capture LED diffusion patterns (requires calibrated camera)
	@echo "Starting pattern capture..."
	@echo "This will illuminate each LED and capture diffusion patterns."
	@echo ""
	@read -p "WLED host IP [192.168.4.174]: " host; \
	host=$${host:-192.168.4.174}; \
	read -p "Output filename [patterns.npz]: " output; \
	output=$${output:-patterns.npz}; \
	$(VENV)/python $(TOOLS_DIR)/capture_diffusion_patterns.py \
		--wled-host $$host \
		--camera-device 0 \
		--output $(DATA_DIR)/patterns/$$output \
		--preview

matrices: ## Compute ATA matrices from captured patterns
	@echo "Computing optimization matrices..."
	@if [ -z "$(PATTERN)" ]; then \
		echo "Usage: make matrices PATTERN=filename.npz"; \
		echo ""; \
		echo "Available pattern files:"; \
		ls -1 $(DATA_DIR)/patterns/*.npz 2>/dev/null || echo "  (none found)"; \
		exit 1; \
	fi
	$(VENV)/python $(TOOLS_DIR)/compute_matrices.py $(DATA_DIR)/patterns/$(PATTERN)

# =============================================================================
# Data Directory Setup
# =============================================================================

setup: ## Setup data directories and initialize config
	@echo "Setting up Prismatron data directories..."
	@echo ""
	@if [ ! -d "$(DATA_DIR)" ]; then \
		echo "Creating $(DATA_DIR) (requires sudo)..."; \
		sudo mkdir -p $(DATA_DIR); \
		sudo chown $(USER):$(USER) $(DATA_DIR); \
	fi
	@mkdir -p $(DATA_DIR)/config
	@mkdir -p $(DATA_DIR)/patterns
	@mkdir -p $(DATA_DIR)/media
	@mkdir -p $(DATA_DIR)/playlists
	@mkdir -p $(DATA_DIR)/uploads
	@mkdir -p $(DATA_DIR)/conversions
	@mkdir -p $(DATA_DIR)/logs
	@echo "Created directory structure:"
	@ls -la $(DATA_DIR)/
	@echo ""
	@if [ ! -f "$(DATA_DIR)/config/config.json" ]; then \
		echo "Creating default config..."; \
		cp $(PROJECT_DIR)/backup/config/config.json $(DATA_DIR)/config/config.json 2>/dev/null || \
		echo '{"debug": false, "web_host": "0.0.0.0", "web_port": 8080, "wled_hosts": ["192.168.4.174"], "wled_port": 4048}' > $(DATA_DIR)/config/config.json; \
		echo "Config created at $(DATA_DIR)/config/config.json"; \
	else \
		echo "Config already exists at $(DATA_DIR)/config/config.json"; \
	fi
	@echo ""
	@echo "Setup complete!"

# =============================================================================
# Service Installation
# =============================================================================

install-service: ## Install user systemd service (includes dummy display)
	@echo "Installing Prismatron service..."
	@echo ""
	@echo "Step 1: Setting up dummy X11 display (requires sudo)..."
	@if [ ! -f "/etc/X11/xorg-dummy.conf" ]; then \
		sudo cp $(SCRIPTS_DIR)/xorg-dummy.conf /etc/X11/xorg-dummy.conf; \
		echo "  Installed xorg-dummy.conf"; \
	else \
		echo "  xorg-dummy.conf already installed"; \
	fi
	@echo ""
	@echo "Step 2: Starting dummy display..."
	$(SCRIPTS_DIR)/start-dummy-display.sh start
	@echo ""
	@echo "Step 3: Installing user service..."
	$(SCRIPTS_DIR)/setup_user_service.sh
	@echo ""
	@echo "Service installation complete!"
	@echo "Start with: systemctl --user start prismatron"

uninstall-service: ## Uninstall user systemd service
	$(SCRIPTS_DIR)/uninstall_user_service.sh

service-status: ## Show service status
	@systemctl --user status prismatron 2>/dev/null || echo "Service not installed"

service-logs: ## Show service logs (follow mode)
	journalctl --user -u prismatron -f

# =============================================================================
# Development
# =============================================================================

test: ## Run pytest tests
	$(VENV)/python -m pytest tests/ -v --tb=short

pre-commit: ## Run pre-commit hooks on all files
	pre-commit run --all-files

dev-server: ## Start development API server
	$(VENV)/python -m uvicorn src.web.api_server:app --reload --host 0.0.0.0 --port 8080

run: ## Run prismatron directly (not as service)
	$(VENV)/python main.py --config $(DATA_DIR)/config/config.json

# =============================================================================
# Cleanup
# =============================================================================

clean: ## Clean build artifacts and cache
	@echo "Cleaning up..."
	rm -rf $(FRONTEND_DIR)/dist $(FRONTEND_DIR)/node_modules/.vite
	rm -rf .pytest_cache/ .mypy_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@echo "Clean complete"
