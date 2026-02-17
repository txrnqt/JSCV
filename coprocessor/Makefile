
# Project Makefile

# Python interpreter
PYTHON := python3
VENV_DIR := .venv
REQ_FILE := requirements.txt
REQ_FILE_MACOS := requirements-macos.txt

# Default target
.PHONY: all
all: help

# -------------------------
# Environment setup
# -------------------------

.PHONY: venv
venv:
	@echo "Creating virtual environment..."
	$(PYTHON) -m venv $(VENV_DIR)

.PHONY: install
install: venv
	@echo "Installing dependencies (cross-platform)..."
	$(VENV_DIR)/bin/pip install --upgrade pip
	$(VENV_DIR)/bin/pip install -r $(REQ_FILE)

.PHONY: install-macos
install-macos: venv
	@echo "Installing macOS dependencies (including CoreML)..."
	$(VENV_DIR)/bin/pip install --upgrade pip
	$(VENV_DIR)/bin/pip install -r $(REQ_FILE_MACOS)

# -------------------------
# Run targets
# -------------------------

.PHONY: run
run:
	$(VENV_DIR)/bin/python main.py

.PHONY: test
test:
	$(VENV_DIR)/bin/python -m unittest discover -s tests

# -------------------------
# Cleaning
# -------------------------

.PHONY: clean
clean:
	rm -rf __pycache__
	rm -rf $(VENV_DIR)
	find . -name "*.pyc" -delete

# -------------------------
# Help
# -------------------------

.PHONY: help
help:
	@echo "Available make targets:"
	@echo "  venv           - Create a virtual environment"
	@echo "  install        - Install cross-platform Python dependencies"
	@echo "  install-macos  - Install macOS dependencies (with CoreML)"
	@echo "  run            - Run the main application"
	@echo "  test           - Run unit tests"
	@echo "  clean          - Remove temporary files and virtual environment"
