.PHONY: help install run clean

PORT ?= 8002

help:
	@echo "Available targets:"
	@echo "  make install  - Install Python dependencies"
	@echo "  make run      - Run the service with uvicorn (default port: 8002)"
	@echo "  make clean    - Remove Python cache files"
	@echo ""
	@echo "You can set a custom port with: make run PORT=8000"

install:
	pip install -r requirements.txt

run:
	uvicorn app:app --host 0.0.0.0 --port $(PORT) --reload

clean:
	find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

