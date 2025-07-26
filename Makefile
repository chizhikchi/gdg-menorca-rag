.PHONY: help install dev-install test format lint clean run-cli run-app

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install production dependencies
	pip install -r requirements.txt

dev-install: ## Install development dependencies
	pip install -e ".[dev]"

test: ## Run tests
	pytest tests/ -v

format: ## Format code with black
	black .

lint: ## Lint code with flake8
	flake8 .

clean: ## Clean generated files
	rm -rf generated_docs/* backups/* __pycache__/ .pytest_cache/
	find . -name "*.pyc" -delete

run-cli: ## Run CLI interface
	python rag_manager.py --help

run-app: ## Run Gradio app
	python app.py

# Docker commands coming soon - temporarily removed for GitHub upload
# docker-build: ## Build Docker image
# 	docker build -t gdg-menorca-rag .
# 
# docker-run: ## Run Docker container
# 	docker run -p 8080:8080 --env-file .env gdg-menorca-rag

generate: ## Generate corpus (interactive)
	python rag_manager.py generate --interactive

status: ## Show corpus status
	python rag_manager.py status

cleanup: ## Cleanup (dry run)
	python rag_manager.py cleanup --dry-run