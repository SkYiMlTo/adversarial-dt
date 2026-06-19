.PHONY: help up down logs test run-s1 run-s2 figures results clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Docker testbed
# ---------------------------------------------------------------------------

up: ## Start the Docker testbed
	cp -n .env.example .env 2>/dev/null || true
	docker compose up -d --build

down: ## Stop the Docker testbed
	docker compose down -v

logs: ## Follow testbed logs
	docker compose logs -f

# ---------------------------------------------------------------------------
# Experiments (offline, no Docker required)
# ---------------------------------------------------------------------------

run-s1: ## Run S1 automated evaluation (Tables 1, 3, 5)
	python3 experiments/run_s1_automated.py

run-s2: ## Run S2 SWaT offline evaluation (Tables 2, 4)
	@echo "S2 requires SWaT dataset. See README.md for instructions."
	@echo "python3 experiments/run_s2_offline.py"
	python3 experiments/run_s2_offline.py

figures: ## Generate publication figures
	python3 experiments/generate_figures.py

results: ## Collect results as LaTeX tables
	python3 experiments/collect_results.py

run-all: run-s1 run-s2 figures results ## Run all experiments and generate outputs

# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------

test: ## Run all tests
	python3 -m pytest tests/ -v

test-ekf: ## Test EKF correctness
	python3 -m pytest tests/test_ekf.py -v

test-cusum: ## Test CUSUM detector
	python3 -m pytest tests/test_cusum.py -v

test-iswt: ## Test ISWT detector
	python3 -m pytest tests/test_iswt.py -v

test-tca: ## Test TCA attack
	python3 -m pytest tests/test_tca.py -v

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

clean: ## Remove generated files
	rm -rf results/ __pycache__ .pytest_cache
	find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name '*.pyc' -delete 2>/dev/null || true

install: ## Install Python dependencies
	pip install -r requirements.txt
