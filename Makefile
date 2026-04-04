.PHONY: install api frontend lint test train clean

install:
	pip install -r requirements.txt && pip install -e ".[dev]"

api:
	uvicorn src.serving.app:app --reload --port 8000

frontend:
	cd frontend && npm run dev

lint:
	ruff check src/ tests/

test:
	pytest tests/ -v --tb=short

train:
	python scripts/train_model.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; \
	find . -type f -name "*.pyc" -delete 2>/dev/null; \
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null; \
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null; \
	echo "Cleaned."
