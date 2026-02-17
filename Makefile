PYTHON ?= python
ifneq ("$(wildcard venv/bin/python)","")
PYTHON := venv/bin/python
endif
REPORTS_DIR ?= reports
ARTIFACTS_DIR ?= artifacts

.PHONY: help app test smoke-local smoke-torch bootstrap-eval docker-build-cpu docker-up-cpu docker-build-gpu docker-up-gpu docker-up-auto

help:
	@echo "Common targets:"
	@echo "  make app             - Run the Streamlit app"
	@echo "  make test            - Run unit tests"
	@echo "  make smoke-local     - Run no-network local baseline smoke test"
	@echo "  make smoke-torch     - Run trainable torch verifier smoke test"
	@echo "  make bootstrap-eval  - Generate a reproducible eval report template"
	@echo "  make docker-build-cpu - Build CPU Docker image"
	@echo "  make docker-up-cpu    - Run app in CPU Docker container"
	@echo "  make docker-build-gpu - Build GPU-capable Docker image"
	@echo "  make docker-up-gpu    - Run app in GPU Docker container"
	@echo "  make docker-up-auto   - Auto-select GPU if available, else CPU"

app:
	$(PYTHON) -m streamlit run app.py

test:
	$(PYTHON) -m pytest -q

smoke-local:
	LLM_PROVIDER=local $(PYTHON) scripts/smoke_local_pipeline.py

smoke-torch:
	$(PYTHON) scripts/smoke_torch_verifier.py

bootstrap-eval:
	$(PYTHON) scripts/bootstrap_eval_report.py --output $(REPORTS_DIR)/eval_report.bootstrap.json

docker-build-cpu:
	docker compose build rag-app-cpu

docker-up-cpu:
	docker compose up --build rag-app-cpu

docker-build-gpu:
	docker compose --profile gpu build rag-app-gpu

docker-up-gpu:
	docker compose --profile gpu up --build rag-app-gpu

docker-up-auto:
	bash scripts/docker_up_auto.sh
