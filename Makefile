PYTHON ?= python
ifneq ("$(wildcard venv/bin/python)","")
PYTHON := venv/bin/python
endif
REPORTS_DIR ?= reports
ARTIFACTS_DIR ?= artifacts
QUALITY_GATE_JSON ?= $(ARTIFACTS_DIR)/eval_gate_report.json
QUALITY_GATE_MD ?= $(ARTIFACTS_DIR)/eval_gate_report.md

.PHONY: help app test smoke-local smoke-torch bootstrap-eval eval-report quality-gate quality-gate-soft prep-scifact train-verifier train-verifier-quick show-model-registry docker-build-cpu docker-up-cpu docker-build-gpu docker-up-gpu docker-up-auto

help:
	@echo "Common targets:"
	@echo "  make app             - Run the Streamlit app"
	@echo "  make test            - Run unit tests"
	@echo "  make smoke-local     - Run no-network local baseline smoke test"
	@echo "  make smoke-torch     - Run trainable torch verifier smoke test"
	@echo "  make bootstrap-eval  - Generate a reproducible eval report template"
	@echo "  make eval-report     - Run retrieval + verdict evaluation harness"
	@echo "  make quality-gate    - Strict quality gate check (non-pass exits with error)"
	@echo "  make quality-gate-soft - Soft gate check for local diagnostics"
	@echo "  make prep-scifact    - Download/process SciFact into train/dev pairs"
	@echo "  make train-verifier  - Train torch verifier on processed SciFact pairs"
	@echo "  make train-verifier-quick - Quick verifier training run for laptop smoke tests"
	@echo "  make show-model-registry - Show recent registered model runs"
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

eval-report:
	$(PYTHON) scripts/evaluate_rag_stack.py \
		--pairs-file data/scifact/processed/dev_pairs.jsonl \
		--scifact-claims-file data/scifact/raw/data/claims_dev.jsonl \
		--scifact-corpus-file data/scifact/raw/data/corpus.jsonl \
		--output-json $(REPORTS_DIR)/eval_report.json \
		--output-md $(REPORTS_DIR)/eval_report.md

quality-gate:
	$(PYTHON) scripts/evaluate_rag_stack.py \
		--pairs-file data/scifact/processed/dev_pairs.jsonl \
		--scifact-claims-file data/scifact/raw/data/claims_dev.jsonl \
		--scifact-corpus-file data/scifact/raw/data/corpus.jsonl \
		--output-json $(QUALITY_GATE_JSON) \
		--output-md $(QUALITY_GATE_MD)
	$(PYTHON) scripts/check_quality_gate.py --report $(QUALITY_GATE_JSON)

quality-gate-soft:
	$(PYTHON) scripts/evaluate_rag_stack.py \
		--pairs-file data/scifact/processed/dev_pairs.jsonl \
		--scifact-claims-file data/scifact/raw/data/claims_dev.jsonl \
		--scifact-corpus-file data/scifact/raw/data/corpus.jsonl \
		--output-json $(QUALITY_GATE_JSON) \
		--output-md $(QUALITY_GATE_MD)
	$(PYTHON) scripts/check_quality_gate.py --report $(QUALITY_GATE_JSON) --allow-fail --allow-pending

prep-scifact:
	$(PYTHON) scripts/prepare_scifact_pairs.py --data-root data/scifact

train-verifier:
	$(PYTHON) scripts/train_torch_verifier.py \
		--train-file data/scifact/processed/train_pairs.jsonl \
		--dev-file data/scifact/processed/dev_pairs.jsonl \
		--output-dir artifacts/torch_verifier \
		--epochs 6 \
		--batch-size 64

train-verifier-quick:
	$(PYTHON) scripts/train_torch_verifier.py \
		--train-file data/scifact/processed/train_pairs.jsonl \
		--dev-file data/scifact/processed/dev_pairs.jsonl \
		--output-dir artifacts/torch_verifier_quick \
		--epochs 2 \
		--batch-size 64 \
		--max-train-samples 1200 \
		--max-dev-samples 400

show-model-registry:
	$(PYTHON) scripts/show_model_registry.py --registry artifacts/model_registry.jsonl --limit 5

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
