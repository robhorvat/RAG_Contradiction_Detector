PYTHON ?= python
ifneq ("$(wildcard venv/bin/python)","")
PYTHON := venv/bin/python
endif
REPORTS_DIR ?= reports
ARTIFACTS_DIR ?= artifacts

.PHONY: help app test smoke-local bootstrap-eval

help:
	@echo "Common targets:"
	@echo "  make app             - Run the Streamlit app"
	@echo "  make test            - Run unit tests"
	@echo "  make smoke-local     - Run no-network local baseline smoke test"
	@echo "  make bootstrap-eval  - Generate a reproducible eval report template"

app:
	$(PYTHON) -m streamlit run app.py

test:
	$(PYTHON) -m pytest -q

smoke-local:
	LLM_PROVIDER=local $(PYTHON) scripts/smoke_local_pipeline.py

bootstrap-eval:
	$(PYTHON) scripts/bootstrap_eval_report.py --output $(REPORTS_DIR)/eval_report.bootstrap.json
