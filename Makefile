PY := .venv/bin/python
SEEDS ?= 0

.PHONY: setup data eval stream export tables verify test lint demo docker clean all

all: data eval stream export tables test

setup:            ## virtual environment with pinned dependencies
	/opt/homebrew/bin/python3.12 -m venv .venv || python3 -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.lock
	$(PY) -c "import torch; print('MPS available:', torch.backends.mps.is_available())"

data:             ## download (verified against git blob SHA-1), audit, feature cache, split manifests
	$(PY) scripts/download.py
	$(PY) scripts/prepare.py

eval:             ## all scorers x calibration strategies x protocols
	for s in $(SEEDS); do $(PY) scripts/run_eval.py --proto lopo lomo time --scorers B0 B1 B2 --seed $$s; done
	for s in $(SEEDS); do $(PY) scripts/run_eval.py --proto lopo lomo time --scorers AE --seed $$s; done

stream:           ## streaming label-delay study (claim C2): alpha sweeps, all scorers, both stream orders, label coverage
	./scripts/run_all_streams.sh
	$(PY) scripts/run_fewshot.py --scorer B2 --alpha 0.05
	$(PY) scripts/run_fewshot.py --scorer B0 --alpha 0.05

export:           ## ONNX FP32 + static INT8, parity and latency (claim C3)
	$(PY) scripts/export_edge.py

tables:           ## results/summary.json, docs/*_tables.md, figures, README results, model card
	$(PY) scripts/make_tables.py
	$(PY) scripts/make_stream_tables.py
	$(PY) scripts/make_headline.py
	$(PY) scripts/update_readme.py
	$(PY) scripts/update_model_card.py

verify:           ## re-run a fast subset and compare to committed results within tolerance
	$(PY) scripts/verify.py

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src scripts tests

demo:             ## start the service and dashboard on http://localhost:8000
	PYTHONPATH=src $(PY) -m uvicorn driftguard.serve.app:app --port 8000

docker:
	docker compose up --build

clean:
	rm -rf .pytest_cache .ruff_cache
