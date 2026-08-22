DATASET ?= all
SCALE ?= demo
PY := .venv/bin/python

.PHONY: venv data eval-bm25 embeddings eval-embeddings compare-retrieval eval-harness test clean-data

venv:
	python3 -m venv .venv
	$(PY) -m pip install -q -r requirements.txt

data:
	$(PY) scripts/build_pipeline.py --dataset $(DATASET) --scale $(SCALE)

eval-bm25:
	$(PY) scripts/run_bm25_eval.py --dataset $(DATASET) --scale $(SCALE)

embeddings:
	$(PY) scripts/compute_embeddings.py --dataset $(DATASET) --scale $(SCALE)

eval-embeddings:
	$(PY) scripts/run_embedding_eval.py --dataset $(DATASET) --scale $(SCALE)

compare-retrieval:
	$(PY) scripts/compare_retrieval.py --dataset $(DATASET) --scale $(SCALE)

eval-harness:
	$(PY) scripts/run_eval_harness.py --dataset $(DATASET) --scale $(SCALE)

test:
	$(PY) -m pytest tests/ -v

clean-data:
	rm -rf data/processed
