DATASET ?= all
SCALE ?= demo
PY := .venv/bin/python

.PHONY: venv data test clean-data

venv:
	python3 -m venv .venv
	$(PY) -m pip install -q -r requirements.txt

data:
	$(PY) scripts/build_pipeline.py --dataset $(DATASET) --scale $(SCALE)

test:
	$(PY) -m pytest tests/ -v

clean-data:
	rm -rf data/processed
