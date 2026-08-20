PY ?= python

.PHONY: help setup fixture test test-fast test-ci eval clean

help:
	@echo "setup      install pinned dependencies (CUDA 12.8 build of torch)"
	@echo "fixture    regenerate the synthetic CI model"
	@echo "test       full suite against whichever FLAME model is on disk"
	@echo "test-fast  skip the training suites"
	@echo "test-ci    full suite against the synthetic fixture only"
	@echo "eval       NoW evaluation (requires Docker; see README)"

setup:
	$(PY) -m pip install -r requirements.txt \
		--extra-index-url https://download.pytorch.org/whl/cu128

fixture:
	$(PY) scripts/make_fixture.py

test:
	$(PY) scripts/run_tests.py

test-fast:
	$(PY) scripts/run_tests.py smoke_flame test_flame_torch test_render test_interface

test-ci: fixture
	FACE3D_MODEL=tests/fixtures/tiny_head.pkl $(PY) scripts/run_tests.py

eval:
	@echo "Not yet wired up — needs Docker and the now_evaluation image."
	@echo "See README, 'Evaluation'."

clean:
	rm -rf out/*.obj out/*.png __pycache__ face3d/__pycache__ scripts/__pycache__
