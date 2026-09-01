PY ?= python

.PHONY: help setup setup-force check-cuda fixture test test-fast test-ci eval clean

# torch must come from the CUDA index matching the installed toolkit, or the
# rasteriser extension will not build against it. See requirements.txt.
CUDA_INDEX ?= https://download.pytorch.org/whl/cu130
TORCH := torch==2.9.0 torchvision==0.24.0

help:
	@echo "setup       install pinned dependencies"
	@echo "setup-force replace a torch built against a different CUDA"
	@echo "check-cuda  report the torch build and whether the kernel loads"
	@echo "fixture     regenerate the synthetic CI model"
	@echo "test        full suite against whichever FLAME model is on disk"
	@echo "test-fast   skip the training suites"
	@echo "test-ci     full suite against the synthetic fixture only"
	@echo "eval        NoW evaluation (requires Docker; see README)"

# --index-url, NOT --extra-index-url: with the latter pip may resolve
# torch==2.9.0 from PyPI instead, because the version matches and only the
# local build tag (+cu130) differs. Install torch first and alone, then
# everything else from the default index.
setup:
	$(PY) -m pip install --index-url $(CUDA_INDEX) $(TORCH)
	$(PY) -m pip install -r requirements.txt
	@$(MAKE) --no-print-directory check-cuda

# pip treats 2.9.0+cu128 and 2.9.0+cu130 as the same version and will not
# replace one with the other, so switching CUDA builds needs --force-reinstall.
# --no-deps because reinstalling torch alone must not drag its companions back
# to a mismatched build.
setup-force:
	$(PY) -m pip install --index-url $(CUDA_INDEX) --force-reinstall --no-deps $(TORCH)
	@$(MAKE) --no-print-directory check-cuda

check-cuda:
	@$(PY) -c "import torch; print(f'torch {torch.__version__}   cuda {torch.version.cuda}   gpu {torch.cuda.is_available()}')"
	@$(PY) -c "from face3d.render import raster_cuda as r; print('cuda rasteriser:', 'ok' if r.available() else r.reason())"

fixture:
	$(PY) tests/fixtures/make_fixture.py

test:
	$(PY) tests/run_tests.py

test-fast:
	$(PY) tests/run_tests.py smoke_flame test_flame_torch test_render test_interface

test-ci: fixture
	FACE3D_MODEL=tests/fixtures/tiny_head.pkl $(PY) tests/run_tests.py

eval-image:
	cd now_evaluation-main && docker build -t noweval .

eval-check:
	$(PY) scripts/eval/now_validate_harness.py identity --subjects 5
	$(PY) scripts/eval/now_validate_harness.py mean --subjects 20

eval:
	@echo "Run 'make eval-check' to validate the harness first."
	@echo "Prediction + scoring of a trained encoder is not wired up yet."

clean:
	find . -name __pycache__ -type d -prune -exec rm -rf {} + ; rm -rf out/*.obj out/*.png
