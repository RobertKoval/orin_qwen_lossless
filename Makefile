# Portable build and verification entry points. CUDA targets require a compatible GPU environment.
include versions.env
PYTHON ?= python3
LLAMA_WORK ?= engines/llamacpp/src
LLAMA_BUILD ?= $(LLAMA_WORK)/build
LLAMA_TEST_BACKEND ?= CUDA0
LLAMA_TEST_OUT ?= build/tests/llamacpp
PYBIN = $(shell dirname "$$(command -v $(firstword $(PYTHON)))")

.PHONY: test-core-host test-core test-vllm test-python-host test-llamacpp verify-xengine build-llamacpp check-patches check-links wheel

test-core-host:
	cmake -S core -B build/core -DBF16N_CUDA=OFF
	cmake --build build/core
	ctest --test-dir build/core --output-on-failure

test-core: test-core-host
	PATH="$(PYBIN):$$PATH" $(PYTHON) -m pytest tests/core -q

test-vllm:
	PATH="$(PYBIN):$$PATH" $(PYTHON) -m pytest tests/vllm -q

test-python-host:
	$(PYTHON) -m pytest --noconftest tests/test_frozen_prompts.py tests/test_compare_outputs.py tests/llamacpp -q

test-llamacpp:
	$(PYTHON) tests/llamacpp/run_tests.py --source "$(LLAMA_WORK)" --build "$(LLAMA_BUILD)" \
	    --out "$(LLAMA_TEST_OUT)" --backend "$(LLAMA_TEST_BACKEND)" --cxx "$(CXX)" $(LLAMA_TEST_ARGS)

verify-xengine:
	$(PYTHON) bench/xengine/verify_ops.py $(XENGINE_ARGS)

build-llamacpp:
	engines/llamacpp/build.sh $(LLAMA_WORK)

check-patches:
	engines/llamacpp/check-patches.sh

check-links:
	$(PYTHON) tools/check_links.py .

wheel:
	$(PYTHON) -m pip wheel . --no-build-isolation --no-deps -w dist
