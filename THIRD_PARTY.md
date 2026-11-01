# Third-party code

This repository is MIT-licensed ([LICENSE](LICENSE)). It contains or modifies code from:

* **llama.cpp / ggml** (MIT License, Copyright (c) 2023-2026 The ggml authors, https://github.com/ggml-org/llama.cpp):
  * `engines/llamacpp/patches/` are modifications of llama.cpp at the commit pinned in `versions.env`, including
    backports of upstream changes;
  * `core/src/mmid_helper.cu` is a port of `ggml/src/ggml-cuda/mmid.cu`, and `core/include/bf16n/compat.cuh` reproduces
    a few ggml-cuda helpers (`common.cuh`, `unary.cuh`); the encoder and kernels in `core/` were extracted from this
    project's own llama.cpp patch;
  * the ggml-linked benches (`engines/llamacpp/bench/`, `bench/xengine/ggml_ref.cpp`) use ggml's public API.
* **vLLM** (Apache License 2.0, https://github.com/vllm-project/vllm): `engines/vllm/vllm_bf16n` subclasses and calls
  vLLM classes; it does not copy vLLM source files. The non-BF16N branch of the shared-expert forward in `model.py`
  restates vLLM's three-line `Qwen2MoeMLP` forward.

The MIT notice of llama.cpp / ggml:

```
MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
