#!/usr/bin/env bash
# Fresh GPU box (Ubuntu, NVIDIA driver with CUDA 12, Ampere or newer because of flash-attn).
# Usage: bash setup_pod.sh   (then: source ~/nanotron/.venv/bin/activate)
set -euo pipefail

cd "${WORKDIR:-$HOME}"
[ -d nanotron ] || git clone https://github.com/huggingface/nanotron.git
cd nanotron
git rev-parse HEAD > ../nanotron_commit.txt
# Upstream bug on main (a1d47c7e): llama.py passes config.model to the parametrizator, which reads
# config.model.* and config.parallelism, so random init of a Llama crashes. qwen.py passes config.
sed -i 's/parametrizator = parametrizator_cls(config=config.model)/parametrizator = parametrizator_cls(config=config)/' src/nanotron/models/llama.py
# Upstream bug on main (a1d47c7e): with tp=1, _ColumnLinearNoAsyncCommunicationReduceScatterMode.backward returns
# grad_input still flattened to 2D, so autograd rejects it. Restore the input shape (a view, no copy).
sed -i 's/^            sub_grad_input = grad_input$/            sub_grad_input = grad_input.view(input_size)/' src/nanotron/parallel/tensor_parallel/functional.py
git diff --stat > ../nanotron_local_patch.txt; git diff >> ../nanotron_local_patch.txt

export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || pip install -q uv || curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv .venv --python 3.11
source .venv/bin/activate

# torch 2.5 because src/nanotron/nn/flex_attention.py needs torch.nn.attention.flex_attention.
uv pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
uv pip install -e .
# nanotron pins numpy<2; recent pyarrow needs numpy 2, and datasets 4 dropped the xPath import nanotron uses.
# Recent transformers disables itself below torch 2.6, so pin it to the SmolLM3 era (mid 2025).
uv pip install pyyaml matplotlib "datasets<4" "pyarrow<18" "numpy<2" "transformers<4.54" psutil setuptools pip
# Prebuilt wheel: building flash-attn from source takes very long on a small box.
uv pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"

# nanotron main imports grouped_gemm unconditionally (nn/moe.py, via scaling/parametrization.py), even for dense models.
uv pip install setuptools wheel ninja
TORCH_CUDA_ARCH_LIST="8.0;8.6" uv pip install --no-build-isolation "git+https://github.com/fanshiqing/grouped_gemm@efe8c40eaf4c8ef57191e0ea9aa4117aa5b1a8f2"

python - <<'EOF'
import torch, flash_attn
print("torch", torch.__version__, "| flash-attn", flash_attn.__version__)
print("gpus:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
EOF
