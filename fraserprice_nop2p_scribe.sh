#!/usr/bin/env bash

set -euo pipefail

HF_REPO="${HF_REPO:-fraserprice/DeepSeek-V4-Flash-DSpark}"
MODEL_DIR="${MODEL_DIR:-}"
IMAGE="${IMAGE:-voipmonitor/vllm:eldritch-enlightenment-v2226f26-b12x15cd38c-cu132-20260629}"
SERVED_NAME="${SERVED_NAME:-DeepSeek-V4-Flash-DSpark}"
PORT="${PORT:-8000}"
GPUS="${GPUS:-0,1}"
TP="${TP:-2}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
DSPARK_TOKENS="${DSPARK_TOKENS:-5}"

command -v hf >/dev/null 2>&1 || pip install -q --upgrade huggingface_hub
if [ -n "$MODEL_DIR" ]; then
  [ -f "$MODEL_DIR/config.json" ] || hf download "$HF_REPO" --local-dir "$MODEL_DIR"
else
  MODEL_DIR="$(hf download "$HF_REPO")"
fi
echo ">> weights at $MODEL_DIR"

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo ">> pulling image $IMAGE"
  docker pull "$IMAGE"
else
  echo ">> image present: $IMAGE"
fi

echo ">> serving '$SERVED_NAME' on http://localhost:$PORT (TP=$TP, GPUs=$GPUS, DSpark draft=$DSPARK_TOKENS)"
exec docker run --rm --name ds4-flash-dspark \
  --init --ipc=host --shm-size 32g --privileged --gpus all \
  -p "${PORT}:${PORT}" \
  -v "${MODEL_DIR}:/model:ro" \
  -v ds4-jit-cache:/cache/jit \
  -e HF_HUB_OFFLINE=1 \
  -e SAFETENSORS_FAST_GPU=1 \
  -e TORCHINDUCTOR_CACHE_DIR=/cache/jit \
  -e TRITON_CACHE_DIR=/cache/jit \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e CUDA_VISIBLE_DEVICES="${GPUS}" \
  -e CUTE_DSL_ARCH=sm_120a \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 \
  -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
  -e NCCL_IB_DISABLE=1 \
  -e NCCL_P2P_LEVEL=SYS -e NCCL_P2P_DISABLE=1 \
  -e NCCL_PROTO=LL,LL128,Simple \
  -e VLLM_ENABLE_PCIE_ALLREDUCE=0 \
  -e VLLM_PCIE_ALLREDUCE_BACKEND=b12x \
  -e VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE=64KB \
  "${IMAGE}" \
  /opt/venv/bin/python -m vllm.entrypoints.cli.main serve /model \
  --served-model-name "${SERVED_NAME}" \
  --host 0.0.0.0 --port "${PORT}" \
  --tensor-parallel-size "${TP}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs 16 \
  --reasoning-parser deepseek_v4 \
  --default-chat-template-kwargs "{\"thinking\": true, \"reasoning_effort\": \"max\"}" \
  --enable-auto-tool-choice \
  --tool-call-parser deepseek_v4 \
  --max-num-batched-tokens 8192 \
  --attention-backend FLASHINFER_MLA_SPARSE_DSV4 \
  --kv-cache-dtype fp8 \
  --speculative-config "{\"method\":\"dspark\",\"num_speculative_tokens\":${DSPARK_TOKENS},\"draft_sample_method\":\"probabilistic\"}" \
  --trust-remote-code \
  --tokenizer-mode deepseek_v4 \
  --block-size 256 \
  --max-cudagraph-capture-size 64 \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}' \
  --async-scheduling \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --enable-flashinfer-autotune \
  --kernel-config.moe_backend flashinfer_cutlass \
  --disable-custom-all-reduce \
  --default-chat-template-kwargs.thinking=true \
  --default-chat-template-kwargs.reasoning_effort=high
