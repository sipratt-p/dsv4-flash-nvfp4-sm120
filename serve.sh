#!/usr/bin/env bash
# Serve DeepSeek-V4-Flash (NVFP4) on SM120 (RTX PRO 6000).
#
# The earlier run_dsv4.sh had rotted (image name, ENTRYPOINT, MODEL_DIR and MoE backend had all drifted),
# so this rebuilds a working launch from scratch. Full background in README.md.
#
# The load-bearing differences (why nothing else boots):
#   - Use the non-REAP nvidia/DeepSeek-V4-Flash-NVFP4 (256 experts). The REAP builds (180B=160 / 162B=144 experts)
#     are rejected by the image's MoE kernel with "Unsupported expert number".
#   - The dsv4-flash-sm120:local image's ENTRYPOINT is ["vllm serve"], so pass CMD as `serve /model ...`
#     via `--entrypoint vllm` (`vllm serve /model` doubles the verb and dies).
#   - SM120 has no native NVFP4 MoE backend (FLASHINFER_TRTLLM / CUTEDSL are gated to family-100).
#     VLLM_TEST_FORCE_FP8_MARLIN=1 forces the MARLIN backend -> weight-only FP4 dequant, works but slower.
#   - Do not add --enable-expert-parallel (with or without EP, MARLIN is what fixes the MoE backend problem).
#
# usage: ./serve.sh           # start with fp8 KV
#        KV=fp8 ./serve.sh
set -uo pipefail

MODEL_DIR="${MODEL_DIR:-./models/DeepSeek-V4-Flash-NVFP4}"
IMAGE="${IMAGE:-dsv4-flash-sm120:local}"
PORT="${PORT:-8000}"
NAME="${NAME:-dsv4}"
KV="${KV:-fp8}"            # this image has no nvfp4 KV support (CacheConfig stops at fp8)
MAXLEN="${MAXLEN:-131072}"
UTIL="${UTIL:-0.95}"

if [ ! -f "$MODEL_DIR/model.safetensors.index.json" ]; then
  echo "!! model weights not found in $MODEL_DIR" >&2
  echo "   hf download nvidia/DeepSeek-V4-Flash-NVFP4 --local-dir $MODEL_DIR" >&2
  exit 1
fi

docker rm -f "$NAME" 2>/dev/null

docker run -d --name "$NAME" \
  --gpus all --ipc=host --shm-size=64g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -e VLLM_TEST_FORCE_FP8_MARLIN=1 \
  -e VLLM_TRITON_MLA_SPARSE=1 \
  -e VLLM_TRITON_MLA_SPARSE_TOPK_CHUNK_SIZE=256 \
  -e VLLM_TRITON_MLA_SPARSE_QUERY_CHUNK_SIZE=128 \
  -e VLLM_TRITON_MLA_SPARSE_MATMUL_DECODE=0 \
  -e VLLM_TRITON_MLA_SPARSE_HEAD_BLOCK_SIZE=4 \
  -e VLLM_USE_FLASHINFER_SAMPLER=0 \
  -e NCCL_P2P_DISABLE=1 -e NCCL_IB_DISABLE=1 \
  -e NCCL_SOCKET_IFNAME=lo -e GLOO_SOCKET_IFNAME=lo \
  -e NCCL_DEBUG=WARN -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -v "$MODEL_DIR":/model:ro \
  --network host \
  --entrypoint vllm "$IMAGE" \
  serve /model \
    --served-model-name deepseek-v4-flash \
    --host 0.0.0.0 --port "$PORT" \
    --trust-remote-code \
    --tensor-parallel-size 2 \
    --disable-custom-all-reduce \
    --kv-cache-dtype "$KV" \
    --block-size 256 \
    --gpu-memory-utilization "$UTIL" \
    --max-model-len "$MAXLEN" \
    --tokenizer-mode deepseek_v4 \
    --tool-call-parser deepseek_v4 \
    --enable-auto-tool-choice \
    --reasoning-parser deepseek_v4

echo "==> launched '$NAME' on :$PORT  (KV=$KV, MoE backend=MARLIN, model=$MODEL_DIR)"
echo "    wait for ready:  ./ready.sh    smoke test:  ./smoke.sh    stop:  ./stop.sh"
