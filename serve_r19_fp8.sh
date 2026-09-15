#!/usr/bin/env bash
# jacklarmer-informed config: FP8 0731 ckpt, b12x MoE, NO sparse-MLA attention,
# FP8_GEMM off (DSpark drafter warmup asserts otherwise), DSpark nst=5.
set -uo pipefail
IMG="voipmonitor/vllm:gilded-gnosis-v20-vllm6634ec9-sib2bff71-fi801d57a-cu132-20260801-r19"
MODEL_DIR="${MODEL_DIR:-$HOME/models/DeepSeek-V4-Flash-0731}"
PORT="${PORT:-8000}"; NAME="${NAME:-dsv4r19fp8}"
DIR="$(dirname "$0")"
docker rm -f "$NAME" 2>/dev/null
docker run -d --name "$NAME" --gpus all --runtime nvidia --ipc host --shm-size 32g --network host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$MODEL_DIR":/model:ro \
  -v "$DIR/r19_fp8_inner.sh":/inner.sh:ro \
  -e CUDA_VISIBLE_DEVICES=0,1 -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e NCCL_P2P_DISABLE=1 -e VLLM_ENABLE_PCIE_ALLREDUCE=0 \
  -e NCCL_IB_DISABLE=1 -e NCCL_PROTO=LL,LL128,Simple \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e VLLM_USE_B12X_FP8_GEMM=0 -e VLLM_USE_B12X_MOE=1 -e VLLM_USE_DEEP_GEMM=0 -e VLLM_USE_B12X_WO_PROJECTION=1 \
  -e CUTE_DSL_ARCH=sm_120a -e VLLM_PREFIX_CACHE_RETENTION_INTERVAL=4096 \
  -e VLLM_USE_AOT_COMPILE=1 -e VLLM_USE_BREAKABLE_CUDAGRAPH=0 -e VLLM_USE_MEGA_AOT_ARTIFACT=1 \
  -e VLLM_MEMORY_PROFILE_INCLUDE_ATTN=1 -e B12X_MHC_MAX_TOKENS=49152 -e VLLM_USE_B12X_MHC=1 \
  -e VLLM_USE_B12X_SPARSE_INDEXER=1 -e B12X_MLA_SM120_UNIFIED=1 -e USES_B12X=True \
  -e B12X_DENSE_SPLITK_TURBO=1 -e B12X_W4A16_TC_DECODE=1 \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e PORT="$PORT" -e UTIL="${UTIL:-0.93}" -e MAXLEN="${MAXLEN:-262144}" -e MNBT="${MNBT:-512}" \
  --entrypoint /bin/bash "$IMG" /inner.sh
echo "==> launched $NAME on :$PORT"
