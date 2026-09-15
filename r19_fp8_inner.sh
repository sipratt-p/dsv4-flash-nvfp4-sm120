#!/usr/bin/env bash
set +B
unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE VLLM_B12X_MLA_EXTEND_MAX_CHUNKS
exec /opt/venv/bin/python -m vllm.entrypoints.cli.main serve /model \
  --served-model-name DeepSeek-V4-Flash --host 0.0.0.0 --port "${PORT:-8000}" \
  --kv-cache-dtype fp8_ds_mla --block-size 256 --load-format safetensors \
  --tensor-parallel-size 2 --moe-backend b12x --linear-backend b12x \
  --gpu-memory-utilization "${UTIL:-0.93}" --max-model-len "${MAXLEN:-262144}" --max-num-seqs 4 \
  --async-scheduling --no-scheduler-reserve-full-isl \
  --max-num-batched-tokens "${MNBT:-512}" --max-cudagraph-capture-size 192 \
  --enable-chunked-prefill --no-enable-prefix-caching \
  --tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4 --enable-auto-tool-choice --reasoning-parser deepseek_v4 \
  --default-chat-template-kwargs '{"thinking": true, "reasoning_effort": "max"}' \
  --disable-custom-all-reduce \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic"}'
