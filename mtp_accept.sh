#!/usr/bin/env bash
# Read the MTP acceptance rate from /metrics.
# Fire a few requests first so there are statistics to read.
set -uo pipefail
PORT="${PORT:-8000}"; EP="http://127.0.0.1:${PORT}/v1/chat/completions"
# warmup: run a few generations to accumulate spec-decode stats
for i in 1 2 3; do
  curl -s -m120 "$EP" -H 'Content-Type: application/json' \
    -d '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"Write a detailed 300-word explanation of how TCP congestion control works."}],"max_tokens":400,"temperature":0.0}' >/dev/null 2>&1
done
echo "=== spec decode / MTP metrics ==="
curl -s "http://127.0.0.1:${PORT}/metrics" 2>/dev/null | grep -iE "spec_decode|accept|draft" | grep -v '^#' | head -20
echo "=== acceptance rate ==="
# Note: vLLM's *_created metrics are Unix timestamps (~1.7e9); summing them gives a bogus 100%.
# Use only the _total suffix for accepted/draft.
curl -s "http://127.0.0.1:${PORT}/metrics" 2>/dev/null | python3 -c "
import sys,re
acc=draft=0.0
for ln in sys.stdin:
    if ln.startswith('#'): continue
    m=re.match(r'(\S+?)(\{[^}]*\})?\s+([0-9.e+-]+)\s*\$', ln)
    if not m: continue
    k,v=m.group(1),float(m.group(3))
    if k=='vllm:spec_decode_num_accepted_tokens_total': acc+=v
    if k=='vllm:spec_decode_num_draft_tokens_total':    draft+=v
if draft>0: print(f'  accepted={acc:.0f} draft={draft:.0f} -> acceptance {acc/draft*100:.1f}%  (mean accept length ~ {1+acc/draft:.2f}x)')
else: print('  (spec decode metrics not available or 0)')
"
echo "  see also vLLM's own windowed figure: docker logs --since 60s dsv4 | grep 'SpecDecoding metrics'"
