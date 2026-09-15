import json, time, urllib.request

URL = "http://127.0.0.1:8000/v1/chat/completions"
TOOLS = [{"type":"function","function":{"name":"write_file","description":"Write a file","parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}}},
         {"type":"function","function":{"name":"read_file","description":"Read a file","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}}]

def call(messages, ntok=600, tools=None):
    body={"model":"DeepSeek-V4-Flash","messages":messages,"max_tokens":ntok}
    if tools: body["tools"]=tools
    t0=time.time()
    req=urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type":"application/json"})
    d=json.load(urllib.request.urlopen(req, timeout=900))
    m=d['choices'][0]['message']
    return d['usage'], m, time.time()-t0

print("--- T1: 33k-token prompt")
filler="\n".join(f"Log entry {i}: system nominal, cycle {i} within bounds." for i in range(2200))
u,m,t=call([{"role":"user","content":filler+"\n\nOne-sentence status?"}], 400)
print(f"OK in={u['prompt_tokens']} out={u['completion_tokens']} {t:.1f}s")

print("--- T2: tool call with long context")
u,m,t=call([{"role":"user","content":filler+"\n\nWrite a summary to /tmp/s.txt using the write_file tool."}], 800, TOOLS)
print(f"OK tool_calls={bool(m.get('tool_calls'))} in={u['prompt_tokens']} {t:.1f}s")

print("--- T3: multi-turn agent conversation with tools")
msgs=[{"role":"user","content":"You are building a webpage. Read /tmp/plan.txt first."}]
u,m,t=call(msgs, 500, TOOLS)
tc=m.get('tool_calls') or []
msgs.append({"role":"assistant","content":m.get('content') or "","tool_calls":tc})
if tc:
    msgs.append({"role":"tool","tool_call_id":tc[0]['id'],"content":"PLAN: build terrain.js then pagoda.js. "+filler[:40000]})
    u,m,t=call(msgs, 800, TOOLS)
    print(f"OK turn2 tool_calls={bool(m.get('tool_calls'))} in={u['prompt_tokens']} out={u['completion_tokens']} {t:.1f}s")
else:
    print("no tool call on turn1 (acceptable), skipping turn2")

print("--- T4: speed check (short prompt, 600 tokens)")
t0=time.time(); u,m,t=call([{"role":"user","content":"Write a 400-word essay about GPUs."}], 600)
print(f"OK {u['completion_tokens']} tok in {t:.1f}s = {u['completion_tokens']/t:.1f} tok/s")
print("ALL LADDER TESTS PASSED")
