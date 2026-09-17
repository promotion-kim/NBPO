import json
from pathlib import Path
tot = {"prompts":0,"events":0,"tokens":0,"seconds":0.0,"capped":0.0,"dups":0,"bytes":0}
for s in range(4):
    d = Path("/work/uf4_20260910/pools/v1/shard%d" % s)
    c = json.loads((d / ("complete_shard%d.json" % s)).read_text())
    print("shard%d: in_shard %d chunks_now %d events %d tokens %d wall %.0fs trunc %d"
          % (s, c["n_prompts_in_shard"], c["chunks_run_now"], c["new_events"],
             c["new_response_tokens"], c["wall_seconds"], c["prompt_truncated"]))
    for m in sorted(d.glob("chunk*.manifest.json")):
        r = json.loads(m.read_text())
        tot["prompts"] += r["n_prompts"]; tot["events"] += r["n_events"]
        tot["tokens"] += r["response_tokens"]; tot["seconds"] += r["seconds"]
        tot["capped"] += r["capped_fraction"] * r["n_events"]
        tot["dups"] += r["duplicate_occurrences"]; tot["bytes"] += r["bytes"]
per_gpu = tot["seconds"] / 4
print()
print("TOTAL prompts %d events %d response_tokens %d" % (tot["prompts"], tot["events"], tot["tokens"]))
print("generate seconds summed over 4 GPUs %.1f -> %.1f s wall per GPU" % (tot["seconds"], per_gpu))
print("aggregate tokens/s across 4 GPUs %.1f" % (tot["tokens"] / per_gpu))
print("capped(length) fraction %.4f   duplicate occurrences %d" % (tot["capped"]/tot["events"], tot["dups"]))
print("bytes %d -> %d per prompt" % (tot["bytes"], tot["bytes"]/tot["prompts"]))
print("PROJECTED policy_train 10k: %.1f GB, %.1f min generation wall on 4 GPUs"
      % (tot["bytes"]/tot["prompts"]*10000/1e9, tot["seconds"]/tot["prompts"]*10000/4/60))
tr = json.loads(Path("/work/uf4_20260910/pools/v1/shard0/prompt_truncation.json").read_text())
print("prompt truncation: %d of %d prompts exceeded cap %d" % (tr["n_truncated"], tr["n_prompts"], tr["max_prompt_tokens"]))
