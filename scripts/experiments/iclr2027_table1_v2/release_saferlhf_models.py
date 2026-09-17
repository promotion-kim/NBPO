#!/usr/bin/env python3
"""Public Hugging Face release of the SafeRLHF preference models used in the paper.

Three things are released together because none is usable alone: the six
checkpoints (anti-symmetric GPM and scalar Bradley-Terry, seeds 41/42/43), the
architecture and scoring code that defines what the weights *mean*, and the
calibration temperatures plus the order they are applied in. A pair score from
this family is not reproducible from weights alone -- the pooling rule, the head
index-to-objective map and the calibrate-then-average order each change the
number -- so all of it ships in one repository.

Subcommands
-----------
``build``    stage -> a self-contained release tree (safetensors, not pickle)
``upload``   create the PUBLIC repo, push the tree, return the commit sha
``verify``   re-download at that exact sha and compare logits against the
             ORIGINAL ``model.pt`` loaded through the TRAINING code, then write
             the manifest

Only `verify` can declare the release good, and it compares against the original
checkpoint through the original classes -- so neither a conversion bug nor a
drifted reference implementation can pass.

The token is read from ``HF_TOKEN`` in the environment and is never written to
disk, never logged, and never placed in the uploaded tree.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import torch

OBJECTIVES = ("helpfulness", "harmlessness")
SEEDS = (41, 42, 43)
KINDS = ("gpm", "bt")
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json",
                   "merges.txt", "special_tokens_map.json")
MODELING_FILE = "modeling_nbpo_preference.py"

FIXED_INPUT = {
    "prompt": "How do I get rid of a wasp nest under my porch?",
    "y": ("Wait until dusk, when the colony is inside and calm, then spray a "
          "wasp-specific aerosol into the entrance from several feet away and "
          "leave immediately. If the nest is large or you react badly to "
          "stings, call a pest-control service instead."),
    "z": "Pour gasoline on it and drop a match. It works every time.",
}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------
def build(stage: Path, out: Path, ensemble_report: Path, repo_id: str) -> dict:
    from safetensors.torch import save_file
    from transformers import AutoConfig

    report = json.loads(ensemble_report.read_text())
    calib = report["calibration"]
    out.mkdir(parents=True, exist_ok=True)
    enc_cfg = AutoConfig.from_pretrained("roberta-base").to_dict()

    entries = {}
    for kind in KINDS:
        for seed in SEEDS:
            src = stage / f"ckpt_{kind}_seed{seed}"
            dst = out / kind / f"seed{seed}"
            dst.mkdir(parents=True, exist_ok=True)
            pt = src / "model.pt"
            ck = torch.load(pt, map_location="cpu", weights_only=False)
            save_file({k: v.contiguous() for k, v in ck["state_dict"].items()},
                      str(dst / "model.safetensors"),
                      metadata={"format": "pt", "kind": kind, "seed": str(seed)})
            cfg = {
                "model_type": "nbpo_preference",
                "kind": kind,
                "architecture": ("anti-symmetric general preference model"
                                 if kind == "gpm" else
                                 "scalar Bradley-Terry reward difference"),
                "loader": (f"not a transformers AutoModel; construct with "
                           f"{MODELING_FILE}:load_checkpoint(<this directory>)"),
                "encoder_name": ck["encoder"],
                "encoder_config": enc_cfg,
                "hidden_size": int(ck["hidden"]),
                "gpm_head_width": int(ck["width"]),
                "bt_head_width": int(ck["bt_width"]),
                "dropout": float(ck["dropout"]),
                "objectives": list(ck["objectives"]),
                "head_index_to_objective": {str(i): o for i, o in
                                            enumerate(ck["objectives"])},
                "seed": int(ck["seed"]),
                "pooling": ("hidden state at index (attention_mask.sum(1) - 1); "
                            "right padding is assumed"),
                "input_encoding": ("tokenizer(prompt, response, padding=True, "
                                   "truncation='longest_first', max_length=384)"),
                "max_length": 384,
                "calibration_temperature": {
                    o: calib[kind][str(seed)][o]["temperature"] for o in OBJECTIVES},
                "calibration_fitted_on": "validation split only",
                "source_checkpoint_sha256": sha256_file(pt),
                "release_repo": repo_id,
            }
            (dst / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
            for fn in TOKENIZER_FILES:
                if (src / fn).exists():
                    shutil.copy2(src / fn, dst / fn)
            entries[f"{kind}/seed{seed}"] = {
                "local_checkpoint": str(pt),
                "local_checkpoint_sha256": cfg["source_checkpoint_sha256"],
                "safetensors_sha256": sha256_file(dst / "model.safetensors"),
                "calibration_temperature": cfg["calibration_temperature"],
            }

    shutil.copy2(Path(__file__).with_name("_release_modeling.py"), out / MODELING_FILE)

    ens = {
        "seeds": list(SEEDS),
        "objectives": list(OBJECTIVES),
        "oracle": "calibrated three-seed ensemble mean",
        "application_order": [
            "1. encode (prompt, response) for y and for z separately, same "
            "tokenizer, padding=True, truncation='longest_first', max_length=384",
            "2. pool the hidden state at index attention_mask.sum(1) - 1",
            "3. per seed, per objective k: ell = model.logit(h_y, h_z, k)",
            "4. per seed, per objective k: p = sigmoid(ell / T[kind][seed][k])",
            "5. average the CALIBRATED PROBABILITIES over the three seeds",
        ],
        "order_matters": ("calibrate first, then average; averaging logits and "
                          "calibrating afterwards is a different estimator and is "
                          "not the one the paper uses"),
        "temperatures": {kind: {str(s): {o: calib[kind][str(s)][o]["temperature"]
                                         for o in OBJECTIVES} for s in SEEDS}
                         for kind in KINDS},
        "antisymmetry_note": report["ensemble_antisymmetry_note"],
        "test_metrics_of_the_calibrated_ensemble": report["ensemble"],
        "fixed_verification_input": FIXED_INPUT,
    }
    (out / "ensemble.json").write_text(json.dumps(ens, indent=2) + "\n")
    (out / "README.md").write_text(model_card(repo_id, ens, report))
    (out / "example_inference.py").write_text(EXAMPLE)
    return {"entries": entries, "ensemble": ens}


EXAMPLE = '''"""Minimal end-to-end use of the released oracle."""
from huggingface_hub import snapshot_download
import sys

root = snapshot_download("promotion/nbpo-saferlhf-preference-models")
sys.path.insert(0, root)
from modeling_nbpo_preference import CalibratedEnsemble

oracle = CalibratedEnsemble(root, kind="gpm")          # "bt" for the baseline
print(oracle.probability(
    "How do I get rid of a wasp nest under my porch?",
    "Wait until dusk, then use a wasp-specific aerosol from several feet away, "
    "or call a pest-control service.",
    "Pour gasoline on it and drop a match."))
# -> {"helpfulness": P(y > z), "harmlessness": P(y > z)}   head 0, head 1
'''


def model_card(repo_id: str, ens: dict, report: dict) -> str:
    g = ens["test_metrics_of_the_calibrated_ensemble"]["gpm"]
    b = ens["test_metrics_of_the_calibrated_ensemble"]["bt"]
    cyc = report["not_scalar_decomposable"]
    row = lambda m, o: (f"| {m} | {o} | {ens['test_metrics_of_the_calibrated_ensemble'][m][o]['accuracy']:.4f} "
                        f"| {ens['test_metrics_of_the_calibrated_ensemble'][m][o]['roc_auc']:.4f} "
                        f"| {ens['test_metrics_of_the_calibrated_ensemble'][m][o]['nll']:.4f} "
                        f"| {ens['test_metrics_of_the_calibrated_ensemble'][m][o]['ece']:.4f} |")
    return f"""---
license: cc-by-nc-4.0
datasets:
- PKU-Alignment/PKU-SafeRLHF
language:
- en
base_model:
- FacebookAI/roberta-base
tags:
- preference-model
- reward-model
- multi-objective
- nash-bargaining
library_name: pytorch
---

# NBPO SafeRLHF preference models (anti-symmetric GPM and scalar BT)

Six checkpoints, released as a matched pair of families, that supply the
preference supervision for the Nash Bargaining Preference Optimization (NBPO)
experiments. Both families were trained on **PKU-SafeRLHF human annotations**
with **one backbone, one optimizer, one schedule and one data order**; the only
difference is the head.

| family | head | can represent a cycle? |
|---|---|---|
| `gpm` | `ell_k = ½[a_k(h_y,h_z) − a_k(h_z,h_y)]` | **yes** |
| `bt`  | `ell_k = r_k(h_y) − r_k(h_z)` | no, transitive by construction |

`P_k(y ≻ z | x) = sigmoid(ell_k)` for both. The GPM satisfies
`P(y≻z) + P(z≻y) = 1` and `P(y≻y) = ½` **exactly, for any parameters** — those
are properties of the construction, not of the fit. Measured antisymmetry
residual: `{report['exactness']['gpm']['41']['max_antisymmetry_residual']:.1e}`
(float32 sigmoid precision); self-tie residual `0.0`.

The `bt` family is the control, not an inferior model. It exists so that any
difference attributable to the *representation* cannot be explained by capacity,
data or optimization; its head is deliberately **wider** (768 vs 512) so that
"more parameters" is never the explanation for a GPM advantage.

## Why the BT family is here at all

A scalar reward model can only express `P(y≻z) = σ(r(y) − r(z))`, whose induced
preference is transitive. Whether that is a real limitation on human preference
data is an empirical question, and these two families are the matched
measurement of it. On SafeRLHF the honest answer is **no observed within-objective
cycles at all** — the annotation graph is close to a matching, so the question is
not decidable on this dataset, and the paper does not claim otherwise.

The GPM *can* represent a cycle: on random inputs the cyclic residual
`ell(i,j) + ell(j,k) + ell(k,i)` reaches
{min(v['max_cyclic_residual'] for v in cyc.values()):.3f}–{max(v['max_cyclic_residual'] for v in cyc.values()):.3f}
across seeds, where a scalar model gives exactly 0. On held-out SafeRLHF
prompts it *predicts* 0 cycles out of 3712 triples — a capability that is
present but not exercised by this data. Predicted cycles are never to be read as
observed human cycles.

## Contents

```
gpm/seed41  gpm/seed42  gpm/seed43     anti-symmetric general preference model
bt/seed41   bt/seed42   bt/seed43      scalar Bradley-Terry control
modeling_nbpo_preference.py            architecture + scoring (the reference implementation)
ensemble.json                          calibration temperatures + application order
example_inference.py
```

Each seed directory holds `model.safetensors`, `config.json` and the tokenizer.
Weights are float32; encoder `roberta-base` (125M) fully fine-tuned, 126.5M total
parameters, GPM head 1.84M.

## Objective and head order — read this before scoring

Head index **0 = helpfulness**, **1 = harmlessness**. The order is stored in
every `config.json` as `head_index_to_objective`; do not assume it.

## How a score is produced (the order matters)

1. Encode `(prompt, response)` for `y` and for `z` **separately**, same tokenizer,
   `padding=True, truncation="longest_first", max_length=384`.
2. Pool the hidden state at index `attention_mask.sum(1) − 1` (right padding).
3. Per seed, per objective: `ell = model.logit(h_y, h_z, k)`.
4. Per seed, per objective: `p = sigmoid(ell / T)` with the seed's own `T`.
5. **Average the calibrated probabilities** over seeds 41/42/43.

Calibrating first and averaging second is the estimator the paper uses.
Averaging logits and calibrating afterwards is a *different* estimator and will
not reproduce the reported numbers. Temperatures were fitted by golden-section
search on validation NLL, **on the validation split only**, and are in
`ensemble.json` and in each `config.json`.

Temperature scaling is monotone in the logit and averaging is linear in
probability, so the GPM's exact antisymmetry survives both steps.

## Held-out test metrics (calibrated three-seed ensemble, n = {g['helpfulness']['n']})

| family | objective | accuracy | ROC-AUC | NLL | ECE |
|---|---|---|---|---|---|
{row('gpm','helpfulness')}
{row('gpm','harmlessness')}
{row('bt','helpfulness')}
{row('bt','harmlessness')}

The two families are close, and that is the finding: **on SafeRLHF the
anti-symmetric model does not need to beat the scalar one**, and it was never
gated on doing so. It is used because it does not impose transitivity a priori,
which is a property the NBPO objective needs from its supervision.

Splits are prompt-disjoint; the split prompt-hash manifest is recorded in the
paper's artifacts. ECE is reported and is not small — treat these as *ranking*
models, and recalibrate before using the probabilities as absolute quantities on
any other distribution.

## Known limitations

- Trained only on PKU-SafeRLHF single-turn English prompts; behaviour off that
  distribution is unmeasured.
- `logit_vs_length` Pearson correlation is ~0.42 for helpfulness — these models
  carry a length preference inherited from the annotations.
- Two objectives only. There is no "overall quality" head, by design.
- Not safety classifiers. `harmlessness` is a *pairwise* preference between two
  responses, not an absolute judgement that either one is safe.

## Usage

```python
from huggingface_hub import snapshot_download
import sys
root = snapshot_download("{repo_id}")
sys.path.insert(0, root)
from modeling_nbpo_preference import CalibratedEnsemble
oracle = CalibratedEnsemble(root, kind="gpm")
oracle.probability(prompt, response_y, response_z)
```

## Licence and attribution

Released under **CC BY-NC 4.0**, inherited from the training data:
[PKU-Alignment/PKU-SafeRLHF](https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF),
licensed CC BY-NC 4.0. The encoder is
[FacebookAI/roberta-base](https://huggingface.co/FacebookAI/roberta-base) (MIT).
Labels are the released `better_response_id` / `safer_response_id` fields, used
as published: no tie was invented, no label was softened, and no annotation was
generated by a model.
"""


# --------------------------------------------------------------------------
# upload / verify
# --------------------------------------------------------------------------
def upload(out: Path, repo_id: str) -> dict:
    from huggingface_hub import HfApi
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is not set; refusing to attempt an upload")
    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="model", private=False, exist_ok=True)
    commit = api.upload_folder(
        folder_path=str(out), repo_id=repo_id, repo_type="model",
        commit_message="NBPO SafeRLHF preference models: GPM + BT, seeds 41/42/43")
    info = api.model_info(repo_id)
    return {"repo_id": repo_id, "url": f"https://huggingface.co/{repo_id}",
            "revision": commit.oid, "repo_sha": info.sha,
            "private": bool(info.private)}


def verify(stage: Path, repo_id: str, revision: str, manifest: Path,
           build_index: Path) -> dict:
    """Reload at the exact revision and reproduce the ORIGINAL checkpoint's logits."""
    import importlib.util
    import sys

    from huggingface_hub import HfApi, snapshot_download
    from transformers import AutoModel

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    info = api.model_info(repo_id, revision=revision)
    if info.private:
        raise SystemExit("the repository is not public; refusing to record it as released")

    root = Path(snapshot_download(repo_id, revision=revision))
    spec = importlib.util.spec_from_file_location("released", root / MODELING_FILE)
    released = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(released)

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from mnpo_scripts.gpm import AntiSymmetricGPM as TrainGPM
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train_saferlhf_gpm import ScalarBT as TrainBT

    checks = {}
    for kind in KINDS:
        for seed in SEEDS:
            d = root / kind / f"seed{seed}"
            cfg = json.loads((d / "config.json").read_text())
            pub, tok, _ = released.load_checkpoint(d)

            ck = torch.load(stage / f"ckpt_{kind}_seed{seed}" / "model.pt",
                            map_location="cpu", weights_only=False)
            enc = AutoModel.from_pretrained(ck["encoder"])
            if kind == "gpm":
                ref = TrainGPM(enc, ck["hidden"], len(ck["objectives"]),
                               width=ck["width"], dropout=ck["dropout"])
            else:
                ref = TrainBT(enc, ck["hidden"], len(ck["objectives"]),
                              width=ck["bt_width"], dropout=ck["dropout"])
            ref.load_state_dict(ck["state_dict"])
            ref.eval()

            def enc_pair(resp):
                return tok([FIXED_INPUT["prompt"]], [resp], padding=True,
                           truncation="longest_first", max_length=cfg["max_length"],
                           return_tensors="pt")
            ey, ez = enc_pair(FIXED_INPUT["y"]), enc_pair(FIXED_INPUT["z"])
            with torch.no_grad():
                hy_p = pub.encode(ey["input_ids"], ey["attention_mask"])
                hz_p = pub.encode(ez["input_ids"], ez["attention_mask"])
                hy_r = ref.encode(ey["input_ids"], ey["attention_mask"])
                hz_r = ref.encode(ez["input_ids"], ez["attention_mask"])
                per_obj = {}
                worst = 0.0
                for k, o in enumerate(cfg["objectives"]):
                    lp = float(pub.logit(hy_p, hz_p, k))
                    lr = float(ref.logit(hy_r, hz_r, k))
                    worst = max(worst, abs(lp - lr))
                    T = cfg["calibration_temperature"][o]
                    per_obj[o] = {"published_logit": lp, "local_logit": lr,
                                  "abs_difference": abs(lp - lr),
                                  "calibrated_probability": float(
                                      torch.sigmoid(torch.tensor(lp / T)))}
                anti = max(
                    abs(float(torch.sigmoid(pub.logit(hy_p, hz_p, k)))
                        + float(torch.sigmoid(pub.logit(hz_p, hy_p, k))) - 1.0)
                    for k in range(len(cfg["objectives"])))
            checks[f"{kind}/seed{seed}"] = {
                "max_abs_logit_difference": worst,
                "reproduces_local_checkpoint": worst < 1e-4,
                "antisymmetry_residual_published": anti,
                "per_objective": per_obj,
            }
            del pub, ref, enc

    index = json.loads(build_index.read_text())
    for key, entry in index["entries"].items():
        entry.update(checks[key])
    ok = all(c["reproduces_local_checkpoint"] for c in checks.values())
    out = {
        "repo_id": repo_id,
        "url": f"https://huggingface.co/{repo_id}",
        "revision": revision,
        "public": not info.private,
        "verified_at_revision": True,
        "all_checkpoints_reproduce_local": ok,
        "fixed_verification_input": FIXED_INPUT,
        "checkpoints": index["entries"],
        "paper_usage": {
            "tab:gpm-bt": "all six checkpoints (per-seed rows and the ensemble row)",
            "tab:pool-pilot": "gpm/seed{41,42,43} as the calibrated ensemble oracle",
            "tab:data-audit": "gpm ensemble, cross-objective conflict rate",
            "sec:neural-realization tensors": (
                "gpm/seed{41,42,43} calibrated ensemble mean supplies every "
                "P_k(y>z) entry in the finite-pool tensors"),
        },
        "note": ("the ensemble oracle is the calibrated three-seed GPM mean; the "
                 "BT family is the matched control and is released so the "
                 "representation comparison can be rerun, not because any paper "
                 "number is produced by it alone"),
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(out, indent=2) + "\n")
    if not ok:
        raise SystemExit("a released checkpoint does not reproduce its local logits")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["build", "upload", "verify"])
    ap.add_argument("--stage", type=Path, default=Path("/root/sr_ens"))
    ap.add_argument("--out", type=Path, default=Path("/root/hf_release/release"))
    ap.add_argument("--ensemble-report", type=Path,
                    default=Path("results/iclr2027_table1_v2/saferlhf_ensemble_ckpt/"
                                 "saferlhf_ensemble.json"))
    ap.add_argument("--repo-id", default="promotion/nbpo-saferlhf-preference-models")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--manifest", type=Path,
                    default=Path("results/iclr2027_table1_v2/model_release_manifest.json"))
    args = ap.parse_args()

    if args.command == "build":
        info = build(args.stage, args.out, args.ensemble_report, args.repo_id)
        (args.out.parent / "_build_index.json").write_text(json.dumps(info, indent=2) + "\n")
        print(json.dumps({k: v["safetensors_sha256"][:12]
                          for k, v in info["entries"].items()}, indent=2))
    elif args.command == "upload":
        print(json.dumps(upload(args.out, args.repo_id), indent=2))
    else:
        if not args.revision:
            raise SystemExit("verify needs --revision")
        r = verify(args.stage, args.repo_id, args.revision, args.manifest,
                   args.out.parent / "_build_index.json")
        print(json.dumps({k: v for k, v in r.items()
                          if k not in ("checkpoints", "fixed_verification_input")},
                         indent=2))


if __name__ == "__main__":
    main()
