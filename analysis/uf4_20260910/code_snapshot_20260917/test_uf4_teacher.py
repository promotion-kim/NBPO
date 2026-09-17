"""The properties that would silently corrupt the UF-4 teacher if they broke."""
import json, sys, torch
sys.path.insert(0, "/work/uf4_20260910/code")
from transformers import AutoTokenizer
from train_uf4_teacher import Collator, FourHeadEncoder, OBJECTIVES, masked_metrics

BACKBONE = "/work/uf4_20260910/assets/ModernBERT-base"
tok = AutoTokenizer.from_pretrained(BACKBONE, local_files_only=True)
fails = []

def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (" | " + detail if detail else ""))
    if not ok:
        fails.append(name)

row = {"instruction": "Explain gradient descent.", "response_a": "It descends the gradient.",
       "response_b": "Pick a random point and stop.",
       "target": {o: v for o, v in zip(OBJECTIVES, [1.0, 0.5, None, 0.0])},
       "mask": {o: v for o, v in zip(OBJECTIVES, [True, True, False, True])}}

for head in ("gpm", "bt"):
    col = Collator(tok, head, 8192, 2048)
    batch = col([row])
    check(f"{head}: two encoder passes per pair", batch["input_ids"].shape[0] == 2,
          str(tuple(batch["input_ids"].shape)))
    check(f"{head}: no token_type_ids", "token_type_ids" not in batch, str(sorted(batch)))
    check(f"{head}: masked criterion is masked",
          batch["mask"].tolist() == [[True, True, False, True]], str(batch["mask"].tolist()))

    torch.manual_seed(0)
    model = FourHeadEncoder(BACKBONE, head).eval()
    with torch.no_grad():
        logits = model(batch["input_ids"], batch["attention_mask"])
    check(f"{head}: one logit per objective", tuple(logits.shape) == (1, 4), str(tuple(logits.shape)))

    # swap complement: P(A>B) + P(B>A) == 1
    swapped = dict(row, response_a=row["response_b"], response_b=row["response_a"])
    b2 = col([swapped])
    with torch.no_grad():
        l2 = model(b2["input_ids"], b2["attention_mask"])
    total = (torch.sigmoid(logits) + torch.sigmoid(l2)).abs()
    check(f"{head}: swap complement P(A>B)+P(B>A)=1",
          bool((total - 1).abs().max() < 1e-5), f"max dev {float((total-1).abs().max()):.2e}")

    # identical responses must be an exact tie
    same = dict(row, response_a="identical text", response_b="identical text")
    b3 = col([same])
    with torch.no_grad():
        l3 = model(b3["input_ids"], b3["attention_mask"])
    residual = float(l3.abs().max())
    check(f"{head}: identical responses give 0.5 to kernel precision",
          residual < 1e-4, f"max |logit| {residual:.2e} -> |P-0.5| {abs(0.5-1/(1+2.718281828**-residual)):.2e}")

# masked cells must not enter the loss or its denominator
logits = torch.tensor([[2.0, -1.0, 5.0, 0.3]])
target = torch.tensor([[1.0, 0.5, 0.0, 0.0]])
mask = torch.tensor([[True, True, False, True]])
m = masked_metrics(logits, target, mask)
check("masked cell excluded from denominator", m["n"].tolist() == [1, 1, 0, 1], str(m["n"].tolist()))
check("tie excluded from accuracy denominator", m["n_decided"].tolist() == [1, 0, 0, 1],
      str(m["n_decided"].tolist()))
loss_all = (-(target*torch.log(torch.sigmoid(logits)) + (1-target)*torch.log(1-torch.sigmoid(logits))) * mask).sum() / mask.sum()
big = logits.clone(); big[0, 2] = -50.0
loss_big = (-(target*torch.log(torch.sigmoid(big)) + (1-target)*torch.log(1-torch.sigmoid(big))) * mask).sum() / mask.sum()
check("masked logit cannot move the loss", bool((loss_all - loss_big).abs() < 1e-9),
      f"delta {float((loss_all-loss_big).abs()):.2e}")

# symmetric length budget: a long A and a long B are truncated to the same size
col = Collator(tok, "gpm", 256, 64)
long_row = {"instruction": "q " * 300, "response_a": "a " * 4000, "response_b": "b " * 4000,
            "target": {o: 1.0 for o in OBJECTIVES}, "mask": {o: True for o in OBJECTIVES}}
b = col([long_row])
ids = b["input_ids"][0].tolist()
text = tok.decode([i for i in ids if i != tok.pad_token_id])
check("oversize sequences counted once each",
      col.stats["oversize_sequences"] == 2 and col.stats["sequences"] == 2
      and col.stats["pairs"] == 1, json.dumps(col.stats))
check("length budget respected", b["input_ids"].shape[1] <= 256, str(b["input_ids"].shape))
check("both responses survive truncation", ("RESPONSE A" in text and "RESPONSE B" in text), text[:80])
na, nb = text.count(" a"), text.count(" b")
check("A and B truncated to the same size", abs(na - nb) <= 1, f"a={na} b={nb}")

print(json.dumps({"failures": fails}))
sys.exit(1 if fails else 0)
