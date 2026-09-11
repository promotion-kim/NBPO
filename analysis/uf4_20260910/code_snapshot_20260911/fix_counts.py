"""Make the trainer's token accounting available to every loss type.

counts, input_tokens and response_tokens were initialized only under
`loss_type in ("nbpo", "nbpo_wbc")`, but the online-reference branch further
down uses all three unconditionally. Any other loss type with
nbpo_online_reference set therefore died with UnboundLocalError before its
first step -- which is what DPO with a frozen online reference is.

Hoisting the block rather than guarding the use sites is the deliberate choice:
the feedback-cost table compares arms on forward tokens, so the DPO baseline
needs the same accounting as the arms it is compared against, not a blank. For
loss_type nbpo the executed code is unchanged.
"""
import hashlib, pathlib, shutil, sys

F = pathlib.Path("/work/nbpo_repair_20260909/code/mnpo_scripts/mnpo_trainer.py")
src = F.read_text()
before = hashlib.sha256(src.encode()).hexdigest()

old = '''        if self.loss_type in ("nbpo", "nbpo_wbc"):
            if not hasattr(self, "_nbpo_token_counts"):
                self._nbpo_token_counts = {}
            counts = self._nbpo_token_counts.setdefault(train_eval, {
                "policy_forward_tokens": 0, "policy_response_tokens": 0,
                "reference_forward_tokens": 0, "reference_response_tokens": 0})
            input_tokens = int(sum(batch[f"{side}_attention_mask"].sum().item()
                                   for side in ("chosen", "rejected")))
            response_tokens = int(sum(batch[f"{side}_labels"].ne(self.label_pad_token_id).sum().item()
                                      for side in ("chosen", "rejected")))
            counts["policy_forward_tokens"] += input_tokens
            counts["policy_response_tokens"] += response_tokens
'''
new = '''        # Token accounting for every loss type, not only the NBPO ones. The
        # online-reference branch below uses counts/input_tokens/response_tokens
        # unconditionally, so scoping them to nbpo made any other loss type with
        # a live reference fail with UnboundLocalError before its first step.
        # The cost table also compares arms on forward tokens, and a baseline
        # needs the same accounting as the arms it is compared against.
        if not hasattr(self, "_nbpo_token_counts"):
            self._nbpo_token_counts = {}
        counts = self._nbpo_token_counts.setdefault(train_eval, {
            "policy_forward_tokens": 0, "policy_response_tokens": 0,
            "reference_forward_tokens": 0, "reference_response_tokens": 0})
        input_tokens = int(sum(batch[f"{side}_attention_mask"].sum().item()
                               for side in ("chosen", "rejected")))
        response_tokens = int(sum(batch[f"{side}_labels"].ne(self.label_pad_token_id).sum().item()
                                  for side in ("chosen", "rejected")))
        counts["policy_forward_tokens"] += input_tokens
        counts["policy_response_tokens"] += response_tokens
'''
if old not in src:
    sys.exit("PATTERN NOT FOUND")
if src.count(old) != 1:
    sys.exit("PATTERN NOT UNIQUE: %d" % src.count(old))
src = src.replace(old, new)

backup = F.with_suffix(".py.pre_counts_%s" % before[:12])
if not backup.exists():
    shutil.copy2(F, backup)
F.write_text(src)
print("patched; backup", backup.name)
print("before", before[:16], "after", hashlib.sha256(src.encode()).hexdigest()[:16])
