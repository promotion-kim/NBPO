"""Expose the bt_reward representation in the UF-4 solver.

BT-RM--Nash is the main-body control that changes the preference
representation: scalar Bradley-Terry rewards per objective instead of a game
value. BTRewardRepresentation already implements it in the training library and
the score chunks already carry r_bt, so the solver only needs to pass them
through. The patch is additive -- adaptive_game and fixed_reference keep the
exact code path that produced the published targets.
"""
import hashlib, pathlib, shutil, sys

F = pathlib.Path("/work/uf4_20260910/code/solve_uf4_targets.py")
src = F.read_text()
before = hashlib.sha256(src.encode()).hexdigest()

edits = []

# 1) import
old = """from mnpo_scripts.nbpo_representations import (
    AdaptiveGameRepresentation, FixedReferenceRepresentation,
)"""
new = """from mnpo_scripts.nbpo_representations import (
    AdaptiveGameRepresentation, BTRewardRepresentation, FixedReferenceRepresentation,
)"""
edits.append((old, new, "import BTRewardRepresentation"))

# 2) load_scores also returns the BT reward table
old = '''    \"\"\"prompt_id -> (A_policy[K,8,8], A_ref[K,8,8]), every chunk hash-verified.\"\"\"'''
new = '''    \"\"\"prompt_id -> (A_policy[K,8,8], A_ref[K,8,8], r_bt[K,2,8]), hash-verified.

    r_bt is the frozen BT head's scalar reward for the eight learner and eight
    comparator occurrences. It is loaded for every representation but only the
    bt_reward representation reads it, so nothing else changes.
    \"\"\"'''
edits.append((old, new, "load_scores docstring"))

old = """            A, Aref = arrays["A_policy"], arrays["A_ref"]
            if A.shape[0] != len(OBJECTIVES) or A.shape[2:] != (POOL, POOL):
                raise ValueError(f"Unexpected score tensor shape in {path}: {A.shape}")
            for index, pid in enumerate(pids):
                if pid in scores:
                    raise ValueError(f"Duplicate scored prompt {pid}")
                scores[pid] = (A[:, index], Aref[:, index])"""
new = """            A, Aref, Rbt = arrays["A_policy"], arrays["A_ref"], arrays["r_bt"]
            if A.shape[0] != len(OBJECTIVES) or A.shape[2:] != (POOL, POOL):
                raise ValueError(f"Unexpected score tensor shape in {path}: {A.shape}")
            if Rbt.shape[0] != len(OBJECTIVES) or Rbt.shape[2:] != (2, POOL):
                raise ValueError(f"Unexpected BT reward shape in {path}: {Rbt.shape}")
            for index, pid in enumerate(pids):
                if pid in scores:
                    raise ValueError(f"Duplicate scored prompt {pid}")
                scores[pid] = (A[:, index], Aref[:, index], Rbt[:, index])"""
edits.append((old, new, "load_scores returns r_bt"))

# 3) CLI choice
old = '''    ap.add_argument("--representation", default="adaptive_game",
                    choices=("adaptive_game", "fixed_reference"),
                    help="fixed_reference freezes the comparator at mu: the beta -> infinity "
                         "limit, where q carries no learner dependence and the proximal solve "
                         "is exact in one map.")'''
new = '''    ap.add_argument("--representation", default="adaptive_game",
                    choices=("adaptive_game", "fixed_reference", "bt_reward"),
                    help="fixed_reference freezes the comparator at mu: the beta -> infinity "
                         "limit, where q carries no learner dependence and the proximal solve "
                         "is exact in one map. bt_reward replaces the game value by a frozen "
                         "scalar Bradley-Terry reward per objective, so there is no opponent "
                         "at all; it changes the preference representation, not the rule.")'''
edits.append((old, new, "CLI representation choice"))

# 4) BT normalization frozen on train, reused on dev
old = """    shared_meta, training, outputs = None, None, {}"""
new = """    shared_meta, training, outputs = None, None, {}
    # The BT reward scale is fit once, on the policy_train comparator pool, and
    # reused unchanged on dev -- the same discipline the dual weights follow, so
    # dev stays a held-out measurement rather than a second fit.
    bt_norm = None"""
edits.append((old, new, "bt_norm init"))

old = """        tensor_dir = out / split / "tensor"
        tensor_dir.mkdir(parents=True, exist_ok=True)"""
new = """        Rbt = np.stack([scores[pid][2] for pid in pids], axis=1)     # (K, X, 2, POOL)
        if args.representation == "bt_reward":
            r_learner_raw, r_ref_raw = Rbt[:, :, 0, :], Rbt[:, :, 1, :]
            if bt_norm is None:
                centre = r_ref_raw.reshape(len(OBJECTIVES), -1).mean(axis=1)
                scale = r_ref_raw.reshape(len(OBJECTIVES), -1).std(axis=1, ddof=1)
                if not np.all(scale > 0):
                    raise ValueError("A BT objective has zero spread on the reference pool")
                bt_norm = {"objectives": list(OBJECTIVES),
                           "mu_k_ref": [float(x) for x in centre],
                           "sigma_k_ref": [float(x) for x in scale],
                           "fitted_on": "policy_train comparator pool, all 8 occurrences",
                           "reused_on": "policy_dev, never refit"}
            centre = np.asarray(bt_norm["mu_k_ref"])[:, None, None]
            scale = np.asarray(bt_norm["sigma_k_ref"])[:, None, None]
            r_learner = (r_learner_raw - centre) / scale
            r_reference = (r_ref_raw - centre) / scale

        tensor_dir = out / split / "tensor"
        tensor_dir.mkdir(parents=True, exist_ok=True)"""
edits.append((old, new, "bt normalization"))

# 5) representation construction
old = """        else:
            rep = FixedReferenceRepresentation(
                torch.from_numpy(A), torch.from_numpy(Aref),
                uniform_policy(len(pids), POOL),
                reference_construction="shared_pool")"""
new = """        elif args.representation == "fixed_reference":
            rep = FixedReferenceRepresentation(
                torch.from_numpy(A), torch.from_numpy(Aref),
                uniform_policy(len(pids), POOL),
                reference_construction="shared_pool")
        elif args.representation == "bt_reward":
            rep = BTRewardRepresentation(
                torch.from_numpy(r_learner), torch.from_numpy(r_reference),
                uniform_policy(len(pids), POOL), normalization=bt_norm)
        else:
            raise ValueError(f"Unhandled representation {args.representation!r}")"""
edits.append((old, new, "representation branch"))

# 6) provenance
old = """                 "representation": args.representation, "eta": args.eta,
                 "dual_weights_scope": "global across prompts","""
new = """                 "representation": args.representation, "eta": args.eta,
                 "bt_reward_normalization": bt_norm,
                 "dual_weights_scope": "global across prompts","""
edits.append((old, new, "record bt normalization"))

for old, new, label in edits:
    if old not in src:
        sys.exit("PATTERN NOT FOUND: %s" % label)
    if src.count(old) != 1:
        sys.exit("PATTERN NOT UNIQUE (%d): %s" % (src.count(old), label))
    src = src.replace(old, new)
    print("applied:", label)

backup = F.with_suffix(".py.pre_btrm_%s" % before[:12])
if not backup.exists():
    shutil.copy2(F, backup)
F.write_text(src)
print("backup:", backup.name)
print("sha256 before:", before)
print("sha256 after :", hashlib.sha256(src.encode()).hexdigest())
