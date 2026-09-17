"""Add an opt-in sequential training sampler for the projection pilot.

Two edits, both off by default so every existing arm keeps its seeded random
sampler and its recorded trajectory:

* mnpo_config.py gains nbpo_sequential_sampler: bool = False.
* MNPOTrainer._get_train_sampler honours it before any other branch.

With one prompt's 28 pairs consecutive in the dataset and 28 rows per optimizer
step, a sequential pass makes each step cover exactly one prompt's complete pair
graph, which evaluates the all-candidate objective exactly while keeping the
loss code, the collator, the memory footprint and the row multiset unchanged.
"""
from pathlib import Path

CODE = Path("/work/nbpo_repair_20260909/code/mnpo_scripts")

CONFIG_OLD = "    nbpo_expected_dataset_manifest_sha256: Optional[str] = None\n"
CONFIG_NEW = (
    "    nbpo_expected_dataset_manifest_sha256: Optional[str] = None\n"
    "    # Opt-in for the projection pilot only: iterate training rows in the order\n"
    "    # they were written instead of sampling them. With a prompt's 28 pairs\n"
    "    # consecutive, one optimizer step then covers that prompt's complete pair\n"
    "    # graph, which is the all-candidate objective evaluated exactly. Default\n"
    "    # False, so every existing arm keeps its seeded random sampler.\n"
    "    nbpo_sequential_sampler: bool = False\n"
)

TRAINER_OLD = """    def _get_train_sampler(self, train_dataset=None):
        if getattr(self.args, "nbpo_target_mode", "sampled") == "canonical_logratio":"""
TRAINER_NEW = """    def _get_train_sampler(self, train_dataset=None):
        if getattr(self.args, "nbpo_sequential_sampler", False):
            # Projection pilot: the row order in the dataset is the contract.
            from torch.utils.data import SequentialSampler
            dataset = train_dataset if train_dataset is not None else self.train_dataset
            return SequentialSampler(dataset)
        if getattr(self.args, "nbpo_target_mode", "sampled") == "canonical_logratio":"""


def patch(path, old, new, label):
    src = path.read_text()
    if new.split("\\n")[0] in src and old not in src:
        print("already patched: %s" % label)
        return
    assert src.count(old) == 1, "%s: found %d" % (label, src.count(old))
    path.write_text(src.replace(old, new, 1))
    print("patched: %s" % label)


def main():
    patch(CODE / "mnpo_config.py", CONFIG_OLD, CONFIG_NEW, "config field")
    patch(CODE / "mnpo_trainer.py", TRAINER_OLD, TRAINER_NEW, "sampler branch")
    import ast
    for name in ("mnpo_config.py", "mnpo_trainer.py"):
        ast.parse((CODE / name).read_text())
        print("syntax ok: %s" % name)


if __name__ == "__main__":
    main()
