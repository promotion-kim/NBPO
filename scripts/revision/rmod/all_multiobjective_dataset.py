"""Evaluation-only dataset wrapper that retains every manifest prompt."""

from robust_multi_objective_decoding.data.multiobjective_dataset import MultiObjectiveDataset


class AllMultiObjectiveDataset(MultiObjectiveDataset):
    def __init__(self, *args, split="test", shard_index=0, num_shards=1, **kwargs):
        del split
        super().__init__(*args, split="dev", **kwargs)
        if not 0 <= shard_index < num_shards:
            raise ValueError(f"invalid shard {shard_index}/{num_shards}")
        if num_shards > 1:
            self.data = self.data.shard(
                num_shards=num_shards, index=shard_index, contiguous=True
            )
