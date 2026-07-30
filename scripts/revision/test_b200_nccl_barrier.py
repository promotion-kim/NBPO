import os

import torch
import torch.distributed as dist


rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(rank)
dist.init_process_group("nccl", device_id=torch.device(f"cuda:{rank}"))
value = torch.tensor([dist.get_rank() + 1.0], device="cuda")
dist.all_reduce(value)
dist.barrier(device_ids=[rank])
print(f"rank={dist.get_rank()} device={rank} value={value.item()}", flush=True)
dist.destroy_process_group()
