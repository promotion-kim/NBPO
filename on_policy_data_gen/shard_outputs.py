import argparse
import json
import math
import os


def load_records(path):
    suffix = os.path.splitext(path)[1].lower()
    if suffix in {".jsonl", ".jsonlines", ".ljson"}:
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
    raise ValueError(f"Unsupported input format: {suffix}")


def main():
    parser = argparse.ArgumentParser(description="Shard generated all_outputs files for parallel reward scoring.")
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_shards", type=int, required=True)
    parser.add_argument("--prefix", default="shard")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")

    records = load_records(args.input_file)
    if args.max_samples is not None:
        records = records[:args.max_samples]

    os.makedirs(args.output_dir, exist_ok=True)
    shard_size = math.ceil(len(records) / args.num_shards) if records else 0

    manifest = []
    for shard_idx in range(args.num_shards):
        start = shard_idx * shard_size
        end = min(start + shard_size, len(records))
        shard_records = records[start:end] if shard_size else []
        shard_path = os.path.join(args.output_dir, f"{args.prefix}_{shard_idx}.jsonl")
        with open(shard_path, "w", encoding="utf-8") as f:
            for record in shard_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        manifest.append({"path": shard_path, "num_records": len(shard_records)})

    manifest_path = os.path.join(args.output_dir, f"{args.prefix}_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote {len(records)} records into {args.num_shards} shards under {args.output_dir}")


if __name__ == "__main__":
    main()
