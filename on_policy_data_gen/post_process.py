import json
import argparse
import os
import re

parser = argparse.ArgumentParser()
parser.add_argument("--generation_file_dir", type=str, help="Diretory containing the generation files", default="datasets/gemma2_ultrafeedback")
args = parser.parse_args()

print(args)

all_data = []
for file_name in os.listdir(args.generation_file_dir):
    if re.fullmatch(r"output_\d+\.json", file_name):
        generation_file = os.path.join(args.generation_file_dir, file_name)
        with open(generation_file, 'r', encoding='utf-8') as f:
            output_data = json.load(f)
            all_data.append(output_data)

if not all_data:
    raise ValueError(f"No output_<seed>.json files found in {args.generation_file_dir}")

num_samples = len(all_data[0])
all_res = []
num_identical = 0
for i in range(num_samples):
    prompt = all_data[0][i]["prompt"]
    gen_text = []
    for data in all_data:
        gen_text.append(data[i]["generated_text"])

    if len(set(gen_text)) == 1:
        # filter out samples where all generated responses are identical
        num_identical += 1
        continue

    all_res.append(
        {
            "prompt": prompt,
            "all_generated_responses": gen_text,
        }
    )

print(f"Filtered out {num_identical} samples with identical generated responses")

with open(os.path.join(args.generation_file_dir, 'all_outputs.json'), 'w', encoding='utf-8') as f:
    json.dump(all_res, f, ensure_ascii=False, indent=4)

print(f"Processed outputs saved to {os.path.join(args.generation_file_dir, 'all_outputs.json')}")
