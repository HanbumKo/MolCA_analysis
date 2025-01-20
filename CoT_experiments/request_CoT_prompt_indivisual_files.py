import re
import selfies as sf
import json
import os
import pandas as pd
import numpy as np
import time

from rdkit import Chem
from rdkit.Chem import MACCSkeys
from tqdm import tqdm
from openai import OpenAI
from glob import glob


os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# import CoT_experiments/keys.txt
with open("CoT_experiments/keys.txt", "r") as f:
    keys = f.readlines()
    keys = [k.strip() for k in keys]
    for key in keys:
        env_name, env_value = key.split("=")
        os.environ[env_name] = env_value

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])



def get_price(response):
    model = response['model']
    if model == "gpt-3.5-turbo":
        prompt_price_per_token = 0.000003
        completion_price_per_token = 0.000006
    elif model == "gpt-4o" or model == "gpt-4o-2024-11-20" or model == "gpt-4o-2024-08-06":
        prompt_price_per_token = 0.0000025
        completion_price_per_token = 0.00001
    elif model == "gpt-4o-mini" or model == "gpt-4o-mini-2024-07-18":
        prompt_price_per_token = 0.00000015
        completion_price_per_token = 0.0000006
    elif model == "o1":
        prompt_price_per_token = 0.000015
        completion_price_per_token = 0.00006
    else:
        prompt_price_per_token = 0.
        completion_price_per_token = 0.
    prompt_tokens = response['usage']['prompt_tokens']
    completion_tokens = response['usage']['completion_tokens']
    price = prompt_tokens * prompt_price_per_token + completion_tokens * completion_price_per_token
    return price




jsonl_file = "CoT_experiments/data/openai_batch/requests/retro_valid_batch_0.jsonl"


reasoning_text_all = []
# import jsonl file
with open(jsonl_file, "r") as f:
    request_data = f.readlines()
    request_data = [json.loads(d) for d in request_data]

for d in tqdm(request_data, desc="Processing", total=len(request_data)):
    response = client.chat.completions.create(**d['body'])
    reasoning_text_all.append(response.choices[0].message.content)


file_name = f"data/presto_data/retro/valid.json"
with open(file_name, 'r') as f:
    data = json.load(f)


assert len(data) == len(reasoning_text_all), f"{len(data)} != {len(reasoning_text_all)}"
for d, reasoning_text in zip(data, reasoning_text_all):
    d['reasoning'] = reasoning_text


# Save to f"CoT_experiments/data/presto_reasoning_data/retro/valid.json"
with open(f"CoT_experiments/data/presto_reasoning_data/retro/valid.json", "w") as f:
    json.dump(data, f, indent=4)
