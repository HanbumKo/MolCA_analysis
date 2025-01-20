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



# load CoT_experiments/data/fewshot_example_all.json
with open("CoT_experiments/data/fewshot_example_all.json", "r") as f:
    fewshot_example_all = json.load(f)
retro_reasoning_texts = [[], [], [], []]

jsonl_file = "CoT_experiments/data/openai_batch/requests/tmp.jsonl"


reasoning_text_all = []
# import jsonl file
with open(jsonl_file, "r") as f:
    data = f.readlines()
    data = [json.loads(d) for d in data]

for d in tqdm(data, desc="Processing", total=len(data)):
    response = client.chat.completions.create(**d['body'])
    reasoning_text_all.append(response.choices[0].message.content)


retro_reasoning_texts[0] = reasoning_text_all[:10]
retro_reasoning_texts[1] = reasoning_text_all[10:20]
retro_reasoning_texts[2] = reasoning_text_all[20:30]
retro_reasoning_texts[3] = reasoning_text_all[30:40]

fewshot_example_all['reasoning_texts']['retro'][0] = retro_reasoning_texts[0]
fewshot_example_all['reasoning_texts']['retro'][1] = retro_reasoning_texts[1]
fewshot_example_all['reasoning_texts']['retro'][2] = retro_reasoning_texts[2]
fewshot_example_all['reasoning_texts']['retro'][3] = retro_reasoning_texts[3]

# Save to CoT_experiments/data/fewshot_example_all.json
with open("CoT_experiments/data/fewshot_example_all_tmp.json", "w") as f:
    json.dump(fewshot_example_all, f, indent=4)
