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


def get_openai_output(request_body):
    response = client.chat.completions.create(
        **request_body
    )
    output = response.choices[0].message.content.strip()
    reason = response.choices[0].finish_reason
    return output, reason, response


jsonl_files = [
    # "CoT_experiments/data/openai_batch/responses/forward_test_batch_0.jsonl",
    # "CoT_experiments/data/openai_batch/responses/forward_train_batch_0.jsonl",
    # "CoT_experiments/data/openai_batch/responses/forward_train_batch_1.jsonl",
    # "CoT_experiments/data/openai_batch/responses/forward_train_batch_2.jsonl",
    # "CoT_experiments/data/openai_batch/responses/forward_train_batch_3.jsonl",
    # "CoT_experiments/data/openai_batch/responses/forward_train_batch_4.jsonl",
    # "CoT_experiments/data/openai_batch/responses/forward_train_batch_5.jsonl",
    # "CoT_experiments/data/openai_batch/responses/forward_train_batch_6.jsonl",
    # "CoT_experiments/data/openai_batch/responses/forward_train_batch_7.jsonl",
    # "CoT_experiments/data/openai_batch/responses/forward_train_batch_8.jsonl",
    # "CoT_experiments/data/openai_batch/responses/retro_test_batch_0.jsonl",
    # "CoT_experiments/data/openai_batch/responses/retro_train_batch_0.jsonl",
    # "CoT_experiments/data/openai_batch/responses/retro_train_batch_1.jsonl",
    # "CoT_experiments/data/openai_batch/responses/retro_train_batch_2.jsonl",
    # "CoT_experiments/data/openai_batch/responses/retro_train_batch_3.jsonl",
    # "CoT_experiments/data/openai_batch/responses/retro_train_batch_4.jsonl",
    # "CoT_experiments/data/openai_batch/responses/retro_train_batch_5.jsonl",
    # "CoT_experiments/data/openai_batch/responses/retro_train_batch_6.jsonl",
    # "CoT_experiments/data/openai_batch/responses/retro_train_batch_7.jsonl",
    # "CoT_experiments/data/openai_batch/responses/retro_train_batch_8.jsonl",
    # "CoT_experiments/data/openai_batch/responses/reagent_test_batch_0.jsonl",
    # "CoT_experiments/data/openai_batch/responses/reagent_valid_batch_0.jsonl",
    # "CoT_experiments/data/openai_batch/responses/reagent_train_batch_0.jsonl",
    # "CoT_experiments/data/openai_batch/responses/reagent_train_batch_1.jsonl",
    # "CoT_experiments/data/openai_batch/responses/reagent_train_batch_2.jsonl",
    # "CoT_experiments/data/openai_batch/responses/reagent_train_batch_3.jsonl",
    # "CoT_experiments/data/openai_batch/responses/catalyst_test_batch_0.jsonl",
    # "CoT_experiments/data/openai_batch/responses/catalyst_valid_batch_0.jsonl",
    # "CoT_experiments/data/openai_batch/responses/catalyst_train_batch_0.jsonl",
    # "CoT_experiments/data/openai_batch/responses/solvent_test_batch_0.jsonl",
    # "CoT_experiments/data/openai_batch/responses/solvent_valid_batch_0.jsonl",
    # "CoT_experiments/data/openai_batch/responses/solvent_train_batch_0.jsonl",
    # "CoT_experiments/data/openai_batch/responses/solvent_train_batch_1.jsonl",
    # "CoT_experiments/data/openai_batch/responses/solvent_train_batch_2.jsonl",
    # "CoT_experiments/data/openai_batch/responses/solvent_train_batch_3.jsonl",
    # "CoT_experiments/data/openai_batch/responses/solvent_train_batch_4.jsonl",
    "CoT_experiments/data/openai_batch/responses/gpt-4o-2024-11-20_test.jsonl",
]


for jsonl_file in jsonl_files:
    print(f"Processing {jsonl_file}")
    with open(jsonl_file, "r") as f:
        response_data = f.readlines()
        response_data = [json.loads(d) for d in response_data]
    with open(jsonl_file.replace("responses", "requests"), "r") as f:
        request_data = f.readlines()
        request_data = [json.loads(d) for d in request_data]
    for i, d in enumerate(response_data):
        finish_reason = d['response']['body']['choices'][0]['finish_reason']
        if finish_reason == "length":
            request_body = request_data[i]['body']
            # request_body["frequency_penalty"] = 0.1
            request_body['max_tokens'] = 1500
            new_result, new_reason, new_response = get_openai_output(request_body)
            print()
            print("="*100)
            print(new_result)
            print(new_reason)
            print("="*100)
            print()
            if new_reason == "length":
                print(f"length failed")
            d['response']['body'] = new_response.to_dict()
    # Save response_data list as jsonl
    with open(jsonl_file.replace(".jsonl", "_tmp.jsonl"), "w") as f:
        for d in response_data:
            f.write(json.dumps(d) + "\n")
        

"""
# Save to data
for task_name in ["forward", "retro", "reagent", "catalyst", "solvent"]:
    for split in ["train", "valid", "test"]:
        jsonl_files = glob(f"CoT_experiments/data/openai_batch/responses/{task_name}_{split}_batch_*.jsonl")
        if task_name == "forward" and split == "valid":
            jsonl_files = glob(f"CoT_experiments/data/openai_batch/responses/{task_name}_train_batch_*.jsonl")
        if task_name == "retro" and split == "valid":
            jsonl_files = glob(f"CoT_experiments/data/openai_batch/responses/{task_name}_train_batch_*.jsonl")
        reasoning_texts = []
        for jsonl_file in jsonl_files:
            with open(jsonl_file, "r") as f:
                # Load the jsonl file
                data = f.readlines()
                data = [json.loads(d) for d in data]
            for i, d in enumerate(data):
                reasoning_texts.append(data[i]['response']['body']['choices'][0]['message']['content'])

        file_name = f"data/presto_data/{task_name}/{split}.json"
        with open(file_name, 'r') as f:
            data = json.load(f)
        
        if task_name == "forward" and split == "train":
            reasoning_texts = reasoning_texts[:-100]
        elif task_name == "retro" and split == "train":
            reasoning_texts = reasoning_texts[:-100]
        elif task_name == "forward" and split == "valid":
            reasoning_texts = reasoning_texts[-100:]
        elif task_name == "retro" and split == "valid":
            reasoning_texts = reasoning_texts[-100:]

        for d, reasoning_text in zip(data, reasoning_texts):
            d['reasoning'] = reasoning_text

        data_name = file_name.split("/")[-1]
        save_path = f"CoT_experiments/data/presto_reasoning_data/{task_name}/{data_name}"
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=4)





"""