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




jsonl_files = [
    # "CoT_experiments/data/openai_batch/requests/forward_test_batch_0.jsonl",
    # "CoT_experiments/data/openai_batch/requests/forward_valid_batch_0.jsonl",
    # "CoT_experiments/data/openai_batch/requests/forward_train_batch_0.jsonl",
    # "CoT_experiments/data/openai_batch/requests/forward_train_batch_1.jsonl",
    # "CoT_experiments/data/openai_batch/requests/forward_train_batch_2.jsonl",
    # "CoT_experiments/data/openai_batch/requests/forward_train_batch_3.jsonl",
    # "CoT_experiments/data/openai_batch/requests/forward_train_batch_4.jsonl",
    # "CoT_experiments/data/openai_batch/requests/forward_train_batch_5.jsonl",
    # "CoT_experiments/data/openai_batch/requests/forward_train_batch_6.jsonl",
    # "CoT_experiments/data/openai_batch/requests/forward_train_batch_7.jsonl",
    # "CoT_experiments/data/openai_batch/requests/forward_train_batch_8.jsonl",
    # "CoT_experiments/data/openai_batch/requests/retro_test_batch_0.jsonl",
    # "CoT_experiments/data/openai_batch/requests/retro_valid_batch_0.jsonl",
    # "CoT_experiments/data/openai_batch/requests/retro_train_batch_0.jsonl",
    # "CoT_experiments/data/openai_batch/requests/retro_train_batch_1.jsonl",
    # "CoT_experiments/data/openai_batch/requests/retro_train_batch_2.jsonl",
    # "CoT_experiments/data/openai_batch/requests/retro_train_batch_3.jsonl",
    # "CoT_experiments/data/openai_batch/requests/retro_train_batch_4.jsonl",
    # "CoT_experiments/data/openai_batch/requests/retro_train_batch_5.jsonl",
    # "CoT_experiments/data/openai_batch/requests/retro_train_batch_6.jsonl",
    # "CoT_experiments/data/openai_batch/requests/retro_train_batch_7.jsonl",
    # "CoT_experiments/data/openai_batch/requests/retro_train_batch_8.jsonl",
    # "CoT_experiments/data/openai_batch/requests/reagent_test_batch_0.jsonl",
    # "CoT_experiments/data/openai_batch/requests/reagent_valid_batch_0.jsonl",
    # "CoT_experiments/data/openai_batch/requests/reagent_train_batch_0.jsonl",
    # "CoT_experiments/data/openai_batch/requests/reagent_train_batch_1.jsonl",
    # "CoT_experiments/data/openai_batch/requests/reagent_train_batch_2.jsonl",
    # "CoT_experiments/data/openai_batch/requests/reagent_train_batch_3.jsonl",
    # "CoT_experiments/data/openai_batch/requests/catalyst_test_batch_0.jsonl",
    # "CoT_experiments/data/openai_batch/requests/catalyst_valid_batch_0.jsonl",
    # "CoT_experiments/data/openai_batch/requests/catalyst_train_batch_0.jsonl",
    # "CoT_experiments/data/openai_batch/requests/solvent_test_batch_0.jsonl",
    # "CoT_experiments/data/openai_batch/requests/solvent_valid_batch_0.jsonl",
    # "CoT_experiments/data/openai_batch/requests/solvent_train_batch_0.jsonl",
    # "CoT_experiments/data/openai_batch/requests/solvent_train_batch_1.jsonl",
    # "CoT_experiments/data/openai_batch/requests/solvent_train_batch_2.jsonl",
    # "CoT_experiments/data/openai_batch/requests/solvent_train_batch_3.jsonl",
    # "CoT_experiments/data/openai_batch/requests/solvent_train_batch_4.jsonl",
]


for jsonl_file in jsonl_files:
    if os.path.exists(jsonl_file.replace("requests", "responses")):
        continue
    # 2. Uploading Your Batch Input File
    batch_input_file = client.files.create(
        file=open(jsonl_file, "rb"),
        purpose="batch"
    )
    # 3. Creating the Batch
    batch_input_file_id = batch_input_file.id
    request_data = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )

    # Save to CoT_experiments/data/openai_batch/request_info/
    with open(jsonl_file.replace("requests", "request_info").replace(".jsonl", ".json"), "w") as f:
        json.dump(request_data.to_dict(), f, indent=4)

    while True:
        time.sleep(10)
        
        batch = client.batches.retrieve(request_data.id)
        print(f"Processing {jsonl_file}")
        print(f"Status: {batch.status}")
        print(f"Total: {batch.request_counts.total}, completed: {batch.request_counts.completed}, failed: {batch.request_counts.failed}")
        print()
        if batch.status == "completed":
            break

    # 5. Retrieving the Results
    file_response = client.files.content(batch.output_file_id)
    
    # 6. Save the results
    with open(jsonl_file.replace("requests", "responses"), "w") as f:
        f.write(file_response.text)


total_price = 0.
# Save to data
for task_name in ["forward", "retro", "reagent", "catalyst", "solvent"]:
    for split in ["train", "valid", "test"]:
# for task_name in ["retro"]:
#     for split in ["test"]:
        jsonl_files = sorted(glob(f"CoT_experiments/data/openai_batch/responses/{task_name}_{split}_batch_*.jsonl"))
        # if task_name == "forward" and split == "valid":
        #     jsonl_files = sorted(glob(f"CoT_experiments/data/openai_batch/responses/{task_name}_train_batch_*.jsonl"))
        # if task_name == "retro" and split == "valid":
        #     jsonl_files = sorted(glob(f"CoT_experiments/data/openai_batch/responses/{task_name}_train_batch_*.jsonl"))
        reasoning_texts = []
        for jsonl_file in jsonl_files:
            with open(jsonl_file, "r") as f:
                # Load the jsonl file
                data = f.readlines()
                data = [json.loads(d) for d in data]
            for i, d in enumerate(data):
                total_price += get_price(d['response']['body'])
                reasoning_texts.append(data[i]['response']['body']['choices'][0]['message']['content'])

        file_name = f"data/presto_data/{task_name}/{split}.json"
        with open(file_name, 'r') as f:
            data = json.load(f)
        
        # if task_name == "forward" and split == "train":
        #     reasoning_texts = reasoning_texts[:-100]
        # elif task_name == "retro" and split == "train":
        #     reasoning_texts = reasoning_texts[:-100]
        # elif task_name == "forward" and split == "valid":
        #     reasoning_texts = reasoning_texts[-100:]
        # elif task_name == "retro" and split == "valid":
        #     reasoning_texts = reasoning_texts[-100:]

        assert len(data) == len(reasoning_texts), f"{len(data)} != {len(reasoning_texts)} in {task_name} {split}"
        for d, reasoning_text in zip(data, reasoning_texts):
            d['reasoning'] = reasoning_text

        data_name = file_name.split("/")[-1]
        save_path = f"CoT_experiments/data/presto_reasoning_data/{task_name}/{data_name}"
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=4)

print(f"Total price: {round(total_price, 2)}$")



