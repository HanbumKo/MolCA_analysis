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



# jsonl_files = glob("CoT_experiments/data/openai_batch/requests/example_*.jsonl")[:3]
jsonl_files = [
    # "CoT_experiments/data/openai_batch/requests/forward_test_batch_0.jsonl",
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
    # "CoT_experiments/data/openai_batch/requests/retro_train_batch_0.jsonl",
    # "CoT_experiments/data/openai_batch/requests/retro_train_batch_1.jsonl",
    # "CoT_experiments/data/openai_batch/requests/retro_train_batch_2.jsonl",
    # "CoT_experiments/data/openai_batch/requests/retro_train_batch_3.jsonl",
    # "CoT_experiments/data/openai_batch/requests/retro_train_batch_4.jsonl",
    # "CoT_experiments/data/openai_batch/requests/retro_train_batch_5.jsonl",
    # "CoT_experiments/data/openai_batch/requests/retro_train_batch_6.jsonl",
    # "CoT_experiments/data/openai_batch/requests/retro_train_batch_7.jsonl",
    # "CoT_experiments/data/openai_batch/requests/retro_train_batch_8.jsonl",
    "CoT_experiments/data/openai_batch/requests/reagent_test_batch_0.jsonl",
    "CoT_experiments/data/openai_batch/requests/reagent_valid_batch_0.jsonl",
    "CoT_experiments/data/openai_batch/requests/reagent_train_batch_0.jsonl",
    "CoT_experiments/data/openai_batch/requests/reagent_train_batch_1.jsonl",
    "CoT_experiments/data/openai_batch/requests/reagent_train_batch_2.jsonl",
    "CoT_experiments/data/openai_batch/requests/reagent_train_batch_3.jsonl",
    "CoT_experiments/data/openai_batch/requests/catalyst_test_batch_0.jsonl",
    "CoT_experiments/data/openai_batch/requests/catalyst_valid_batch_0.jsonl",
    "CoT_experiments/data/openai_batch/requests/catalyst_train_batch_0.jsonl",
    "CoT_experiments/data/openai_batch/requests/solvent_test_batch_0.jsonl",
    "CoT_experiments/data/openai_batch/requests/solvent_valid_batch_0.jsonl",
    "CoT_experiments/data/openai_batch/requests/solvent_train_batch_0.jsonl",
    "CoT_experiments/data/openai_batch/requests/solvent_train_batch_1.jsonl",
    "CoT_experiments/data/openai_batch/requests/solvent_train_batch_2.jsonl",
    "CoT_experiments/data/openai_batch/requests/solvent_train_batch_3.jsonl",
    "CoT_experiments/data/openai_batch/requests/solvent_train_batch_4.jsonl",
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



# Save to data
for task_name in ["forward", "retro", "reagent"]:
    jsonl_files = glob(f"CoT_experiments/data/openai_batch/responses/{task_name}_batch_*.jsonl")
    reasoning_texts = []
    for jsonl_file in jsonl_files:
        with open(jsonl_file, "r") as f:
            # Load the jsonl file
            data = f.readlines()
            data = [json.loads(d) for d in data]
        for d in data:
            reasoning_texts.append(data[0]['response']['body']['choices'][0]['message']['content'])

    if task_name == "forward":
        file_name = "data/biot5_plus_data/tasks_plus/task214_forward_reaction_prediction_molinst_mol_train.json"
    elif task_name == "retro":
        file_name = "data/biot5_plus_data/tasks_plus/task217_retrosynthesis_molinst_mol_train.json"
    elif task_name == "reagent":
        file_name = "data/biot5_plus_data/tasks_plus/task211_reagent_prediction_molinst_mol_train.json"
    with open(file_name, 'r') as f:
        data = json.load(f)
    for d, reasoning_text in zip(data['Instances'], reasoning_texts):
        d['reasoning'] = reasoning_text

    data_name = file_name.split("/")[-1]
    save_path = f"CoT_experiments/data/biot5_plus_reasoning_data/{data_name}"
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=4)





