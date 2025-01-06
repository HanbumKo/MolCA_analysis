import re
import selfies as sf
import json
import os
import time
import pandas as pd
import numpy as np

from openai import OpenAI
from tqdm import tqdm
from glob import glob
from collections import defaultdict
from utils.help_funcs import calculate_smiles_metrics


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


n_shot_examples = {}
# Load CoT_experiments/data/openai_batch/responses/forward_batch_0.jsonl
with open("CoT_experiments/data/openai_batch/responses/forward_batch_0.jsonl", "r") as f:
    batch_0 = [json.loads(line) for line in f.readlines()][:10]
    data = [b['response']['body']['choices'][0]['message']['content'] for b in batch_0]
    n_shot_examples["forward"] = data
# Load CoT_experiments/data/openai_batch/responses/retro_batch_0.jsonl
with open("CoT_experiments/data/openai_batch/responses/retro_batch_0.jsonl", "r") as f:
    batch_0 = [json.loads(line) for line in f.readlines()][:10]
    data = [b['response']['body']['choices'][0]['message']['content'] for b in batch_0]
    n_shot_examples["retro"] = data
# Load CoT_experiments/data/openai_batch/responses/reagent_batch_0.jsonl
with open("CoT_experiments/data/openai_batch/responses/reagent_batch_0.jsonl", "r") as f:
    batch_0 = [json.loads(line) for line in f.readlines()][:10]
    data = [b['response']['body']['choices'][0]['message']['content'] for b in batch_0]
    n_shot_examples["reagent"] = data


questions = {"forward": [], "retro": [], "reagent": []}
answers = {"forward": [], "retro": [], "reagent": []}
file_name = glob(f"data/biot5_plus_data/tasks_plus/*_forward_reaction_prediction_molinst_mol_train.json")[0]
with open(file_name, 'r') as f:
    data = json.load(f)['Instances'][:10]
for d in data:
    instruction = d['instruction']
    smiles = d['input'].split("[START_I_SMILES]")[1].split("[END_I_SMILES]")[0]
    question = f"{instruction}\n[START_I_SMILES]{smiles}[END_I_SMILES]"
    questions["forward"].append(question)
    ground_truth = d['output'][0].split("[START_I_SMILES]")[1].split("[END_I_SMILES]")[0]
    answers["forward"].append(ground_truth)
file_name = glob(f"data/biot5_plus_data/tasks_plus/*_retrosynthesis_molinst_mol_train.json")[0]
with open(file_name, 'r') as f:
    data = json.load(f)['Instances'][:10]
for d in data:
    instruction = d['instruction']
    smiles = d['input'].split("[START_I_SMILES]")[1].split("[END_I_SMILES]")[0]
    question = f"{instruction}\n[START_I_SMILES]{smiles}[END_I_SMILES]"
    questions["retro"].append(question)
    ground_truth = d['output'][0].split("[START_I_SMILES]")[1].split("[END_I_SMILES]")[0]
    answers["retro"].append(ground_truth)
file_name = glob(f"data/biot5_plus_data/tasks_plus/*_reagent_prediction_molinst_mol_train.json")[0]
with open(file_name, 'r') as f:
    data = json.load(f)['Instances'][:10]
for d in data:
    instruction = d['instruction']
    smiles = d['input'].split("[START_I_SMILES]")[1].split("[END_I_SMILES]")[0]
    left_smiles, right_smiles = smiles.split(">>")
    question = f"{instruction}\n[START_I_SMILES]{left_smiles}[END_I_SMILES]>>[START_I_SMILES]{right_smiles}[END_I_SMILES]"
    questions["reagent"].append(question)
    ground_truth = d['output'][0].split("[START_I_SMILES]")[1].split("[END_I_SMILES]")[0]
    answers["reagent"].append(ground_truth)



def parse_first_number(string):
    """
    문자열에서 첫 번째 숫자를 파싱하는 함수.
    
    :param string: 입력 문자열
    :return: 첫 번째 숫자 (정수 또는 실수) 또는 None
    """
    match = re.search(r'-?\d+(\.\d+)?', string)
    if match:
        return float(match.group()) if '.' in match.group() else int(match.group())
    return "NONE"


def regression_evaluate_gpt(predictions, targets):
    validity = []
    predictions_processed = []
    targets_processed = []
    for p, t in zip(predictions, targets):
        # Convert to float
        try:
            number_parsed = str(parse_first_number(p))
            p = float(number_parsed)
            t = float(t)
        except:
            validity.append(False)
            continue
        predictions_processed.append(p)
        targets_processed.append(t)
        validity.append(True)
    validity = np.array(validity)
    validity = np.mean(validity)
    if validity == 0:
        return None, None, None, None
    predictions = np.array(predictions_processed)
    targets = np.array(targets_processed)
    mae = np.mean(np.abs(predictions - targets))
    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)
    # mape = np.mean(np.abs(predictions - targets) / targets)
    return mae, mse, rmse, validity


def get_openai_request_body(user_query, task_name, model, n_shot):
    if task_name == "forward":
        system_message = """You are a chemistry reaction expert, and your task is to predict the possible products from a precursor consisting of reactants and reagents in a forward reaction prediction task. When given a precursor, you need to predict the possible product and provide an answer to the user. During the prediction, you must generate the chemical reaction process step-by-step, following these specific guidelines:
    1. Separate the precursor into reactants and reagents.
    2. Identify and describe the substructures and chemical properties of the reactants that will undergo reaction to form the product.
    3. Explain the mechanism suggested by the action of the reagent or the reaction conditions.
    4. If any major intermediates are anticipated, briefly describe their structure or functional mechanism.
    5. Derive the product in SMILES format.

After following the steps above, present the final predicted product wrapped in <ANSWER> and </ANSWER> tags."""
    elif task_name == "retro":
        system_message = """You are an expert in chemical reactions, and your task is to predict possible reactants in retrosynthesis tasks, where the product is given, and the goal is to identify the reactants that can generate the product. When making predictions, you should generate the process of inferring the reactants step-by-step, following these specific steps:
    1. Analyze the key functional groups of the product.
    2. Identify the bonds that need to be formed or transformed.
    3. Trace back to the most logical reaction type (e.g., nucleophilic substitution, acid-base reaction, oxidation-reduction, aromatic substitution, etc.).
    4. Propose the necessary reactants.
    5. Briefly justify the proposed pathway.

After explaining the process, you should present the final predicted reactants enclosed in <ANSWER> and </ANSWER>."""
    elif task_name == "reagent":
        system_message = """You are a chemical reaction expert tasked with predicting possible reagents in a reagent prediction task. When given a product, you must deduce and provide the possible reagents that could produce the product from the provided reactants. Your prediction process should follow a step-by-step reasoning approach as outlined below:
    1. Compare Reactants and Product: Identify which functional groups or bonds have changed between the reactants and product.
    2. Infer General Mechanism: Deduce the general mechanism that can account for the identified changes.
    3. Describe Reagents and Conditions: Outline the necessary reagents and reaction conditions (e.g., acid/base, oxidizing/reducing agents, catalysts) for the reaction type, step-by-step.
    4. Final Verification: Confirm that the proposed reagents can realistically produce the product via the deduced mechanism.
    5. Provide Reagents in SMILES Format: Present the suggested reagents in SMILES (Simplified Molecular Input Line Entry System) format.

After completing the above steps, enclose the final predicted reactants within <ANSWER> and </ANSWER> tags."""
    else:
        raise ValueError(f"Invalid task: {task_name}")

    if n_shot == 1:
        system_message += "\n\nHere's an example"
    elif n_shot > 1:
        system_message += f"\n\nHere are {n_shot} examples"

    for i in range(n_shot):
        system_message += f"\n\n### User's Message\n{questions[task_name][i]}\n\n### Your Response\n{n_shot_examples[task_name][i]}\n<ANSWER>{answers[task_name][i]}</ANSWER>"

    body_dict = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": system_message,
            },
            {
                "role": "user",
                "content": user_query
            }
        ],
        "temperature": 0.0,
        "max_tokens": 1000
    }

    return body_dict


def smiles_to_selfies(smiles):
    try:
        selfies_string = sf.encoder(smiles)
        return selfies_string
    except Exception as e:
        print(f"Error during conversion: {e}")
        return "NONE"


files = [
    ("data/biot5_plus_data/tasks_plus/task216_forward_reaction_prediction_molinst_mol_test.json", "forward"),
    ("data/biot5_plus_data/tasks_plus/task219_retrosynthesis_molinst_mol_test.json", "retro"),
    ("data/biot5_plus_data/tasks_plus/task213_reagent_prediction_molinst_mol_test.json", "reagent"),
]

for model in ["gpt-4o-mini", "gpt-3.5-turbo", "gpt-4o-2024-11-20"]:
    # if CoT_experiments/data/openai_batch/requests/{model}_test.jsonl does not exist, create it
    if not os.path.exists(f"CoT_experiments/data/openai_batch/requests/{model}_test.jsonl"):
        request_list = []
        for n_shot in [0, 1, 2, 3]:
            for file_name, task_name in files:
                with open(file_name, 'r') as f:
                    data = json.load(f)
                for i, d in enumerate(data["Instances"]):
                    instruction = d['instruction']
                    smiles = d['input'].split("[START_I_SMILES]")[1].split("[END_I_SMILES]")[0]
                    user_message = f"{instruction}\n{smiles}"
                    body_dict = get_openai_request_body(user_message, task_name, model, n_shot)
                    request_dict = {
                        "custom_id": f"{model}_{n_shot}_{task_name}_test_{i}",
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": body_dict
                    }
                    request_list.append(request_dict)
        with open(f"CoT_experiments/data/openai_batch/requests/{model}_test.jsonl", "w") as f:
            for request_dict in request_list:
                f.write(json.dumps(request_dict) + "\n")



################################################################################################
# Reaction prediction tasks
jsonl_files = [
    # "CoT_experiments/data/openai_batch/requests/gpt-3.5-turbo_test.jsonl",
    "CoT_experiments/data/openai_batch/requests/gpt-4o-mini_test.jsonl",
    # "CoT_experiments/data/openai_batch/requests/gpt-4o-2024-11-20_test.jsonl",
]


for jsonl_file in jsonl_files:
    # if jsonl_file.replace("requests", "responses") already exists, skip
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



jsonl_files = [
    # "CoT_experiments/data/openai_batch/responses/gpt-3.5-turbo_test.jsonl",
    "CoT_experiments/data/openai_batch/responses/gpt-4o-mini_test.jsonl",
    # "CoT_experiments/data/openai_batch/responses/gpt-4o-2024-11-20_test.jsonl",
]

reasoning_text_dict = {}
answer_dict = {}
raw_response_dict = {}
for jsonl_file in jsonl_files:
    with open(jsonl_file, "r") as f:
        data = [json.loads(line) for line in f.readlines()]
    for d in data:
        custom_id = d['custom_id']
        raw_response = d['response']['body']['choices'][0]['message']['content']
        if "<ANSWER>" in raw_response and "</ANSWER>" in raw_response:
            reasoning_text = d['response']['body']['choices'][0]['message']['content'].split("<ANSWER>")[0].strip()
            answer = d['response']['body']['choices'][0]['message']['content'].split("<ANSWER>")[1].split("</ANSWER>")[0]
        else:
            reasoning_text = raw_response
            answer = ""
        reasoning_text_dict[custom_id] = reasoning_text
        answer_dict[custom_id] = answer
        raw_response_dict[custom_id] = raw_response



num_shots = [0, 1, 2, 3]
# for model in ["gpt-4o-mini", "gpt-3.5-turbo", "gpt-4o-2024-11-20"]:
for model in ["gpt-4o-mini"]:
    for n_shot in num_shots:
        for file_name, task_name in files:
            ground_truth_list = []
            raw_output_list = []
            prediction_list = []
            reasoning_text_list = []
            with open(file_name, 'r') as f:
                dataset = json.load(f)["Instances"]
            for i, d in enumerate(dataset):
                ground_truth = d['output'][0].split("[START_I_SMILES]")[1].split("[END_I_SMILES]")[0]
                raw_response = raw_response_dict[f"{model}_{n_shot}_{task_name}_test_{i}"]
                prediction = answer_dict[f"{model}_{n_shot}_{task_name}_test_{i}"]
                reasoning_text = reasoning_text_dict[f"{model}_{n_shot}_{task_name}_test_{i}"]                
                ground_truth_list.append(ground_truth)
                raw_output_list.append(raw_response)
                prediction_list.append(prediction)
                reasoning_text_list.append(reasoning_text)
            if task_name == "forward":
                eval_results = calculate_smiles_metrics(prediction_list, ground_truth_list, metrics=('exact_match', 'fingerprint'))
            elif task_name == "retro" or task_name == "reagent":
                eval_results = calculate_smiles_metrics(prediction_list, ground_truth_list, metrics=('exact_match', 'fingerprint', 'multiple_match'))
            print("="*100)
            print(f"Task: {task_name}, Model: {model}, N shot: {n_shot}")
            for k, v in eval_results.items():
                print(f"{k}: {v}")
            print("="*100)
            print()
            # save eval_results to file
            with open(f"CoT_experiments/results/cot_prompt_test/eval_results/{task_name}_{model}_{n_shot}.txt", "w") as f:
                f.write(str(eval_results))
            gt_preds = [
                {"ground_truth": gt, "prediction": pred, "reasoning_text": reason_t, "raw_output": raw} for gt, pred, reason_t, raw in zip(ground_truth_list, prediction_list, reasoning_text_list, raw_output_list)
            ]
            with open(f"CoT_experiments/results/cot_prompt_test/gt_preds/{task_name}_{model}_{n_shot}.json", "w") as f:
                json.dump(gt_preds, f, indent=4)
            print(f"Results saved for task: {task_name}, model: {model}, N shot: {n_shot}")