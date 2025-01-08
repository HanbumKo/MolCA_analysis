import re
import selfies as sf
import json
import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from openai import OpenAI
from tqdm import tqdm
from glob import glob
from collections import defaultdict
from utils.help_funcs import calculate_smiles_metrics
from utils.evaluator import MoleculeSMILESEvaluator


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


def parse_output(raw_response):
    reasoning_text = extract_reasoning_content(raw_response)
    answer_text = extract_answer_content(raw_response)
    if len(reasoning_text) == 0:
        reasoning_text = raw_response.split("<ANSWER>")[0]
    return reasoning_text, answer_text
    

def extract_reasoning_content(text):
    try:
        return text.split("<REASONING>")[1].split("</REASONING>")[0]
    except:
        # No "<REASONING>" or "</REASONING>" found
        return ""


def extract_answer_content(text):
    try:
        return text.split("<ANSWER>")[1].split("</ANSWER>")[0]
    except:
        # No "<ANSWER>" or "</ANSWER>" found
        return ""


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


def get_openai_request_body(d, n_shot, task_name, use_reasoning, seed, model):
    system_prompt = d['system_prompt']
    user_prompt = d['user_prompt'].replace(" .", ".").replace("[START_I_SMILES]", "").replace("[END_I_SMILES]", "").strip()
    if use_reasoning:
        system_prompt += " Your prediction process should follow a step-by-step reasoning approach as outlined below:"
        if task_name == "forward":
            system_prompt += f"""
1. Separate the precursor into reactants and reagents.
2. Identify and describe the substructures and chemical properties of the reactants that will undergo reaction to form the product.
3. Explain the mechanism suggested by the action of the reagent or the reaction conditions.
4. If any major intermediates are anticipated, briefly describe their structure or functional mechanism.
5. Derive the product in SMILES format.

"""
        elif task_name == "retro":
            system_prompt += f"""
1. Analyze the key functional groups of the product.
2. Identify the bonds that need to be formed or transformed.
3. Trace back to the most logical reaction type (e.g., nucleophilic substitution, acid-base reaction, oxidation-reduction, aromatic substitution, etc.).
4. Propose the necessary reactants.
5. Briefly justify the proposed pathway.

"""
        elif task_name == f"reagent":
            system_prompt += """
1. Compare the reactants and products to identify any functional groups or bonds that have changed.
2. Infer the general mechanism that could facilitate such transformations.
3. Describe the reagents and reaction conditions (e.g., acid/base, oxidizing/reducing agents, catalysts) required for the identified reaction type in a step-by-step manner.
4. Review whether the proposed reagents can feasibly enable the formation of the products through the suggested mechanism.
5. Represent the reagents in SMILES format.

"""
        elif task_name == f"catalyst":
            system_prompt += """
1. Identify the reaction type and key conditions (temperature, pH, acid/base, etc.).
2. List potential catalyst candidates commonly used for the reaction (e.g., acid/base catalysts, metal complexes).
3. Briefly explain the role of the catalyst in the reaction mechanism.
4. Select a catalyst compatible with the reactants and propose reaction conditions.
5. Summarize the reasons for selecting the catalyst and provide the output in SMILES format.

"""
        elif task_name == f"solvent":
            system_prompt += """
1. Identify reaction mechanisms (acid/base, oxidation/reduction, etc.) and reaction sensitivities (heat, moisture, etc.).
2. Classify candidates based on solvent properties (polarity, boiling point, viscosity, etc.).
3. Consider the stability of reactants and products (water solubility, acid/base resistance, etc.).
4. Evaluate practical factors such as toxicity, cost, and flammability.
5. Summarize the rationale for selecting the optimal solvent (and co-solvent) and derive it in SMILES format.

"""
        else:
            raise ValueError(f"Invalid task: {task_name}")
        system_prompt += "The reasoning content should be enclosed with <REASONING> and </REASONING> tags, while the prediction, represented in SMILES notation, should be enclosed with <ANSWER> and </ANSWER> tags."

    else:
        system_prompt += " The prediction, represented in SMILES notation, should be enclosed with <ANSWER> and </ANSWER> tags."

    if n_shot == 1:
        system_prompt += "\n\nHere's an example"
    elif n_shot > 1:
        system_prompt += f"\n\nHere are {n_shot} examples"

    for i in range(n_shot):
        if use_reasoning:
            system_prompt += f"\n\n### User's Message\n{user_prompts[task_name][seed][i]}\n\n### Your Response\n<REASONING>{reasoning_texts[task_name][seed][i]}</REASONING>\n<ANSWER>{groudn_truths[task_name][seed][i]}</ANSWER>"
        else:
            system_prompt += f"\n\n### User's Message\n{user_prompts[task_name][seed][i]}\n\n### Your Response\n<ANSWER>{groudn_truths[task_name][seed][i]}</ANSWER>"
    body_dict = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt
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



n_shot_indices = {
    "forward": [
        [ 68007, 117631, 110806,  12507,  90314,  85034,  21948,  63507,  58936,  42481], # Seed 0
        [110380,  57767,   5210,  52537,  78208,  41492,  84434,  27686,  85363,  72254], # Seed 1
        [123018, 113459,  68692,  40973,  10737,   9162, 120447,  38970,  31720,  10615], # Seed 2
        [  3975,  67723,  98728,  39088,  33407,  91390,   3255,  86443,  40078,  53187], # Seed 3
    ],
    "retro": [
        [ 95111,  76087,  73572,  38415,  90519,  76941,  13845,  76137, 122776,  39798], # Seed 0
        [107191,  73947,  49590,  11030, 124896,  27047,  94889,  22705, 113058,  73661], # Seed 1
        [  4627,  67235, 108075,  85037, 122617, 126780,  42660,   4510,  16475,   5489], # Seed 2
        [ 18802, 125942,  43866,  30621,  36661,  28829,   1791,  75190,  80399, 109299], # Seed 3
    ],
    "reagent": [
        [ 29865,  38926,  24317,  31727,  28592,  30285,   2766,    388,  32816,  44858], # Seed 0
        [ 56968,  22898,  47068,   8615,  52197,  45646,  34943,  12647,  37485,  37402], # Seed 1
        [ 55361,  13570,  11903,  43322,  53862,  52297,  10586,  17279,  40066,   6928], # Seed 2
        [ 11200,  44758,  33218,  19854,  29028,  30069,  53427,   6425,  42490,  48422], # Seed 3
    ],
    "catalyst": [
        [  4047,   6527,   9810,   2697,   6344,    662,   3841,   1744,   6624,   9577], # Seed 0
        [  4111,  10100,  10002,   9939,   2252,   9122,   8894,   3759,   6852,   5361], # Seed 1
        [  8441,   2040,   4556,   9647,   7162,   9920,    677,   6107,   9879,   1543], # Seed 2
        [  3618,   4217,   9311,   1637,   4803,   3075,   7599,   5323,   6984,   7422], # Seed 3
    ],
    "solvent": [
        [  4721,  10575,  11841,  31973,  13721,  25730,  47724,  68704,  29500,  13373], # Seed 0
        [ 42088,    458,  14606,  29001,  29840,   6709,  53310,  10261,  23381,  57110], # Seed 1
        [ 15740,  28080,  11761,   9362,  35394,  34529,  30720,  68803,  28544,  29689], # Seed 2
        [ 34902,  66293,  53970,  67347,  35434,  59133,  13931,  60174,  31764,  22231], # Seed 3
    ],
}

task_names = ["forward", "retro", "reagent", "catalyst", "solvent"]
n_test_samples = 100
answer_max_length = 200

# 1. Load n-shot examples
reasoning_texts = {task_name: [[], [], [], []] for task_name in task_names}
system_prompts = {task_name: [[], [], [], []] for task_name in task_names}
user_prompts = {task_name: [[], [], [], []] for task_name in task_names}
groudn_truths = {task_name: [[], [], [], []] for task_name in task_names}

for task_name in task_names:
    file_name = f"CoT_experiments/data/presto_reasoning_data/{task_name}/train.json"
    with open(file_name, 'r') as f:
        data = json.load(f)
    for seed, idx in enumerate(n_shot_indices[task_name]):
        for i in idx:
            reasoning_texts[task_name][seed].append(data[i]['reasoning'])
            system_prompts[task_name][seed].append(data[i]['system_prompt'])
            user_prompts[task_name][seed].append(data[i]['user_prompt'].replace(" .", ".").replace("[START_I_SMILES]", "").replace("[END_I_SMILES]", "").strip())
            if task_name == "forward":
                groudn_truths[task_name][seed].append(data[i]['product'])
            elif task_name == "retro":
                groudn_truths[task_name][seed].append(data[i]['reactants'])
            elif task_name == "reagent":
                groudn_truths[task_name][seed].append(data[i]['reagents'])
            elif task_name == "catalyst":
                groudn_truths[task_name][seed].append(data[i]['catalyst'])
            elif task_name == "solvent":
                groudn_truths[task_name][seed].append(data[i]['solvent'])
############################################################################################################



# 2. Create OpenAI batch API requests
for model in ["gpt-4o-mini", "gpt-3.5-turbo", "gpt-4o-2024-11-20"]:
    if os.path.exists(f"CoT_experiments/data/openai_batch/requests/{model}_test.jsonl"):
        continue
    request_list = []
    for seed in range(4):
        for use_reasoning in [False, True]:
            for task_name in task_names:
                file_name = f"CoT_experiments/data/presto_reasoning_data/{task_name}/test.json"
                with open(file_name, 'r') as f:
                    data = json.load(f)
                data = data[:n_test_samples] # only test 100 samples
                for n_shot in [0, 1, 2, 3, 4, 5, 6, 7]:
                    for i, d in enumerate(data):
                        body_dict = get_openai_request_body(d, n_shot, task_name, use_reasoning, seed, model)
                        request_dict = {
                            "custom_id": f"{model}_seed{seed}_reasoning{use_reasoning}_{task_name}_nshot{n_shot}_test_instance{i}",
                            "method": "POST",
                            "url": "/v1/chat/completions",
                            "body": body_dict
                        }
                        request_list.append(request_dict)
            with open(f"CoT_experiments/data/openai_batch/requests/{model}_test.jsonl", "w") as f:
                for request_dict in request_list:
                    f.write(json.dumps(request_dict) + "\n")
############################################################################################################


# 3. Run OpenAI batch API requests
jsonl_files = [
    # "CoT_experiments/data/openai_batch/requests/gpt-3.5-turbo_test.jsonl",
    # "CoT_experiments/data/openai_batch/requests/gpt-4o-mini_test.jsonl",
    "CoT_experiments/data/openai_batch/requests/gpt-4o-2024-11-20_test.jsonl",
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
############################################################################################################


# 4. Parse output
jsonl_files = [
    # "CoT_experiments/data/openai_batch/responses/gpt-3.5-turbo_test.jsonl",
    # "CoT_experiments/data/openai_batch/responses/gpt-4o-mini_test.jsonl",
    "CoT_experiments/data/openai_batch/responses/gpt-4o-2024-11-20_test.jsonl",
]

price = 0.
reasoning_model_generated_dict = {}
answer_predict_dict = {}
raw_response_dict = {}
for jsonl_file in jsonl_files:
    with open(jsonl_file, "r") as f:
        data = [json.loads(line) for line in f.readlines()]
    for d in tqdm(data):
        price += get_price(d['response']['body'])
        custom_id = d['custom_id']
        raw_response = d['response']['body']['choices'][0]['message']['content']
        reasoning_text, answer = parse_output(raw_response)
        reasoning_model_generated_dict[custom_id] = reasoning_text
        answer_predict_dict[custom_id] = answer[:answer_max_length]
        raw_response_dict[custom_id] = raw_response
############################################################################################################


# 5. Save results to a file
reasoning_ground_truth_dict = {}
answer_ground_truth_dict = {}
question_dict = {}
# for model in ["gpt-4o-mini", "gpt-3.5-turbo", "gpt-4o-2024-11-20"]:
for model in ["gpt-4o-2024-11-20"]:
    result_dict = {}
    for task_name in task_names:
        result_dict[task_name] = {}
        file_name = f"CoT_experiments/data/presto_reasoning_data/{task_name}/test.json"
        with open(file_name, 'r') as f:
            data = json.load(f)
        data = data[:n_test_samples] # only test 100 samples
        for i, d in enumerate(data):
            question = d['user_prompt'].replace("[START_I_SMILES]", "").replace("[END_I_SMILES]", "").strip()
            if task_name == "forward":
                answer_gt = d['product']
            elif task_name == "retro":
                answer_gt = d['reactants']
            elif task_name == "reagent":
                answer_gt = d['reagents']
            elif task_name == "catalyst":
                answer_gt = d['catalyst']
            elif task_name == "solvent":
                answer_gt = d['solvent']
            else:
                raise ValueError(f"Unknown task: {task_name}")
            result_dict[task_name][f"instance_{i}"] = {
                "Question": question,
                "Ground truth answer": answer_gt,
                "Ground truth reasoning": d['reasoning'],
                "Model answer": {},
                "Model reasoning": {},
            }
            for use_reasoning, use_reasoning_text in zip([False, True], ["w/o reasoning instruction", "w/ reasoning instruction"]):
                result_dict[task_name][f"instance_{i}"]["Model answer"][use_reasoning_text] = {}
                result_dict[task_name][f"instance_{i}"]["Model reasoning"][use_reasoning_text] = {}
                for n_shot, n_shot_text in zip([0, 1, 2, 3, 4, 5, 6, 7], ["0-shot", "1-shot", "2-shot", "3-shot", "4-shot", "5-shot", "6-shot", "7-shot"]):
                    result_dict[task_name][f"instance_{i}"]["Model answer"][use_reasoning_text][n_shot_text] = {}
                    result_dict[task_name][f"instance_{i}"]["Model reasoning"][use_reasoning_text][n_shot_text] = {}
                    # reasoning_result_list = [reasoning_model_generated_dict[f"{model}_seed{s}_reasoning{use_reasoning}_{task_name}_nshot{n_shot_text}_test_instance{i}"] for s in range(4)]
                    for seed, seed_text in zip(range(4), ["seed0", "seed1", "seed2", "seed3"]):
                        result_dict[task_name][f"instance_{i}"]["Model reasoning"][use_reasoning_text][n_shot_text][seed_text] = reasoning_model_generated_dict[f"{model}_seed{seed}_reasoning{use_reasoning}_{task_name}_nshot{n_shot}_test_instance{i}"]
                        result_dict[task_name][f"instance_{i}"]["Model answer"][use_reasoning_text][n_shot_text][seed_text] = answer_predict_dict[f"{model}_seed{seed}_reasoning{use_reasoning}_{task_name}_nshot{n_shot}_test_instance{i}"]
    # Save to CoT_experiments/results/cot_prompt_test/gt_preds/{model}.json
    with open(f"CoT_experiments/results/cot_prompt_test/gt_preds/{model}.json", "w") as f:
        json.dump(result_dict, f, indent=2)
############################################################################################################


# 6. Evaluate results
evaluator = MoleculeSMILESEvaluator()
# for model in ["gpt-4o-mini", "gpt-3.5-turbo", "gpt-4o-2024-11-20"]:
for model in ["gpt-4o-2024-11-20"]:
    eval_results = {}
    for task_name in task_names:
        eval_results[task_name] = {}
        for use_reasoning, use_reasoning_text in zip([False, True], ["w/o reasoning instruction", "w/ reasoning instruction"]):
            eval_results[task_name][use_reasoning_text] = {}
            for n_shot, n_shot_text in zip([0, 1, 2, 3, 4, 5, 6, 7], ["0-shot", "1-shot", "2-shot", "3-shot", "4-shot", "5-shot", "6-shot", "7-shot"]):
                eval_results[task_name][use_reasoning_text][n_shot_text] = {}
                for seed, seed_text in zip(range(4), ["seed0", "seed1", "seed2", "seed3"]):
                    answer_predict_list = [answer_predict_dict[f"{model}_seed{seed}_reasoning{use_reasoning}_{task_name}_nshot{n_shot}_test_instance{i}"] for i in range(n_test_samples)]
                    answer_gt_list = [result_dict[task_name][f"instance_{i}"]["Ground truth answer"] for i in range(n_test_samples)]
                    eval_results[task_name][use_reasoning_text][n_shot_text][seed_text] = evaluator.evaluate(answer_predict_list, answer_gt_list)
                    print(f"{model} {task_name} {use_reasoning_text} {n_shot_text} {seed_text}: {eval_results[task_name][use_reasoning_text][n_shot_text][seed_text]}")
    # Save to CoT_experiments/results/cot_prompt_test/eval_results/{model}.json
    with open(f"CoT_experiments/results/cot_prompt_test/eval_results/{model}.json", "w") as f:
        json.dump(eval_results, f, indent=2)
############################################################################################################

