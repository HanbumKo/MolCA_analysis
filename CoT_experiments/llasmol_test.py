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
from pprint import pprint

from llasmol.generation import LlaSMolGeneration



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
    raw_response = raw_response.replace("<unk>", "")
    if "<SMILES>" in raw_response and "</SMILES>" in raw_response:
        answer_text = raw_response.split("<SMILES>")[1].split("</SMILES>")[0].strip()
        reasoning_text = raw_response.split("<SMILES>")[0]
    elif "<SMILES>" not in raw_response and "</SMILES>" in raw_response:
        answer_text = raw_response.split("</SMILES>")[0].strip().split("\n")[-1].split(" ")[-1].strip()
        reasoning_text = raw_response.split("</SMILES>")[0].strip()
    else:
        answer_text = ""
        reasoning_text = raw_response
    answer_text = answer_text.replace("`", "").replace("[SMILES]", "").strip()

    return reasoning_text, answer_text


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


def get_openai_request_body(d, n_shot, task_name, reasoning, seed):
    system_prompt = d['system_prompt']
    if reasoning != "no":
        system_prompt += " The prediction, represented in SMILES notation, should be enclosed with <ANSWER> and </ANSWER> tags."
    messages = [
        {"role": "system", "content": system_prompt}
    ]

    for i in range(n_shot):
        user_prompt = user_prompts[task_name][seed][i].replace("[END_I_SMILES].[START_I_SMILES]", ".").replace("[START_I_SMILES]", "<SMILES> ").replace("[END_I_SMILES]", " </SMILES>").strip()
        if reasoning != "no":
            user_prompt += " Think step by step."
        assistant_prompt = ""
        if reasoning != "no":
            if reasoning == "generated":
                assistant_prompt += f"{reasoning_texts[task_name][seed][i]}\n"
            elif reasoning == "zeroshotcot":
                assistant_prompt += f"{reasoning_zeroshotcot[task_name][seed][i]}\n"
            elif reasoning == "manual":
                raise ValueError("Not implemented")
        assistant_prompt += f"<SMILES> {groudn_truths[task_name][seed][i]} </SMILES>"

        messages.append({"role": "user", "content": user_prompt})
        messages.append({"role": "assistant", "content": assistant_prompt})

    user_prompt = d['user_prompt'].replace("[END_I_SMILES].[START_I_SMILES]", ".").replace("[START_I_SMILES]", "<SMILES> ").replace("[END_I_SMILES]", " </SMILES>").strip()
    if reasoning != "no":
        user_prompt += " Think step by step."

    messages.append({"role": "user", "content": user_prompt})

    body_dict = {
        "messages": messages,
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


task_names = ["forward", "retro", "reagent", "catalyst", "solvent"]
n_test_samples = 100
answer_max_length = 200
# if "CoT_experiments/data/fewshot_example_all.json" already exists, skip
if os.path.exists("CoT_experiments/data/fewshot_example_all.json"):
    with open("CoT_experiments/data/fewshot_example_all.json", "r") as f:
        fewshot_example_all = json.load(f)
    system_prompts = fewshot_example_all["system_prompts"]
    user_prompts = fewshot_example_all["user_prompts"]
    reasoning_texts = fewshot_example_all["reasoning_texts_syntheticreact"]
    reasoning_zeroshotcot = fewshot_example_all["reasoning_texts_gpt4ostepbystep"]
    groudn_truths = fewshot_example_all["ground_truths"]
else:

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

    # 1. Load n-shot examples
    reasoning_texts = {task_name: [[], [], [], []] for task_name in task_names}
    system_prompts = {task_name: [[], [], [], []] for task_name in task_names}
    user_prompts = {task_name: [[], [], [], []] for task_name in task_names}
    groudn_truths = {task_name: [[], [], [], []] for task_name in task_names}


    for task_name in task_names:
        file_name = f"CoT_experiments/data/presto_reasoning_data/{task_name}/train.json"
        with open(file_name, 'r') as f:
            data = json.load(f)
        for seed, idx in enumerate(n_shot_indices[task_name][:7]):
            for i in idx:
                reasoning_texts[task_name][seed].append(data[i]['reasoning'])
                system_prompts[task_name][seed].append(data[i]['system_prompt'])
                user_prompts[task_name][seed].append(data[i]['user_prompt'].replace(" .", ".").replace(" ?", "?").replace("[START_I_SMILES]", "").replace("[END_I_SMILES]", "").strip())
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

generator = LlaSMolGeneration('osunlp/LlaSMol-Mistral-7B', device='cuda')

model = "llasmol"
reasoning_model_generated_dict = {}
answer_predict_dict = {}
raw_response_dict = {}






# if f"CoT_experiments/results/cot_prompt_test/gt_preds/llasmol.json" exists, load and skip
if os.path.exists(f"CoT_experiments/results/cot_prompt_test/gt_preds/llasmol.json"):
    with open(f"CoT_experiments/results/cot_prompt_test/gt_preds/llasmol.json", "r") as f:
        result_dict = json.load(f)
    for task_name in task_names:
        file_name = f"CoT_experiments/data/presto_reasoning_data/{task_name}/test.json"
        with open(file_name, 'r') as f:
            data = json.load(f)
        data = data[:n_test_samples] # only test 100 samples
        for i, d in enumerate(data):
            for reasoning in ["no", "generated", "zeroshotcot"]:
                if reasoning not in result_dict[task_name][f"instance_{i}"]["Model answer"]:
                    result_dict[task_name][f"instance_{i}"]["Model answer"][reasoning] = {}
                    result_dict[task_name][f"instance_{i}"]["Model reasoning"][reasoning] = {}
                    result_dict[task_name][f"instance_{i}"]["Model raw response"][reasoning] = {}
                for n_shot, n_shot_text in zip([0, 1, 2, 3, 4, 5, 6, 7], ["0-shot", "1-shot", "2-shot", "3-shot", "4-shot", "5-shot", "6-shot", "7-shot"]):
                    if n_shot_text in result_dict[task_name][f"instance_{i}"]["Model answer"][reasoning]:
                        result_dict[task_name][f"instance_{i}"]["Model answer"][reasoning][n_shot_text] = {}
                        result_dict[task_name][f"instance_{i}"]["Model reasoning"][reasoning][n_shot_text] = {}
                        result_dict[task_name][f"instance_{i}"]["Model raw response"][reasoning][n_shot_text] = {}
                        messages = [get_openai_request_body(d, n_shot, task_name, reasoning, seed)['messages'][1:] for seed in range(4)]
                        results = generator.generate_given_messages(messages, max_input_tokens=8192, max_new_tokens=1024, batch_size=4, stop_strings="</SMILES>")
                        print(f"processed {model} {task_name} {reasoning} {n_shot_text} instance{i}")
                        for seed, seed_text in zip(range(4), ["seed0", "seed1", "seed2", "seed3"]):
                            result_dict[task_name][f"instance_{i}"]["Model raw response"][reasoning][n_shot_text][seed_text] = results[seed]['output'][0]
                    else:
                        for seed, seed_text in zip(range(4), ["seed0", "seed1", "seed2", "seed3"]):
                            raw_response = result_dict[task_name][f"instance_{i}"]["Model raw response"][reasoning][n_shot_text][seed_text]
                            reasoning_text, answer_text = parse_output(raw_response)
                            result_dict[task_name][f"instance_{i}"]["Model reasoning"][reasoning][n_shot_text][seed_text] = reasoning_text
                            result_dict[task_name][f"instance_{i}"]["Model answer"][reasoning][n_shot_text][seed_text] = answer_text
                            result_dict[task_name][f"instance_{i}"]["Model raw response"][reasoning][n_shot_text][seed_text] = raw_response
                            raw_response_dict[f"{model}_seed{seed}_reasoning{reasoning}_{task_name}_nshot{n_shot_text}_test_instance{i}"] = raw_response
                            reasoning_model_generated_dict[f"{model}_seed{seed}_reasoning{reasoning}_{task_name}_nshot{n_shot_text}_test_instance{i}"] = reasoning_text
                            answer_predict_dict[f"{model}_seed{seed}_reasoning{reasoning}_{task_name}_nshot{n_shot_text}_test_instance{i}"] = answer_text



with open(f"CoT_experiments/results/cot_prompt_test/gt_preds/llasmol.json", "w") as f:
    json.dump(result_dict, f, indent=4)











"""



# if f"CoT_experiments/results/cot_prompt_test/gt_preds/llasmol.json" exists, load and skip
if os.path.exists(f"CoT_experiments/results/cot_prompt_test/gt_preds/llasmol.json"):
    with open(f"CoT_experiments/results/cot_prompt_test/gt_preds/llasmol.json", "r") as f:
        result_dict = json.load(f)
    for task_name in task_names:
        file_name = f"CoT_experiments/data/presto_reasoning_data/{task_name}/test.json"
        with open(file_name, 'r') as f:
            data = json.load(f)
        data = data[:n_test_samples] # only test 100 samples
        for i, d in enumerate(data):
            for reasoning in ["no", "generated"]:
                for n_shot, n_shot_text in zip([0, 1, 2, 3, 4, 5, 6, 7], ["0-shot", "1-shot", "2-shot", "3-shot", "4-shot", "5-shot", "6-shot", "7-shot"]):
                    for seed, seed_text in zip(range(4), ["seed0", "seed1", "seed2", "seed3"]):
                        raw_response = result_dict[task_name][f"instance_{i}"]["Model raw response"][reasoning][n_shot_text][seed_text]
                        reasoning_text, answer_text = parse_output(raw_response)
                        raw_response_dict[f"{model}_seed{seed}_reasoning{reasoning}_{task_name}_nshot{n_shot_text}_test_instance{i}"] = raw_response
                        reasoning_model_generated_dict[f"{model}_seed{seed}_reasoning{reasoning}_{task_name}_nshot{n_shot_text}_test_instance{i}"] = reasoning_text
                        answer_predict_dict[f"{model}_seed{seed}_reasoning{reasoning}_{task_name}_nshot{n_shot_text}_test_instance{i}"] = answer_text

else:
    result_dict = {}
for task_name in task_names:
    if task_name in result_dict:
        continue
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
            "Model raw response": {}
        }
        for reasoning in ["no", "generated", "zeroshotcot"]:
            result_dict[task_name][f"instance_{i}"]["Model answer"][reasoning] = {}
            result_dict[task_name][f"instance_{i}"]["Model reasoning"][reasoning] = {}
            result_dict[task_name][f"instance_{i}"]["Model raw response"][reasoning] = {}
            for n_shot, n_shot_text in zip([0, 1, 2, 3, 4, 5, 6, 7], ["0-shot", "1-shot", "2-shot", "3-shot", "4-shot", "5-shot", "6-shot", "7-shot"]):
                result_dict[task_name][f"instance_{i}"]["Model answer"][reasoning][n_shot_text] = {}
                result_dict[task_name][f"instance_{i}"]["Model reasoning"][reasoning][n_shot_text] = {}
                result_dict[task_name][f"instance_{i}"]["Model raw response"][reasoning][n_shot_text] = {}
                messages = [get_openai_request_body(d, n_shot, task_name, reasoning, seed)['messages'][1:] for seed in range(4)]
                results = generator.generate_given_messages(messages, max_input_tokens=8192, max_new_tokens=1024, batch_size=4, stop_strings="</SMILES>")
                for seed, seed_text in zip(range(4), ["seed0", "seed1", "seed2", "seed3"]):
                    # body_dict = get_openai_request_body(d, n_shot, task_name, reasoning, seed)
                    # messages = body_dict['messages']
                    # messages = messages[1:] # remove system message
                    # result = generator.generate_given_messages([messages], max_input_tokens=8192, max_new_tokens=2048, stop_strings="</SMILES>")[0]
                    raw_response = results[seed]['output'][0]
                    reasoning_text, answer_text = parse_output(raw_response)
                    answer_text = answer_text[:answer_max_length]
                    result_dict[task_name][f"instance_{i}"]["Model reasoning"][reasoning][n_shot_text][seed_text] = reasoning_text
                    result_dict[task_name][f"instance_{i}"]["Model answer"][reasoning][n_shot_text][seed_text] = answer_text
                    result_dict[task_name][f"instance_{i}"]["Model raw response"][reasoning][n_shot_text][seed_text] = raw_response
                    reasoning_model_generated_dict[f"{model}_seed{seed}_reasoning{reasoning}_{task_name}_nshot{n_shot_text}_test_instance{i}"] = reasoning_text
                    answer_predict_dict[f"{model}_seed{seed}_reasoning{reasoning}_{task_name}_nshot{n_shot}_test_instance{i}"] = answer_text
                    print(f"{model} {task_name} {reasoning} {n_shot_text} {seed_text} instance{i}")

    with open(f"CoT_experiments/results/cot_prompt_test/gt_preds/llasmol.json", "w") as f:
        json.dump(result_dict, f, indent=4)




############################################################################################################



evaluator = MoleculeSMILESEvaluator()
model = "llasmol"
eval_results = {}
for task_name in task_names:
    eval_results[task_name] = {}
    for reasoning in ["no", "generated"]:
        eval_results[task_name][reasoning] = {}
        for n_shot, n_shot_text in zip([0, 1, 2, 3, 4, 5, 6, 7], ["0-shot", "1-shot", "2-shot", "3-shot", "4-shot", "5-shot", "6-shot", "7-shot"]):
            eval_results[task_name][reasoning][n_shot_text] = {}
            for seed, seed_text in zip(range(4), ["seed0", "seed1", "seed2", "seed3"]):
                # answer_predict_list = [answer_predict_dict[f"{model}_seed{seed}_reasoning{reasoning}_{task_name}_nshot{n_shot}_test_instance{i}"] for i in range(n_test_samples)]
                answer_predict_list = [result_dict[task_name][f"instance_{i}"]["Model answer"][reasoning][n_shot_text][seed_text] for i in range(n_test_samples)]
                answer_gt_list = [result_dict[task_name][f"instance_{i}"]["Ground truth answer"] for i in range(n_test_samples)]
                eval_results[task_name][reasoning][n_shot_text][seed_text] = evaluator.evaluate(answer_predict_list, answer_gt_list)
                print(f"{model} {task_name} {reasoning} {n_shot_text} {seed_text}: {eval_results[task_name][reasoning][n_shot_text][seed_text]}")
# Save to CoT_experiments/results/cot_prompt_test/eval_results/{model}.json
with open(f"CoT_experiments/results/cot_prompt_test/eval_results/{model}.json", "w") as f:
    json.dump(eval_results, f, indent=4)
"""