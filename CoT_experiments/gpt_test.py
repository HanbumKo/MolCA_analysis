import re
import selfies as sf
import json
import os
from openai import OpenAI
import pandas as pd
import numpy as np
from tqdm import tqdm
from glob import glob
from collections import defaultdict
from utils.help_funcs import calculate_smiles_metrics


os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

client = OpenAI(api_key="xx")



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


def get_openai_output(message, model="gpt-3.5-turbo", task_name="homo"):
    if task_name == "homo" or task_name == "lumo" or task_name == "gap":
        system_message = (
            "You are a chemistry expert and a helpful assistant. "
            "The user will ask about molecules and their property values, "
            "and you must provide those property values without using tools. "
            "When you answer, please explicitly show your detailed reasoning steps "
            "(chain-of-thought) before providing the final numerical answer. "
            "The numerical answer should be enclosed with <NUM> and </NUM>. "
            "Molecules will be given in the SMILES format enclosed with <SMILES> and </SMILES>."
        )
    elif task_name == "forward" or task_name == "retro" or task_name == "reagent":
        system_message = (
            "Your task is to predict the outcomes related to chemical reaction tasks that the user asks about. "
            "The user will mainly inquire about forward reaction prediction, single-step retrosynthesis, "
            "and reagent prediction. The user will provide the molecular SMILES enclosed within `<SMILES>` "
            "and `</SMILES>`. You should provide your prediction as a SMILES representation enclosed within `<ANSWER>` and `</ANSWER>`. "
            "When responding, make sure to include the full reasoning process leading to your prediction."
        )
    else:
        raise ValueError(f"Invalid task: {task_name}")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": system_message,
            },
            {
                "role": "user",
                "content": message
            }
        ],
        temperature=0.0,
        max_tokens=500
    )
    output = response.choices[0].message.content.strip()

    return output


def smiles_to_selfies(smiles):
    try:
        selfies_string = sf.encoder(smiles)
        return selfies_string
    except Exception as e:
        print(f"Error during conversion: {e}")
        return "NONE"



################################################################################################


"""
# Property prediction tasks (regression)
test_i = 99999
files = [
    ("data/biot5_plus_data/tasks_plus/task279_property_prediction_molinst_mol_homo_test.json", "homo"),
    ("data/biot5_plus_data/tasks_plus/task282_property_prediction_molinst_mol_lumo_test.json", "lumo"),
    ("data/biot5_plus_data/tasks_plus/task285_property_prediction_molinst_mol_gap_test.json", "gap"),
]
models = ["gpt-3.5-turbo", "gpt-4o-2024-11-20"]
for model in models:
    for file_name, task_name in files:
        ground_truth_list = []
        raw_output_list = []
        prediction_list = []
        with open(file_name, 'r') as f:
            dataset = json.load(f)["Instances"]
        for i, d in enumerate(dataset):
            instruction = d['instruction']
            iupac = d['input'].split('<boi>')[1].split('<eoi>')[0]
            smiles = d['input'].split("[START_I_SMILES]")[1].split("[END_I_SMILES]")[0]
            user_message = f"{instruction}\n\n<SMILES>{smiles}</SMILES>"
            ground_truth = d['output'][0]
            raw_response = get_openai_output(message=user_message, model=model, task_name=task_name)
            if "<NUM>" in raw_response and "</NUM>" in raw_response:
                prediction = raw_response.split("<NUM>")[1].split("</NUM>")[0]
            else:
                prediction = None
            ground_truth_list.append(ground_truth)
            raw_output_list.append(raw_response)
            prediction_list.append(prediction)
            if i%test_i == test_i-1:
                break
        mae, mse, rmse, validity = regression_evaluate_gpt(prediction_list, ground_truth_list)
        eval_results = {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "validity": validity
        }
        print("="*100)
        print(f"Task: {task_name}, Model: {model}")
        print(f"MAE: {mae}, MSE: {mse}, RMSE: {rmse}, Validity: {validity}")
        print("="*100)
        print()
        # save eval_results to file
        with open(f"CoT_experiments/results/cot_prompt_test/eval_results/{task_name}_{model}.txt", "w") as f:
            f.write(str(eval_results))
        gt_preds = [
            {"ground_truth": gt, "prediction": pred, "raw_output": raw} for gt, pred, raw in zip(ground_truth_list, prediction_list, raw_output_list)
        ]
        with open(f"CoT_experiments/results/cot_prompt_test/gt_preds/{task_name}_{model}.json", "w") as f:
            json.dump(gt_preds, f, indent=4)
        print(f"Results saved for task: {task_name}, model: {model}")
"""



# Reaction prediction tasks
test_i = 99999
files = [
    ("data/biot5_plus_data/tasks_plus/task216_forward_reaction_prediction_molinst_mol_test.json", "forward"),
    ("data/biot5_plus_data/tasks_plus/task219_retrosynthesis_molinst_mol_test.json", "retro"),
    ("data/biot5_plus_data/tasks_plus/task213_reagent_prediction_molinst_mol_test.json", "reagent"),
]
models = ["gpt-3.5-turbo", "gpt-4o-2024-11-20"]
for model in models:
    for file_name, task_name in files:
        ground_truth_list = []
        raw_output_list = []
        prediction_list = []
        with open(file_name, 'r') as f:
            dataset = json.load(f)["Instances"]
        for i, d in enumerate(dataset):
            instruction = d['instruction']
            # iupac = d['input'].split('<boi>')[1].split('<eoi>')[0]
            smiles = d['input'].split("[START_I_SMILES]")[1].split("[END_I_SMILES]")[0]
            user_message = f"{instruction}\n\n<SMILES>{smiles}</SMILES>"
            ground_truth = d['output'][0].split("[START_I_SMILES]")[1].split("[END_I_SMILES]")[0]
            raw_response = get_openai_output(message=user_message, model=model, task_name=task_name)
            if "<ANSWER>" in raw_response and "</ANSWER>" in raw_response:
                prediction = raw_response.split("<ANSWER>")[1].split("</ANSWER>")[0]
            else:
                prediction = None
            ground_truth_list.append(ground_truth)
            raw_output_list.append(raw_response)
            prediction_list.append(prediction)
            if i%test_i == test_i-1:
                break
        if task_name == "forward":
            eval_results = calculate_smiles_metrics(prediction_list, ground_truth_list, metrics=('exact_match', 'fingerprint'))
        elif task_name == "retro" or task_name == "reagent":
            eval_results = calculate_smiles_metrics(prediction_list, ground_truth_list, metrics=('exact_match', 'fingerprint', 'multiple_match'))
        print("="*100)
        print(f"Task: {task_name}, Model: {model}")
        for k, v in eval_results.items():
            print(f"{k}: {v}")
        print("="*100)
        print()
        # save eval_results to file
        with open(f"CoT_experiments/results/cot_prompt_test/eval_results/{task_name}_{model}.txt", "w") as f:
            f.write(str(eval_results))
        gt_preds = [
            {"ground_truth": gt, "prediction": pred, "raw_output": raw} for gt, pred, raw in zip(ground_truth_list, prediction_list, raw_output_list)
        ]
        with open(f"CoT_experiments/results/cot_prompt_test/gt_preds/{task_name}_{model}.json", "w") as f:
            json.dump(gt_preds, f, indent=4)
        print(f"Results saved for task: {task_name}, model: {model}")