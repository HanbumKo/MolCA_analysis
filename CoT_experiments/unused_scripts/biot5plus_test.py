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
from transformers import AutoTokenizer, T5ForConditionalGeneration
from transformers import GenerationConfig



def get_galactica_output(question, task_name, model_transformers, n_shot):
    n_shot_example_text = ""
    for i in range(n_shot):
        n_shot_example_text += f"Question: {questions[task_name][i]}\n<work>\n{n_shot_examples[task_name][i]}\n</work>\n\nAnswer: [START_I_SMILES]{answers[task_name][i]}[END_I_SMILES]\n\n"
    input_text = f"{n_shot_example_text}Question: {question}\n<work>"
    input_id = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda:0")
    output = model_transformers.generate(
        input_id,
        tokenizer=tokenizer,
        do_sample=False,
        top_p=0.,
        temperature=0.,
        num_beams=1,
        max_new_tokens=500,
        stop_strings="</work>",
    )
    output_text = tokenizer.decode(output[0], skip_special_tokens=False).replace("</s>", "")
    if not "</work>" in output_text.split("<work>")[-1]:
        output_text = output_text + "</work>"
    # print(output_text)
    input_text = f"{output_text}\n\nAnswer: "
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda:0")
    output = model_transformers.generate(
        input_ids,
        tokenizer=tokenizer,
        do_sample=False,
        top_p=0.,
        temperature=0.,
        num_beams=1,
        max_new_tokens=100,
        # stop_strings="</work>",
    )
    output_text = tokenizer.decode(output[0], skip_special_tokens=False)
    if not "</s>" in output_text:
        output_text = output_text + "</s>"
    # print(output_text)
    return output_text


def smiles_to_selfies(smiles):
    try:
        selfies_string = sf.encoder(smiles)
        return selfies_string
    except Exception as e:
        print(f"Error during conversion: {e}")
        return "NONE"


# Import N shot examples
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


# Import questions and answers
questions = {"forward": [], "retro": [], "reagent": []}
answers = {"forward": [], "retro": [], "reagent": []}
# file_name = glob(f"data/biot5_plus_data_backup/tasks_plus/*_forward_reaction_prediction_molinst_mol_train.json")[0]
# with open(file_name, 'r') as f:
#     data = json.load(f)['Instances'][:10]
# for d in data:
#     instruction = d['instruction']
#     input_text = d['input']
#     question = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n<bom>input_text<eom>\n\n### Response:\n"
#     ground_truth = d['output'][0]
#     questions["forward"].append(question)
#     answers["forward"].append(ground_truth)
# file_name = glob(f"data/biot5_plus_data_backup/tasks_plus/*_retrosynthesis_molinst_mol_train.json")[0]
# with open(file_name, 'r') as f:
#     data = json.load(f)['Instances'][:10]
# for d in data:
#     instruction = d['instruction']
#     smiles = d['input'].split("[START_I_SMILES]")[1].split("[END_I_SMILES]")[0]
#     question = f"{instruction}\n[START_I_SMILES]{smiles}[END_I_SMILES]"
#     questions["retro"].append(question)
#     ground_truth = d['output'][0].split("[START_I_SMILES]")[1].split("[END_I_SMILES]")[0]
#     answers["retro"].append(ground_truth)
# file_name = glob(f"data/biot5_plus_data_backup/tasks_plus/*_reagent_prediction_molinst_mol_train.json")[0]
# with open(file_name, 'r') as f:
#     data = json.load(f)['Instances'][:10]
# for d in data:
#     instruction = d['instruction']
#     smiles = d['input'].split("[START_I_SMILES]")[1].split("[END_I_SMILES]")[0]
#     left_smiles, right_smiles = smiles.split(">>")
#     question = f"{instruction}\n[START_I_SMILES]{left_smiles}[END_I_SMILES]>>[START_I_SMILES]{right_smiles}[END_I_SMILES]"
#     questions["reagent"].append(question)
#     ground_truth = d['output'][0].split("[START_I_SMILES]")[1].split("[END_I_SMILES]")[0]
#     answers["reagent"].append(ground_truth)

# f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n<boi>{IUPAC}<eoi> <bom>{SELFIES}<eom>\n\n### Response:\n"

################################################################################################

# Reaction prediction tasks
test_i = 999999
files = [
    ("data/biot5_plus_data_backup/tasks_plus/task216_forward_reaction_prediction_molinst_mol_test.json", "forward"),
    ("data/biot5_plus_data_backup/tasks_plus/task219_retrosynthesis_molinst_mol_test.json", "retro"),
    ("data/biot5_plus_data_backup/tasks_plus/task213_reagent_prediction_molinst_mol_test.json", "reagent"),
]

num_shots = [0, 1, 2]
models = ["QizhiPei/biot5-plus-base-mol-instructions-molecule"]
for model in models:
    for n_shot in num_shots:
        model_replaced = model.split("/")[1]
        tokenizer = AutoTokenizer.from_pretrained(model)
        model_transformers = T5ForConditionalGeneration.from_pretrained(model, device_map=0)
        generation_config = model_transformers.generation_config
        generation_config.max_length = 512
        generation_config.num_beams = 1
        # generation_config = GenerationConfig.from_pretrained(model)
        for file_name, task_name in files:
            ground_truth_list = []
            full_text_list = []
            prediction_list = []
            with open(file_name, 'r') as f:
                dataset = json.load(f)["Instances"]
            for i, d in tqdm(enumerate(dataset), total=len(dataset), desc=f"Task: {task_name}, Model: {model_replaced}"):
                instruction = d['instruction']
                input_text = d['input']
                ground_truth = d['output'][0]
                prompt = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
                input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
                outputs = model_transformers.generate(input_ids, generation_config=generation_config)
                predict = tokenizer.decode(outputs[0], skip_special_tokens=True).replace(" ", "")
                # iupac = d['input'].split('<boi>')[1].split('<eoi>')[0]
                # smiles = d['input'].split("[START_I_SMILES]")[1].split("[END_I_SMILES]")[0]
                # if task_name == "reagent":
                #     left_smiles, right_smiles = smiles.split(">>")
                #     question = f"{instruction}\n[START_I_SMILES]{left_smiles}[END_I_SMILES]>>[START_I_SMILES]{right_smiles}[END_I_SMILES]"
                # else:
                #     question = f"{instruction}\n[START_I_SMILES]{smiles}[END_I_SMILES]"
                # ground_truth = d['output'][0].split("[START_I_SMILES]")[1].split("[END_I_SMILES]")[0]
                # full_text = get_galactica_output(question, task_name=task_name, model_transformers=model_transformers, n_shot=n_shot)
                # prediction = full_text.split("Answer: ")[-1].split("</s>")[0].replace('"', "").strip()
                ground_truth_list.append(ground_truth)
                # full_text_list.append(full_text)
                prediction_list.append(predict)
                if i%test_i == test_i-1:
                    break
            if task_name == "forward":
                eval_results = calculate_smiles_metrics(prediction_list, ground_truth_list, metrics=('exact_match', 'fingerprint'))
            elif task_name == "retro" or task_name == "reagent":
                eval_results = calculate_smiles_metrics(prediction_list, ground_truth_list, metrics=('exact_match', 'fingerprint', 'multiple_match'))
            print("="*100)
            print(f"Task: {task_name}, Model: {model}, n_shot: {n_shot}")
            for k, v in eval_results.items():
                print(f"{k}: {v}")
            print("="*100)
            print()
            # save eval_results to file
            with open(f"CoT_experiments/results/cot_prompt_test/eval_results/{task_name}_{model_replaced}_{n_shot}.txt", "w") as f:
                f.write(str(eval_results))
            gt_preds = [
                {"ground_truth": gt, "prediction": pred, "full_text_list": ft} for gt, pred, ft in zip(ground_truth_list, prediction_list, full_text_list)
            ]
            with open(f"CoT_experiments/results/cot_prompt_test/gt_preds/{task_name}_{model_replaced}_{n_shot}.json", "w") as f:
                json.dump(gt_preds, f, indent=4)
            print(f"Results saved for task: {task_name}, model: {model_replaced}_{n_shot}")