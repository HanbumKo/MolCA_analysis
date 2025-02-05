import random
import torch
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import re

from openai import OpenAI
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import MACCSkeys
from glob import glob
from utils.evaluator import MoleculeSMILESEvaluator

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from llasmol.generation import LlaSMolGeneration
# from llasmol.generation import tokenize as llasmol_tokenize
# from PRESTO.presto.inference import load_trained_lora_model, load_trained_model
# from PRESTO.presto.data_tools import encode_chat, parse_chat_output, encode_interleaved_data
# REASONING_SYSTEM_MESSAGE = """ You must describe your reasoning process in the \"## Reasoning\" section and provide the answer in the \"## Answer\" section."""
# SMILES_PROMPT = """ You should enclose the predicted SMILES in <SMILES> and </SMILES>"""


# REASONING_SYSTEM_MESSAGE = """You are a chemical reaction expert, and you need to provide your reasoning process and answers to the user's questions about chemical reactions. Your response should be divided into two sections. In the first section, labeled `## Reasoning`, you should explain your reasoning process. In the second section, labeled `## Answer`, you should provide your final prediction of the molecule in SMILES format. If there are multiple possible final predicted molecules, only the most probable SMILES should be provided as the answer in `## Answer`. You must not answer with a chemical formula or synonym, such as `H2O` or `DCC`, instead of the molecule's SMILES. Your response should look like the following:

# ## Reasoning

# {Your reasoning process}

# ## Answer

# {Your predicted molecule in SMILES format. Only the most probable SMILES should be provided.}"""
REASONING_SYSTEM_MESSAGE = """ Your response should be divided into two sections. In the first section, labeled `## Reasoning`, you should explain your reasoning process. In the second section, labeled `## Answer`, you should provide your final prediction of the molecule in SMILES format. If there are multiple possible final predicted molecules, only the most probable SMILES should be provided as the answer in `## Answer`. You must not answer with a chemical formula or synonym, such as `H2O` or `DCC`, instead of the molecule's SMILES. Your response should look like the following:

## Reasoning

{Your reasoning process}

## Answer

{Your predicted molecule in SMILES format. Only the most probable SMILES should be provided.}"""

# NO_REASONING_SYSTEM_MESSAGE = """You are an expert in chemical reactions and must provide answers to the user’s questions related to chemical reactions. Without any additional explanation, you should only respond with the predicted molecule in its SMILES format. You must not answer with a chemical formula or synonym, such as `H2O` or `DCC`, instead of the molecule's SMILES."""
NO_REASONING_SYSTEM_MESSAGE = """ Without any additional explanation, you should only respond with the predicted molecule in its SMILES format. If there are multiple possible final predicted molecules, only the most probable SMILES should be provided as the answer. You must not answer with a chemical formula or synonym, such as `H2O` or `DCC`, instead of the molecule's SMILES."""

# import CoT_experiments/keys.txt
with open("CoT_experiments/keys.txt", "r") as f:
    keys = f.readlines()
    keys = [k.strip() for k in keys]
    for key in keys:
        env_name, env_value = key.split("=")
        os.environ[env_name] = env_value

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])



def get_openai_result(system_prompt, user_prompt, use_cot, task, model="gpt-4o-2024-11-20"):
    if task == "forward":
        final_prompt = ""
    elif task == "retro":
        final_prompt = "The all reactants as one SMILES connected by `.` is: "
    elif task == "reagent":
        final_prompt = "The all reagents as one SMILES connected by `.` is: "
    elif task == "catalyst":
        final_prompt = "The all catalysts as one SMILES connected by `.` is: "
    elif task == "solvent":
        final_prompt = "The all solvents as one SMILES connected by `.` is: "
    else:
        raise ValueError(f"Unknown task: {task}")
    if use_cot:
        system_prompt_all = system_prompt + REASONING_SYSTEM_MESSAGE
        # system_prompt_all += f"Here are the few-shot examples:\n\n"
        # for i, (fewshot_user_prompt, fewshot_reasoning, fewshot_ground_truth) in enumerate(zip(fewshot_user_prompts, fewshot_reasonings, fewshot_ground_truths)):
        #     system_prompt_all += f"# User question\n\n{fewshot_user_prompt}\n\n# Your response\n\n## Reasoning\n\n{fewshot_reasoning}\n\n## Answer\n\n{fewshot_ground_truth}\n\n---\n\n"
        # system_prompt_all += "Do not directly reference the `## Reasoning` section of the example above. Instead, you should provide your own reasoning and answer based on the user's question."

        fewshot_messages = []
        for i, (fewshot_user_prompt, fewshot_reasoning, fewshot_ground_truth) in enumerate(zip(fewshot_user_prompts, fewshot_reasonings, fewshot_ground_truths)):
            fewshot_user_prompt = fewshot_user_prompt.replace("[START_I_SMILES]", "").replace("[END_I_SMILES]", "")
            fewshot_messages.append(
                {
                    "role": "user",
                    "content": f"{fewshot_user_prompt}"
                }
            )
            fewshot_messages.append(
                {
                    "role": "assistant",
                    "content": f"## Reasoning\n\n{fewshot_reasoning}\n\n## Answer\n\n{fewshot_ground_truth}"
                }
            )
        body_dict = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt_all,
                },
                *fewshot_messages,
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            "temperature": 0,
            "max_completion_tokens": 1000,
            # "stop": "## Answer"
        }
        response = client.chat.completions.create(**body_dict)
        reasoning_text_answer = response.choices[0].message.content.strip()

        reasoning_text = reasoning_text_answer.split("## Answer")[0].strip()
        answer = reasoning_text_answer.split("## Answer")[-1].strip()

        if answer.endswith("."):
            answer = answer[:-1].strip()

        return reasoning_text, answer
    else:
        fewshot_messages = []
        for i, (user_prompt, reasoning, ground_truth) in enumerate(zip(fewshot_user_prompts, fewshot_reasonings, fewshot_ground_truths)):
            fewshot_messages.append(
                {
                    "role": "user",
                    "content": f"{user_prompt}"
                }
            )
            fewshot_messages.append(
                {
                    "role": "assistant",
                    "content": f"{ground_truth}"
                }
            )
        body_dict = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt + NO_REASONING_SYSTEM_MESSAGE,
                },
                *fewshot_messages,
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            "temperature": 0,
            "max_completion_tokens": 100,
            # "stop": "</SMILES>"
        }
        response = client.chat.completions.create(**body_dict)
        answer = response.choices[0].message.content

        if answer.endswith("."):
            answer = answer[:-1].strip()

        return answer.strip()


def get_llasmol_result(generator, user_prompt, use_cot, task):
    step_instruct = "You should generate the reasoning process to follow following steps:\n"
    if task == "forward":
        step_instruct += """    1. Analyze the substructures in the precursor that are crucial for the chemical reaction.
    2. Infer the type of chemical reaction and the mechanism given the substructures in precursor.
    3. Infer the product in SMILES format."""
        final_prompt = "The predicted product in SMILES format is: "
    elif task == "retro":
        step_instruct += """    1. Analyze the substructures in the product that are crucial for the chemical reaction.
    2. Infer the type of chemical reaction and the mechanism given the functional groups in product.
    3. Infer the reactant (in SMILES format) that could enable such transformations."""
        final_prompt = "The predicted reactant in SMILES format is: "
    elif task == "reagent":
        step_instruct += """    1. Compare the reactant and product to identify any functional groups or bonds that have changed.
    2. Infer the general mechanism that could facilitate such transformations.
    3. Infer the reagent (in SMILES format) that could enable such transformations."""
        final_prompt = "The predicted reagent in SMILES format is: "
    elif task == "catalyst":
        step_instruct += """    1. Compare the reactant and product to identify any functional groups or bonds that have changed.
    2. Infer the general mechanism that could facilitate such transformations.
    3. Infer the catalyst (in SMILES format) that could enable such transformations."""
        final_prompt = "The predicted catalyst in SMILES format is: "
    elif task == "solvent":
        step_instruct += """    1. Compare the reactant and product to identify any functional groups or bonds that have changed.
    2. Infer the general mechanism that could facilitate such transformations.
    3. Infer the solvent (in SMILES format) that could enable such transformations."""
        final_prompt = "The predicted solvent in SMILES format is: "
    else:
        raise ValueError(f"Unknown task: {task}")
    if use_cot:
        # system_prompt_all = system_prompt + REASONING_SYSTEM_MESSAGE + "\n\n"
        system_prompt_all = system_prompt + REASONING_SYSTEM_MESSAGE + "\n\n" + step_instruct + "\n\n"
        # system_prompt_all += f"Here are the few-shot examples:\n\n"
        # for i, (fewshot_user_prompt, fewshot_reasoning, fewshot_ground_truth) in enumerate(zip(fewshot_user_prompts, fewshot_reasonings, fewshot_ground_truths)):
        #     system_prompt_all += f"# User question\n\n{fewshot_user_prompt}\n\n# Your response\n\n## Reasoning\n\n{fewshot_reasoning}\n\n## Answer\n\n{fewshot_ground_truth}\n\n---\n\n"
        # system_prompt_all += "Do not directly reference the `## Reasoning` section of the example above. Instead, you should provide your own reasoning and answer based on the user's question."

        fewshot_messages = []
        for fewshot_i, (fewshot_user_prompt, fewshot_reasoning, fewshot_ground_truth) in enumerate(zip(fewshot_user_prompts, fewshot_reasonings, fewshot_ground_truths)):
            fewshot_user_prompt = fewshot_user_prompt.replace("[END_I_SMILES]>>[START_I_SMILES]", "[END_I_SMILES] >> [START_I_SMILES]").replace("[END_I_SMILES].[START_I_SMILES]", ".").replace("[START_I_SMILES]", "<SMILES> ").replace("[END_I_SMILES]", " </SMILES>")
            if fewshot_i == 0:
                fewshot_messages.append(
                    {
                        "role": "user",
                        "content": f"{system_prompt_all}{fewshot_user_prompt}"
                    }
                )
            else:
                fewshot_messages.append(
                    {
                        "role": "user",
                        "content": f"{fewshot_user_prompt}"
                    }
                )
            fewshot_messages.append(
                {
                    "role": "assistant",
                    "content": f"## Reasoning\n\n{fewshot_reasoning}\n\n## Answer\n\n{fewshot_ground_truth}"
                }
            )
        messages = fewshot_messages + [
            {
                "role": "user",
                "content": user_prompt
            },
            {
                "role": "assistant",
                "content": "## Reasoning\n\n"
            }
        ]
        results = generator.generate_given_messages(messages, max_input_tokens=8192, max_new_tokens=1000, batch_size=1, stop_strings="## Answer", do_sample=False)
        reasoning_text = results[0]['output'][0]
        if "<SMILES>" in reasoning_text and "</SMILES>" in reasoning_text:
            reasoning_text = results[0]['output'][0].split("<SMILES>")[0].strip()
            answer = results[0]['output'][0].split("<SMILES>")[-1].split("</SMILES>")[0].replace("`", "").strip()
            return reasoning_text, answer
        if reasoning_text.endswith("</s>"):
            reasoning_text = reasoning_text[:-4].strip()
        if "## Answer" in reasoning_text:
            reasoning_text = reasoning_text.split("## Answer")[0].strip()
        messages = fewshot_messages + [
            {
                "role": "user",
                "content": user_prompt
            },
            {
                "role": "assistant",
                # "content": f"{reasoning_text}\n\n## Answer\n\n{final_prompt}"
                "content": f"{reasoning_text}\n\n## Answer\n\n"
            }
        ]
        try:
            results = generator.generate_given_messages(messages, max_input_tokens=8192, max_new_tokens=200, batch_size=1, stop_strings="</SMILES>", do_sample=False)
            response = results[0]['output'][0]
        except:
            response = f"{reasoning_text}\n\n## Answer\n\n<SMILES> </SMILES>"
        answer = response.split(final_prompt)[-1].split("<SMILES>")[-1].split("</SMILES>")[0].strip()
        answer = answer.replace(f"{reasoning_text}\n{final_prompt}", "").replace("`", "").replace("</s>", "").strip()
        if answer.endswith("."):
            answer = answer[:-1].strip()


        return reasoning_text, answer
    else:
        messages = [
            {
                "role": "user",
                "content": system_prompt + NO_REASONING_SYSTEM_MESSAGE + "\n\n" + user_prompt
            }
        ],
        results = generator.generate_given_messages(messages, max_input_tokens=8192, max_new_tokens=1000, batch_size=1, stop_strings="</SMILES>", do_sample=False)
        response = results[0]['output'][0]
        answer = response.split("<SMILES>")[-1].split("</SMILES>")[0].replace("`", "").strip()

        return answer


def get_presto_result(model, tokenizer, system_prompt, user_prompt, use_cot, task):
    if task == "forward":
        final_prompt = "A potential product: "
    elif task == "retro":
        final_prompt = "Here are possible reactants: "
    elif task == "reagent":
        final_prompt = "A possible reagents can be "
    elif task == "catalyst":
        final_prompt = "A probable catalyst could be "
    elif task == "solvent":
        final_prompt = "A potential answer could be: "
    else:
        raise ValueError(f"Unknown task: {task}")

    user_prompt = user_prompt.replace("[END_I_SMILES].[START_I_SMILES]", ".").replace("[END_I_SMILES]>>[START_I_SMILES]", ">>")
    all_smiles = re.split(r'\.|>>', user_prompt.split("[START_I_SMILES]")[-1].split("[END_I_SMILES]")[0])
    # sort all_smiles by length
    all_smiles = sorted(all_smiles, key=lambda x: len(x), reverse=True)
    user_prompt = user_prompt.replace("[START_I_SMILES]", "").replace("[END_I_SMILES]", "")
    for smiles in all_smiles:
        user_prompt = user_prompt.replace(smiles, "<molecule_2d>")

    max_new_tokens = 256
    top_k = 50
    top_p = 0.8
    do_sample = True
    temperature = 0.2

    if use_cot:
        d = {
            "molecules": {
                "smiles": all_smiles
            },
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt
                },
                {
                    "role": "assistant",
                    "content": "Let's think step by step. "
                }
            ]
        }
        encoded_dict = encode_chat(d, tokenizer, model.modalities)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids=encoded_dict["input_ids"].unsqueeze(0).to(model.device),
                max_new_tokens=max_new_tokens,
                use_cache=True,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
                temperature=temperature,
                modality_inputs={
                    m.name: [encoded_dict[m.name]] for m in model.modalities
                },
            )
        response = tokenizer.decode(
            output_ids[0, encoded_dict["input_ids"].shape[0]:],
            skip_special_tokens=False,
        ).strip()

        reasoning_text  = response.replace("</s>", "").strip()
        d = {
            "molecules": {
                "smiles": all_smiles
            },
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt
                },
                {
                    "role": "assistant",
                    "content": f"Let's think step by step. {reasoning_text}\n{final_prompt}"
                }
            ]
        }
        encoded_dict = encode_chat(d, tokenizer, model.modalities)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids=encoded_dict["input_ids"].unsqueeze(0).to(model.device),
                max_new_tokens=max_new_tokens,
                use_cache=True,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
                temperature=temperature,
                modality_inputs={
                    m.name: [encoded_dict[m.name]] for m in model.modalities
                },
            )
        response = tokenizer.decode(
            output_ids[0, encoded_dict["input_ids"].shape[0]:],
            skip_special_tokens=True,
        ).strip()
        answer = response.split(" .")[0].split(" ")[-1]
        return f"Let's think step by step. {reasoning_text}\n{final_prompt}", answer

    else:
        d = {
            "molecules": {
                "smiles": all_smiles
            },
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ]
        }
        encoded_dict = encode_chat(d, tokenizer, model.modalities)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids=encoded_dict["input_ids"].unsqueeze(0).to(model.device),
                max_new_tokens=max_new_tokens,
                use_cache=True,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
                temperature=temperature,
                modality_inputs={
                    m.name: [encoded_dict[m.name]] for m in model.modalities
                },
            )
        response = tokenizer.decode(
            output_ids[0, encoded_dict["input_ids"].shape[0]:],
            skip_special_tokens=True,
        ).strip()
        answer = response.split(" .")[0].split(" ")[-1]
        return answer



n_test_samples = 100
# task_names = ["reagent"]
# task_names = ["reagent", "catalyst", "solvent"]
task_names = ["forward", "retro", "reagent", "catalyst", "solvent"]

if not os.path.exists("CoT_experiments/results/zeroshot_test/raw_answer_reasoning_fewshot.json"):
    raw_answer_reasoning_fewshot = {}
    for i in range(1, 11):
        raw_answer_reasoning_fewshot[i] = {}
else:
    with open("CoT_experiments/results/zeroshot_test/raw_answer_reasoning_fewshot.json", "r") as f:
        raw_answer_reasoning_fewshot = json.load(f)


for n_shot in [5]:
# for n_shot in [1, 2, 3, 4, 5]:
    for task_name in task_names:
        file_name = f"CoT_experiments/data/presto_reasoning_data/{task_name}/test.json"
        with open(file_name, 'r') as f:
            data = json.load(f)
        data = data[:n_test_samples]

        fewshot_file_name = f"CoT_experiments/data/presto_reasoning_data/{task_name}/valid.json"
        with open(fewshot_file_name, 'r') as f:
            fewshot_data = json.load(f)
        fewshot_data = fewshot_data[:n_shot]
        fewshot_user_prompts = []
        fewshot_reasonings = []
        fewshot_ground_truths = []
        for d in fewshot_data:
            if task_name == "forward":
                ground_truth = d['product']
            elif task_name == "retro":
                ground_truth = d['reactants']
            elif task_name == "reagent":
                ground_truth = d['reagents']
            elif task_name == "catalyst":
                ground_truth = d['catalyst']
            elif task_name == "solvent":
                ground_truth = d['solvent']
            else:
                raise ValueError(f"Unknown task: {task_name}")
            user_prompt = d['user_prompt']
            fewshot_user_prompts.append(user_prompt)
            fewshot_ground_truths.append(ground_truth)
            fewshot_reasonings.append(d['reasoning'])

        if str(n_shot) not in raw_answer_reasoning_fewshot:
            raw_answer_reasoning_fewshot[str(n_shot)] = {}
        if task_name not in raw_answer_reasoning_fewshot[str(n_shot)]:
            raw_answer_reasoning_fewshot[str(n_shot)][task_name] = {}

        for i, d in enumerate(data):
            system_prompt = d['system_prompt']
            user_prompt = d['user_prompt'].replace("[START_I_SMILES]", "").replace("[END_I_SMILES]", "")
            if task_name == "forward":
                ground_truth = d['product']
            elif task_name == "retro":
                ground_truth = d['reactants']
            elif task_name == "reagent":
                ground_truth = d['reagents']
            elif task_name == "catalyst":
                ground_truth = d['catalyst']
            elif task_name == "solvent":
                ground_truth = d['solvent']
            else:
                raise ValueError(f"Unknown task: {task_name}")
            if not f"instance_{i}" in raw_answer_reasoning_fewshot[str(n_shot)][task_name]:
                raw_answer_reasoning_fewshot[str(n_shot)][task_name][f"instance_{i}"] = {
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "ground_truth": ground_truth,
                }


        """
    ############################################# GPT-3.5 few-shot #############################################
        for i, d in enumerate(data):
            if raw_answer_reasoning_fewshot[n_shot][task_name][f"instance_{i}"].get("gpt-3.5-turbo") is not None:
                continue
            system_prompt = d['system_prompt']
            user_prompt = d['user_prompt'].replace("[START_I_SMILES]", "").replace("[END_I_SMILES]", "")
            if task_name == "forward":
                ground_truth = d['product']
            elif task_name == "retro":
                ground_truth = d['reactants']
            elif task_name == "reagent":
                ground_truth = d['reagents']
            elif task_name == "catalyst":
                ground_truth = d['catalyst']
            elif task_name == "solvent":
                ground_truth = d['solvent']
            else:
                raise ValueError(f"Unknown task: {task_name}")
            reasoning, cot_answer = get_openai_result(system_prompt, user_prompt, use_cot=True, task=task_name, model="gpt-3.5-turbo")
            nocot_answer = get_openai_result(system_prompt, user_prompt, use_cot=False, task=task_name, model="gpt-3.5-turbo")
            raw_answer_reasoning_fewshot[n_shot][task_name][f"instance_{i}"]["gpt-3.5-turbo"] = {
                "nocot_answer": nocot_answer,
                "cot_answer": cot_answer,
                "cot_reasoning_text": reasoning,
            }
            print(f"Task: {task_name}, Instance: {i}, GPT-3.5-turbo")
    #############################################################################################################

    
    ############################################# GPT-4.o few-shot #############################################
        for i, d in enumerate(data):
            if raw_answer_reasoning_fewshot[str(n_shot)][task_name][f"instance_{i}"].get("gpt-4o-2024-11-20") is not None:
                continue
            system_prompt = d['system_prompt']
            user_prompt = d['user_prompt'].replace("[START_I_SMILES]", "").replace("[END_I_SMILES]", "")
            if task_name == "forward":
                ground_truth = d['product']
            elif task_name == "retro":
                ground_truth = d['reactants']
            elif task_name == "reagent":
                ground_truth = d['reagents']
            elif task_name == "catalyst":
                ground_truth = d['catalyst']
            elif task_name == "solvent":
                ground_truth = d['solvent']
            else:
                raise ValueError(f"Unknown task: {task_name}")
            reasoning, cot_answer = get_openai_result(system_prompt, user_prompt, use_cot=True, task=task_name, model="gpt-4o-2024-11-20")
            # nocot_answer = get_openai_result(system_prompt, user_prompt, use_cot=False, task=task_name, model="gpt-4o-2024-11-20")
            raw_answer_reasoning_fewshot[str(n_shot)][task_name][f"instance_{i}"]["gpt-4o-2024-11-20"] = {
                # "nocot_answer": nocot_answer,
                "cot_answer": cot_answer,
                "cot_reasoning_text": reasoning,
            }
            print(f"Task: {task_name}, Instance: {i}, GPT-4o")
            print("=="*100)
            print(f"Ground truth: {ground_truth}")
            print("_"*100)
            print(f"  CoT answer: {cot_answer}")
            print("_"*100)
            print(f"   Reasoning: {reasoning}")
            print("_"*100)
            # print(f"No CoT answer: {nocot_answer}")
            print()
            print()
            print()
            print()
            print()
    #############################################################################################################
        with open(f"CoT_experiments/results/zeroshot_test/raw_answer_reasoning_fewshot.json", "w") as f:
            json.dump(raw_answer_reasoning_fewshot, f, indent=4)

    """
    ############################################# LlaSMol few-shot #############################################
        generator = LlaSMolGeneration('osunlp/LlaSMol-Mistral-7B', device='cuda')
        for i, d in enumerate(data):
            if raw_answer_reasoning_fewshot[str(n_shot)][task_name][f"instance_{i}"].get("LlaSMol") is not None:
                continue
            system_prompt = d['system_prompt']
            user_prompt = d['user_prompt'].replace("[END_I_SMILES]>>[START_I_SMILES]", "[END_I_SMILES] >> [START_I_SMILES]").replace("[END_I_SMILES].[START_I_SMILES]", ".").replace("[START_I_SMILES]", "<SMILES> ").replace("[END_I_SMILES]", " </SMILES>")
            if task_name == "forward":
                ground_truth = d['product']
            elif task_name == "retro":
                ground_truth = d['reactants']
            elif task_name == "reagent":
                ground_truth = d['reagents']
            elif task_name == "catalyst":
                ground_truth = d['catalyst']
            elif task_name == "solvent":
                ground_truth = d['solvent']
            else:
                raise ValueError(f"Unknown task: {task_name}")
            try:
                reasoning = d["reasoning"]
            except KeyError:
                print(task_name, d.keys())
                raise KeyError
            reasoning_text, cot_answer = get_llasmol_result(generator, user_prompt, use_cot=True, task=task_name)
            # nocot_answer = get_llasmol_result(generator, user_prompt, use_cot=False, task=task_name)
        
            raw_answer_reasoning_fewshot[str(n_shot)][task_name][f"instance_{i}"]["LlaSMol"] = {
                # "nocot_answer": nocot_answer,
                "cot_answer": cot_answer,
                "cot_reasoning_text": reasoning_text,
            }
            
            # print("_"*100)
            print(f"Task: {task_name}, Instance: {i}, LlaSMol")
            print("=="*100)
            print(f"Ground truth: {ground_truth}")
            print("_"*100)
            print(f"  CoT answer: {cot_answer}")
            print("_"*100)
            print(f"Reasoning: {reasoning_text}")
            print("_"*100)
            # print(f"No CoT answer: {nocot_answer}")
            print()
            print()
            print()
            print()
            print()
            # print()
        del generator
    #############################################################################################################
        with open(f"CoT_experiments/results/zeroshot_test/raw_answer_reasoning_fewshot.json", "w") as f:
            json.dump(raw_answer_reasoning_fewshot, f, indent=4)
    
    """
    ############################################# PRESTO zero-shot #############################################
        model, tokenizer = load_trained_model(
            model_name_or_path="/home/hko/MolCA_analysis/CoT_experiments/PRESTO/checkpoints/PRESTO",
            pretrained_projectors_path="/home/hko/MolCA_analysis/CoT_experiments/PRESTO/checkpoints/PRESTO/non_lora_trainables.bin",
            load_bits=16,
            device_map="cuda:0"
        )
        for i, d in enumerate(data):
            if raw_answer_reasoning_fewshot[n_shot][task_name][f"instance_{i}"].get("PRESTO") is not None:
                continue
            system_prompt = d['system_prompt']
            user_prompt = d['user_prompt']
            if task_name == "forward":
                ground_truth = d['product']
            elif task_name == "retro":
                ground_truth = d['reactants']
            elif task_name == "reagent":
                ground_truth = d['reagents']
            elif task_name == "catalyst":
                ground_truth = d['catalyst']
            elif task_name == "solvent":
                ground_truth = d['solvent']
            else:
                raise ValueError(f"Unknown task: {task_name}")
            try:
                reasoning = d["reasoning"]
            except KeyError:
                print(task_name, d.keys())
                raise KeyError
            reasoning_text, cot_answer = get_presto_result(model, tokenizer, system_prompt, user_prompt, use_cot=True, task=task_name)
            nocot_answer = get_presto_result(model, tokenizer, system_prompt, user_prompt, use_cot=False, task=task_name)
            raw_answer_reasoning_fewshot[n_shot][task_name][f"instance_{i}"]["PRESTO"] = {
                "nocot_answer": nocot_answer,
                "cot_answer": cot_answer,
                "cot_reasoning_text": reasoning_text,
            }
            # print("_"*100)
            print(f"Task: {task_name}, Instance: {i}, PRESTO")
            # print(f"reasoning_text: {reasoning_text}")
            # print(f"groud_truth: {ground_truth}")
            # print(f"cot_answer: {cot_answer}")
            # print(f"nocot_answer: {nocot_answer}")
            print()
        del model
        del tokenizer
    #############################################################################################################

        with open(f"CoT_experiments/results/zeroshot_test/raw_answer_reasoning_fewshot.json", "w") as f:
            json.dump(raw_answer_reasoning_fewshot, f, indent=4)

        
    ######################################### ReactReasoner zero-shot #########################################
        reactexplainer_model_path = "/home/hko/MolCA_analysis/CoT_experiments/llasmol/checkpoint/CoT-osunlp_LlaSMol-Mistral-7B/checkpoint-92800"
        base_model_name = "mistralai/Mistral-7B-v0.1"
        # base_model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map="cuda")
        # tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        # peft_model = PeftModel.from_pretrained(base_model, reactexplainer_model_path).to("cuda")
        generator = LlaSMolGeneration(reactexplainer_model_path, base_model_name, device='cuda')
        for i, d in enumerate(data):
            # if raw_answer_reasoning_fewshot[n_shot][task_name][f"instance_{i}"].get("ReactReasoner") is not None:
            #     continue
            system_prompt = d['system_prompt']
            user_prompt = d['user_prompt'].replace("[END_I_SMILES]>>[START_I_SMILES]", "[END_I_SMILES] >> [START_I_SMILES]").replace("[END_I_SMILES].[START_I_SMILES]", ".").replace("[START_I_SMILES]", "<SMILES> ").replace("[END_I_SMILES]", " </SMILES>")
            if task_name == "forward":z
                ground_truth = d['product']
            elif task_name == "retro":
                ground_truth = d['reactants']
            elif task_name == "reagent":
                ground_truth = d['reagents']
            elif task_name == "catalyst":
                ground_truth = d['catalyst']
            elif task_name == "solvent":
                ground_truth = d['solvent']
            else:
                raise ValueError(f"Unknown task: {task_name}")
            try:
                reasoning = d["reasoning"]
            except KeyError:
                print(task_name, d.keys())
                raise KeyError
            # input_text = f"<s>[INST] {system_prompt}\n\n{user_prompt} [/INST] "
            nocot_answer = get_llasmol_result(generator, f"{system_prompt}\n\n{user_prompt}", use_cot=False, task=task_name)
        
            raw_answer_reasoning_fewshot[n_shot][task_name][f"instance_{i}"]["ReactReasoner"] = {
                "nocot_answer": nocot_answer,
                # "cot_answer": cot_answer,
                # "cot_reasoning_text": reasoning_text,
            }
            
            # print("_"*100)
            print(f"Task: {task_name}, Instance: {i}, ReactReasoner")
        del generator
    #############################################################################################################

        with open(f"CoT_experiments/results/zeroshot_test/raw_answer_reasoning_fewshot.json", "w") as f:
            json.dump(raw_answer_reasoning_fewshot, f, indent=4)
        """



    # Evaluate
    evaluator = MoleculeSMILESEvaluator()
    task_avg_results = {
        "cot": {
            "exact_match": [],
            "bleu": [],
            "levenshtein": [],
            "rdk_sims": [],
            "maccs_sims": [],
            "morgan_sims": [],
            "validity": [],
        },
    }
    print(f"Few-shot: {n_shot}")
    for task_name in task_names:
        cot_predictions = []
        nocot_predictions = []
        ground_truths = []
        print(f"Task: {task_name}")
        # for model_name in ["gpt-3.5-turbo", "gpt-4o-2024-11-20", "LlaSMol", "PRESTO", "ReactExplainer"]:
        for model_name in ["LlaSMol"]:
            if model_name == "ReactExplainer":
                for i, d in enumerate(data):
                    cot_predictions.append(raw_answer_reasoning_fewshot[n_shot][task_name][f"instance_{i}"][model_name]["cot_answer"])
                    ground_truths.append(raw_answer_reasoning_fewshot[n_shot][task_name][f"instance_{i}"]["ground_truth"])
                cot_result_dict = evaluator.evaluate(cot_predictions, ground_truths)


                print(f"CoT, Task: {task_name}, Model: {model_name} (exact match, validity): {round(cot_result_dict['exact_match'], 3)} / {round(cot_result_dict['validity'], 3)}")
                print(f"CoT, Task: {task_name}, Model: {model_name} (bleu, levenshtein): {round(cot_result_dict['bleu'], 3)} / {round(cot_result_dict['levenshtein'], 3)}")
                print(f"CoT, Task: {task_name}, Model: {model_name} (rdk_sims, maccs_sims, morgan_sims): {round(cot_result_dict['rdk_sims'], 3)} / {round(cot_result_dict['maccs_sims'], 3)} / {round(cot_result_dict['morgan_sims'], 3)}")
                print()
                print()
                print("-"*100)
                print()
                continue
            for i, d in enumerate(data):
                cot_predictions.append(raw_answer_reasoning_fewshot[str(n_shot)][task_name][f"instance_{i}"][model_name]["cot_answer"])
                # nocot_predictions.append(raw_answer_reasoning_fewshot[n_shot][task_name][f"instance_{i}"][model_name]["nocot_answer"])
                ground_truths.append(raw_answer_reasoning_fewshot[str(n_shot)][task_name][f"instance_{i}"]["ground_truth"])
            # nocot_result_dict = evaluator.evaluate(nocot_predictions, ground_truths)
            cot_result_dict = evaluator.evaluate(cot_predictions, ground_truths)

            exact_match_cot = round(cot_result_dict['exact_match'], 3)
            bleu_cot = round(cot_result_dict['bleu'], 3)
            levenshtein_cot = round(cot_result_dict['levenshtein'], 3)
            rdk_sims_cot = round(cot_result_dict['rdk_sims'], 3)
            maccs_sims_cot = round(cot_result_dict['maccs_sims'], 3)
            morgan_sims_cot = round(cot_result_dict['morgan_sims'], 3)
            validity_cot = round(cot_result_dict['validity'], 3)
            print(f"CoT, Model: {model_name}")
            print(f"{exact_match_cot}\t{bleu_cot}\t{levenshtein_cot}\t{rdk_sims_cot}\t{maccs_sims_cot}\t{morgan_sims_cot}\t{validity_cot}")

            # print(f"No CoT, Task: {task_name}, Model: {model_name} (exact match, validity): {round(nocot_result_dict['exact_match'], 3)} / {round(nocot_result_dict['validity'], 3)}")
            # print(f"No CoT, Task: {task_name}, Model: {model_name} (bleu, levenshtein): {round(nocot_result_dict['bleu'], 3)} / {round(nocot_result_dict['levenshtein'], 3)}")
            # print(f"No CoT, Task: {task_name}, Model: {model_name} (rdk_sims, maccs_sims, morgan_sims): {round(nocot_result_dict['rdk_sims'], 3)} / {round(nocot_result_dict['maccs_sims'], 3)} / {round(cot_result_dict['morgan_sims'], 3)}")
            # print(f"CoT, Task: {task_name}, Model: {model_name} (exact match, validity): {round(cot_result_dict['exact_match'], 3)} / {round(cot_result_dict['validity'], 3)}")
            # print(f"CoT, Task: {task_name}, Model: {model_name} (bleu, levenshtein): {round(cot_result_dict['bleu'], 3)} / {round(cot_result_dict['levenshtein'], 3)}")
            # print(f"CoT, Task: {task_name}, Model: {model_name} (rdk_sims, maccs_sims, morgan_sims): {round(cot_result_dict['rdk_sims'], 3)} / {round(cot_result_dict['maccs_sims'], 3)} / {round(cot_result_dict['morgan_sims'], 3)}")
            # print()

            # for key, value in nocot_result_dict.items():
            #     task_avg_results["nocot"][key].append(value)
            #     print(f"No CoT, {model_name} {key}: {round(value, 3)}")
            # for key, value in cot_result_dict.items():
            #     task_avg_results["cot"][key].append(value)
            #     print(f"CoT, {model_name} {key}: {round(value, 3)}")
            print()
            print("-"*100)
    print("="*100)


        # with open(f"CoT_experiments/results/zeroshot_test/raw_answer_reasoning_fewshot.json", "w") as f:
        #     json.dump(raw_answer_reasoning_fewshot, f, indent=4)