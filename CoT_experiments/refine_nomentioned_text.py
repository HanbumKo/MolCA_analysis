import re
import selfies as sf
import json
import os
import time
import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
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


def get_request_body(data_dict, use_react_doc,  use_subs, use_step_inst, text_len, task_name, model="gpt-4o-mini"):
    assert use_react_doc in [True, False]
    assert use_subs in [True, False]
    assert use_step_inst in [True, False]
    assert text_len in [0, 1, 2]

    task_name_dict = {"forward": "product", "retro": "reactant", "reagent": "reagent", "catalyst": "catalyst", "solvent": "solvent"}

    text_len_dict = {0: ".", 1: " in a single paragraph.", 2: " in two paragraphs."}

    if task_name == "forward":
        system_message = f"You are a chemical reaction description generator, tasked with creating a step-by-step explanation of the process involved in a forward reaction prediction task. This task involves predicting possible product from precursors composed of reactant and reagent."
        if use_step_inst:
            system_message += f""" Specifically, you should generate the explanation to follow following steps:"
    1. Separate the precursor into reactant and reagent if possible.
    2. Infer the mechanism suggested by how the reagent acts or the reaction conditions.
    3. Derive the product in SMILES format.

"""

    elif task_name == "retro":
        system_message = "You are a retrosynthesis process explanation generator, tasked with creating a text that explains how a product can be transformed back into its reactant."
        if use_step_inst:
            system_message += f""" Specifically, you should generate the explanation to follow following steps:
    1. Analyze the substructures in the product that are crucial for the chemical reaction.
    2. Infer the type of chemical reaction and the mechanism given the functional groups in product.
    3. Infer the reactant in SMILES format.

"""

    elif task_name == "reagent":
        system_message = "You are a chemical reaction explanation text generator tasked with creating reasoning for a reagent prediction task, where the goal is to infer possible reagent that enable the production of given product from specific reactant in a chemical reaction."
        if use_step_inst:
            system_message += f""" Specifically, you should generate the explanation to follow following steps:
    1. Compare the reactant and product to identify any functional groups or bonds that have changed.
    2. Infer the general mechanism that could facilitate such transformations.
    3. Infer the reagent in SMILES format.

"""

    elif task_name == "catalyst":
        system_message = "You are a chemical reaction explanation text generator tasked with creating step-by-step reasoning for a catalyst prediction task, where the goal is to infer possible catalyst that enable the production of given product from specific reactant in a chemical reaction."
        if use_step_inst:
            system_message += f""" Specifically, you should generate the explanation to follow following steps:
    1. Compare the reactant and product to identify any functional groups or bonds that have changed.
    2. Infer the general mechanism that could facilitate such transformations.
    3. Infer the catalyst in SMILES format.

"""

    elif task_name == "solvent":
        system_message = "You are a chemical reaction explanation text generator tasked with creating reasoning for a solvent prediction task, where the goal is to infer possible solvent that enable the production of given product from specific reactant in a chemical reaction."
        if use_step_inst:
            system_message += f""" Specifically, you should generate the explanation to follow following steps:
    1. Compare the reactant and product to identify any functional groups or bonds that have changed.
    2. Infer the general mechanism that could facilitate such transformations.
    3. Infer the solvent in SMILES format.

"""
    else:
        raise ValueError(f"Invalid task: {task_name}")
    
    system_message += f"""# Information about chemical reactions that users ask about:
"""
    if data_dict.get("precursor"):
        system_message += f"## Precursor (SMILES format)\n`{data_dict['precursor']}`\n\n"
    if data_dict.get("reactants"):
        system_message += f"## Reactant (SMILES format)\n`{data_dict['reactants']}`\n\n"
    if data_dict.get("reagents"):
        system_message += f"## Reagent (SMILES format)\n`{data_dict['reagents']}`\n\n"
    if data_dict.get("catalyst"):
        system_message += f"## Catalyst (SMILES format)\n`{data_dict['catalyst']}`\n\n"
    if data_dict.get("solvent"):
        system_message += f"## Solvent (SMILES format)\n`{data_dict['solvent']}`\n\n"
    if data_dict.get("product"):
        system_message += f"## Product (SMILES format)\n`{data_dict['product']}`\n\n"
    
    if use_subs:
        if data_dict.get("generated_substructure_from_precursor"):
            system_message += f"## Substructure newly formed in the product as a result of the reaction\n{data_dict['generated_substructure_from_precursor']}\n\n"
        if data_dict.get("removed_substructure_from_precursor"):
            system_message += f"## Substructure removed from the precursor as a result of the reaction\n{data_dict['removed_substructure_from_precursor']}\n\n"
        if data_dict.get("generated_substructure_from_reactant"):
            system_message += f"## Substructure newly formed in the product as a result of the reaction\n{data_dict['generated_substructure_from_reactant']}\n\n"
        if data_dict.get("removed_substructure_from_reactant"):    
            system_message += f"## Substructure removed from the reactant as a result of the reaction\n{data_dict['removed_substructure_from_reactant']}\n\n"
    if use_react_doc:
        if data_dict.get("predicted_reaction"):
            system_message += f"## Explanation of the chemical reaction and its mechanism\nThe reaction is {data_dict['predicted_reaction']}. {data_dict['predicted_reaction_doc']}"

    system_message += f"""


# Important rules:
    - You are fully aware of the above information, but when responding to the user, you should answer as if you are reasoning or deducing it.
    - Demonstrate your reasoning process step by step.
    - Ensure logical consistency in each step and clearly connect each reasoning step."""
    if text_len == 1 or text_len == 2:
            system_message += f"""
    - Make the description{text_len_dict[text_len]}"""

    body_dict = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": system_message,
            },
            {
                "role": "user",
                "content": data_dict['user_prompt']
            }
        ],
        "temperature": 0.0,
        "max_tokens": 1000
    }
    
    return body_dict



with open('CoT_experiments/data/maccskeys/MACCSKeys_descriptions.json', 'r') as f:
    maccskeys_descriptions = json.load(f)
    subs = ["NONE"] + [d['substructure'] for d in maccskeys_descriptions.values()]

mechanism_docs = {}
for file_name in glob("CoT_experiments/data/reaction_docs/docs_chatgpt/*.txt"):
    mechanism_name = os.path.basename(file_name).replace(".txt", "").replace("_", " ")
    with open(file_name, 'r') as f:
        data = f.read()
    mechanism_docs[mechanism_name] = data

"""
files = [
    ("CoT_experiments/data/presto_reasoning_data/forward/train.json", "train", "forward"),
    ("CoT_experiments/data/presto_reasoning_data/forward/test.json", "test", "forward"),
    ("CoT_experiments/data/presto_reasoning_data/forward/valid.json", "valid", "forward"),
    ("CoT_experiments/data/presto_reasoning_data/retro/train.json", "train", "retro"),
    ("CoT_experiments/data/presto_reasoning_data/retro/test.json", "test", "retro"),
    ("CoT_experiments/data/presto_reasoning_data/retro/valid.json", "valid", "retro"),
    ("CoT_experiments/data/presto_reasoning_data/reagent/train.json", "train", "reagent"),
    ("CoT_experiments/data/presto_reasoning_data/reagent/test.json", "test", "reagent"),
    ("CoT_experiments/data/presto_reasoning_data/reagent/valid.json", "valid", "reagent"),
    ("CoT_experiments/data/presto_reasoning_data/catalyst/train.json", "train", "catalyst"),
    ("CoT_experiments/data/presto_reasoning_data/catalyst/test.json", "test", "catalyst"),
    ("CoT_experiments/data/presto_reasoning_data/catalyst/valid.json", "valid", "catalyst"),
    ("CoT_experiments/data/presto_reasoning_data/solvent/train.json", "train", "solvent"),
    ("CoT_experiments/data/presto_reasoning_data/solvent/test.json", "test", "solvent"),
    ("CoT_experiments/data/presto_reasoning_data/solvent/valid.json", "valid", "solvent"),
]

# models = ["gpt-3.5-turbo", "gpt-4o", "gpt-4o-mini"]
model = "gpt-4o-mini"
total_price = 0.

batch_i = 0
request_list = []
for file_name, split, task in tqdm(files, desc="Loading data", total=len(files)):
    total = 0
    with open(file_name, 'r') as f:
        data = json.load(f)
    for i, d in tqdm(enumerate(data), total=len(data)):
        if task == "forward":
            gt = d["product"]
        elif task == "retro":
            gt = d["reactants"]
        elif task == "reagent":
            gt = d["reagents"]
        elif task == "catalyst":
            gt = d["catalyst"]
        elif task == "solvent":
            gt = d["solvent"]
        if gt in d["reasoning"]:
            total += 1
            continue

        user_prompt = d['user_prompt'].replace("[START_I_SMILES]", "").replace("[END_I_SMILES]", "")
        data_dict = {}
        data_dict['user_prompt'] = user_prompt
        if d.get("precursor"):
            data_dict["precursor"] = d["precursor"]
            precursor_mol = Chem.MolFromSmiles(d["precursor"])
            precursor_fp = list(MACCSkeys.GenMACCSKeys(precursor_mol))
            precursor_fp_nonzero = [i for i, v in enumerate(precursor_fp) if v]
            precursor_subs = [subs[i] for i in precursor_fp_nonzero]
            precursor_subs_str = "\n".join(precursor_subs)
            data_dict["exist_precursor"] = precursor_subs_str
        if d.get("reactants"):
            data_dict["reactants"] = d["reactants"]
            reactants_mol = Chem.MolFromSmiles(d["reactants"])
            reactants_fp = list(MACCSkeys.GenMACCSKeys(reactants_mol))
            reactants_fp_nonzero = [i for i, v in enumerate(reactants_fp) if v]
            reactants_subs = [subs[i] for i in reactants_fp_nonzero]
            reactants_subs_str = "\n".join(reactants_subs)
            data_dict["exist_reactants"] = reactants_subs_str
        if d.get("reagents"):
            data_dict["reagents"] = d["reagents"]
            reagents_mol = Chem.MolFromSmiles(d["reagents"])
            reagents_fp = list(MACCSkeys.GenMACCSKeys(reagents_mol))
            reagents_fp_nonzero = [i for i, v in enumerate(reagents_fp) if v]
            reagents_subs = [subs[i] for i in reagents_fp_nonzero]
            reagents_subs_str = "\n".join(reagents_subs)
            data_dict["exist_reagents"] = reagents_subs_str
        if d.get("catalyst"):
            data_dict["catalyst"] = d["catalyst"]
            catalyst_mol = Chem.MolFromSmiles(d["catalyst"])
            catalyst_fp = list(MACCSkeys.GenMACCSKeys(catalyst_mol))
            catalyst_fp_nonzero = [i for i, v in enumerate(catalyst_fp) if v]
            catalyst_subs = [subs[i] for i in catalyst_fp_nonzero]
            catalyst_subs_str = "\n".join(catalyst_subs)
            data_dict["exist_catalyst"] = catalyst_subs_str
        if d.get("solvent"):
            data_dict["solvent"] = d["solvent"]
            solvent_mol = Chem.MolFromSmiles(d["solvent"])
            solvent_fp = list(MACCSkeys.GenMACCSKeys(solvent_mol))
            solvent_fp_nonzero = [i for i, v in enumerate(solvent_fp) if v]
            solvent_subs = [subs[i] for i in solvent_fp_nonzero]
            solvent_subs_str = "\n".join(solvent_subs)
            data_dict["exist_solvent"] = solvent_subs_str
        if d.get("product"):
            data_dict["product"] = d["product"]
            products_mol = Chem.MolFromSmiles(d["product"])
            products_fp = list(MACCSkeys.GenMACCSKeys(products_mol))
            products_fp_nonzero = [i for i, v in enumerate(products_fp) if v]
            products_subs = [subs[i] for i in products_fp_nonzero]
            products_subs_str = "\n".join(products_subs)
            data_dict["exist_product"] = products_subs_str
        if d.get("predicted_reaction"):
            data_dict["predicted_reaction"] = d["predicted_reaction"]
            predicted_reaction_doc = mechanism_docs[d["predicted_reaction"]]
            data_dict["predicted_reaction_doc"] = predicted_reaction_doc

        if task == "forward":
            generated_substructure_fp = ((np.array(products_fp) - np.array(precursor_fp))==1)[:-1]
            generated_substructure_fp_nonzero = [i for i, v in enumerate(generated_substructure_fp) if v]
            generated_substructure_subs = [subs[i] for i in generated_substructure_fp_nonzero]
            generated_substructure_subs_str = "\n".join(generated_substructure_subs)
            removed_substructure_fp = ((np.array(precursor_fp) - np.array(products_fp))==1)[:-1]
            removed_substructure_fp_nonzero = [i for i, v in enumerate(removed_substructure_fp) if v]
            removed_substructure_subs = [subs[i] for i in removed_substructure_fp_nonzero]
            removed_substructure_subs_str = "\n".join(removed_substructure_subs)
            data_dict["generated_substructure_from_precursor"] = generated_substructure_subs_str
            data_dict["removed_substructure_from_precursor"] = removed_substructure_subs_str
        elif task == "retro":
            generated_substructure_fp = ((np.array(products_fp) - np.array(reactants_fp))==1)[:-1]
            generated_substructure_fp_nonzero = [i for i, v in enumerate(generated_substructure_fp) if v]
            generated_substructure_subs = [subs[i] for i in generated_substructure_fp_nonzero]
            generated_substructure_subs_str = "\n".join(generated_substructure_subs)
            removed_substructure_fp = ((np.array(reactants_fp) - np.array(products_fp))==1)[:-1]
            removed_substructure_fp_nonzero = [i for i, v in enumerate(removed_substructure_fp) if v]
            removed_substructure_subs = [subs[i] for i in removed_substructure_fp_nonzero]
            removed_substructure_subs_str = "\n".join(removed_substructure_subs)
            data_dict["generated_substructure_from_reactant"] = generated_substructure_subs_str
            data_dict["removed_substructure_from_reactant"] = removed_substructure_subs_str
        elif task == "reagent":
            generated_substructure_fp = ((np.array(products_fp) - np.array(reactants_fp))==1)[:-1]
            generated_substructure_fp_nonzero = [i for i, v in enumerate(generated_substructure_fp) if v]
            generated_substructure_subs = [subs[i] for i in generated_substructure_fp_nonzero]
            generated_substructure_subs_str = "\n".join(generated_substructure_subs)
            removed_substructure_fp = ((np.array(reactants_fp) - np.array(products_fp))==1)[:-1]
            removed_substructure_fp_nonzero = [i for i, v in enumerate(removed_substructure_fp) if v]
            removed_substructure_subs = [subs[i] for i in removed_substructure_fp_nonzero]
            removed_substructure_subs_str = "\n".join(removed_substructure_subs)
            data_dict["generated_substructure_from_reactant"] = generated_substructure_subs_str
            data_dict["removed_substructure_from_reactant"] = removed_substructure_subs_str
        elif task == "catalyst":
            generated_substructure_fp = ((np.array(products_fp) - np.array(reactants_fp))==1)[:-1]
            generated_substructure_fp_nonzero = [i for i, v in enumerate(generated_substructure_fp) if v]
            generated_substructure_subs = [subs[i] for i in generated_substructure_fp_nonzero]
            generated_substructure_subs_str = "\n".join(generated_substructure_subs)
            removed_substructure_fp = ((np.array(reactants_fp) - np.array(products_fp))==1)[:-1]
            removed_substructure_fp_nonzero = [i for i, v in enumerate(removed_substructure_fp) if v]
            removed_substructure_subs = [subs[i] for i in removed_substructure_fp_nonzero]
            removed_substructure_subs_str = "\n".join(removed_substructure_subs)
            data_dict["generated_substructure_from_reactant"] = generated_substructure_subs_str
            data_dict["removed_substructure_from_reactant"] = removed_substructure_subs_str
        elif task == "solvent":
            generated_substructure_fp = ((np.array(products_fp) - np.array(reactants_fp))==1)[:-1]
            generated_substructure_fp_nonzero = [i for i, v in enumerate(generated_substructure_fp) if v]
            generated_substructure_subs = [subs[i] for i in generated_substructure_fp_nonzero]
            generated_substructure_subs_str = "\n".join(generated_substructure_subs)
            removed_substructure_fp = ((np.array(reactants_fp) - np.array(products_fp))==1)[:-1]
            removed_substructure_fp_nonzero = [i for i, v in enumerate(removed_substructure_fp) if v]
            removed_substructure_subs = [subs[i] for i in removed_substructure_fp_nonzero]
            removed_substructure_subs_str = "\n".join(removed_substructure_subs)
            data_dict["generated_substructure_from_reactant"] = generated_substructure_subs_str
            data_dict["removed_substructure_from_reactant"] = removed_substructure_subs_str


        body_dict = get_request_body(data_dict, use_react_doc=True, use_subs=True, use_step_inst=True, text_len=2, task_name=task, model="gpt-4o-mini")
        
        request_dict = {
            "custom_id": f"{task}_{split}_{i}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body_dict
        }
        request_list.append(request_dict)
        if len(request_list) == 25000:
            # Save the request_list as jsonl file in CoT_experiments/data/openai_batch/requests/
            with open(f"CoT_experiments/data/openai_batch/requests/refine_batch_{batch_i}.jsonl", "w") as f:
                for req in request_list:
                    f.write(json.dumps(req) + "\n")
            request_list = []
            batch_i += 1
    print(f"{task} {split} done.")
    print(f"Total: {total} / {len(data)}")
    print()

if request_list:
    with open(f"CoT_experiments/data/openai_batch/requests/refine_batch_{batch_i}.jsonl", "w") as f:
        for req in request_list:
            f.write(json.dumps(req) + "\n")
    batch_i += 1



    

jsonl_files = [
    "CoT_experiments/data/openai_batch/requests/refine_batch_0.jsonl",
    # "CoT_experiments/data/openai_batch/requests/refine_batch_1.jsonl",
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

"""
jsonl_files = [
    "CoT_experiments/data/openai_batch/responses/refine_batch_0.jsonl",
    # "CoT_experiments/data/openai_batch/responses/refine_batch_1.jsonl",
]


for task_name in ["forward", "retro", "reagent", "catalyst", "solvent"]:
    print(f"Processing {task_name}")
    # Load CoT_experiments/data/presto_reasoning_data/{task_name}/train.json
    with open(f"CoT_experiments/data/presto_reasoning_data/{task_name}/train.json", "r") as f:
        train_data = json.load(f)
    # Load CoT_experiments/data/presto_reasoning_data/{task_name}/valid.json
    with open(f"CoT_experiments/data/presto_reasoning_data/{task_name}/valid.json", "r") as f:
        valid_data = json.load(f)
    # Load CoT_experiments/data/presto_reasoning_data/{task_name}/test.json
    with open(f"CoT_experiments/data/presto_reasoning_data/{task_name}/test.json", "r") as f:
        test_data = json.load(f)

    for jsonl_file in jsonl_files:
        with open(jsonl_file, "r") as f:
            responses = f.readlines()
        for response in tqdm(responses, desc="Processing responses"):
            response = json.loads(response)
            custom_id = response["custom_id"] # retro_train_123346
            task, split, idx = custom_id.split("_")
            idx = int(idx)
            if task != task_name:
                continue
            if split == "train":
                data = train_data
            elif split == "valid":
                data = valid_data
            elif split == "test":
                data = test_data
            data[idx]["reasoning"] = response['response']['body']["choices"][0]["message"]["content"]

    with open(f"CoT_experiments/data/presto_reasoning_data/{task_name}/train.json", "w") as f:
        json.dump(train_data, f, indent=4)
    with open(f"CoT_experiments/data/presto_reasoning_data/{task_name}/valid.json", "w") as f:  
        json.dump(valid_data, f, indent=4)
    with open(f"CoT_experiments/data/presto_reasoning_data/{task_name}/test.json", "w") as f:
        json.dump(test_data, f, indent=4)
