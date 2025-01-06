import re
import selfies as sf
import json
import os
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


def get_openai_request_body(data_dict, task_name, model="gpt-4o-mini"):
    if task_name == "forward":
        system_message = """You are a chemical reaction description generator, tasked with creating a step-by-step explanation of the process involved in a forward reaction prediction task. This task involves predicting possible products from precursors composed of reactants and reagents. Specifically, you should generate the explanation by following these steps:
    1. Separate the precursor into reactants and reagents if possible.
    2. Identify and describe the substructures involved in the reaction and their chemical characteristics that contribute to forming the product.
    3. Explain the mechanism suggested by how the reagent acts or the reaction conditions.
    4. If any major intermediates are expected, briefly describe their structure or reaction mechanism.
    5. Derive the product in SMILES format.

The user will provide the following information:
    - Precursor (in SMILES format)
    - Products (in SMILES format)
    - Substructures present in the precursor
    - Substructures present in the products
    - A description of the chemical reaction and its mechanism

It is important to relate the substructures present in the precursor to the chemical reaction mechanism. Using all the information provided by the user, synthesize a coherent, logical explanation in a single paragraph, following the steps mentioned above."""
    elif task_name == "retro":
        system_message = """You are a text generator specialized in explaining chemical reactions, particularly in retrosynthesis tasks, where you predict the reactants capable of producing the given products. Your goal is to generate a step-by-step reasoning process for inferring the reactants, following these specific steps:
    1. Analyze the key functional groups in the products.
    2. Identify the sites where bond formation or transformation is required.
    3. Backtrack to determine the most plausible reaction type (e.g., nucleophilic substitution, acid-base reaction, oxidation-reduction, aromatic substitution, etc.).
    4. Propose the necessary reactants.
    5. Briefly justify the proposed pathway.

The user will provide the following information:
    - Reactants (in SMILES format)
    - Products (in SMILES format)
    - Substructures present in the reactants
    - Substructures present in the products
    - Explanation of the chemical reaction and its mechanism

You should integrate the substructures present in the products with the chemical reaction mechanism for better explanation. Using all the. information provided by the user, generate a natural and logical explanation in one concise paragraph, adhering to the aforementioned steps."""
    elif task_name == "reagent":
        system_message = """You are a chemical reaction explanation text generator tasked with creating step-by-step reasoning for a reagent prediction task, where the goal is to infer possible reagents that enable the production of given products from specific reactants in a chemical reaction. Your explanation should follow these specific steps:
    1. Compare the reactants and products to identify any functional groups or bonds that have changed.
    2. Infer the general mechanism that could facilitate such transformations.
    3. Describe the reagents and reaction conditions (e.g., acid/base, oxidizing/reducing agents, catalysts) required for the identified reaction type in a step-by-step manner.
    4. Review whether the proposed reagents can feasibly enable the formation of the products through the suggested mechanism.
    5. Represent the reagents in SMILES format.

The user will provide the following information:
    - Reactants (in SMILES format)
    - Reagents (in SMILES format)
    - Products (in SMILES format)
    - Substructures present in the reactants
    - Substructures present in the reagents
    - Substructures present in the products
    - A description of the chemical reaction and its mechanism

Ensure the explanation logically correlates the substructures present in the reactants and products with the reaction mechanism. Synthesize all the provided information into a coherent, natural, and logical explanation in a single paragraph, adhering to the outlined steps."""
    elif task_name == "catalyst":
        system_message = """You are a chemical reaction explanation text generator tasked with creating step-by-step reasoning for a catalyst prediction task, where the goal is to infer possible catalyst that enable the production of given products from specific reactants in a chemical reaction. Your explanation should follow these specific steps:
    1. Identify the reaction type and key conditions (temperature, pH, acid/base, etc.).
    2. List potential catalyst candidates commonly used for the reaction (e.g., acid/base catalysts, metal complexes).
    3. Briefly explain the role of the catalyst in the reaction mechanism.
    4. Select a catalyst compatible with the reactants and propose reaction conditions.
    5. Summarize the reasons for selecting the catalyst and provide the output in SMILES format.

The user will provide the following information:
    - Reactants (in SMILES format)
    - Reagents (in SMILES format)
    - Products (in SMILES format)
    - Substructures present in the reactants
    - Substructures present in the reagents
    - Substructures present in the products
    - A description of the chemical reaction and its mechanism

Ensure the explanation logically correlates the substructures present in the reactants and products with the reaction mechanism. Synthesize all the provided information into a coherent, natural, and logical explanation in a single paragraph, adhering to the outlined steps."""
    elif task_name == "solvent":
        system_message = """You are a chemical reaction explanation text generator tasked with creating step-by-step reasoning for a solvent prediction task, where the goal is to infer possible solvent that enable the production of given products from specific reactants in a chemical reaction. Your explanation should follow these specific steps:
    1. Identify reaction mechanisms (acid/base, oxidation/reduction, etc.) and reaction sensitivities (heat, moisture, etc.).
    2. Classify candidates based on solvent properties (polarity, boiling point, viscosity, etc.).
    3. Consider the stability of reactants and products (water solubility, acid/base resistance, etc.).
    4. Evaluate practical factors such as toxicity, cost, and flammability.
    5. Summarize the rationale for selecting the optimal solvent (and co-solvent) and derive it in SMILES format.

The user will provide the following information:
    - Reactants (in SMILES format)
    - Reagents (in SMILES format)
    - Products (in SMILES format)
    - Substructures present in the reactants
    - Substructures present in the reagents
    - Substructures present in the products
    - A description of the chemical reaction and its mechanism

Ensure the explanation logically correlates the substructures present in the reactants and products with the reaction mechanism. Synthesize all the provided information into a coherent, natural, and logical explanation in a single paragraph, adhering to the outlined steps."""
    else:
        raise ValueError(f"Invalid task: {task_name}")
    
    user_message = ""
    if data_dict.get("precursor"):
        user_message += f"### Precursor (SMILES format)\n{data_dict['precursor']}\n\n"
    if data_dict.get("reactants"):
        user_message += f"### Reactants (SMILES format)\n{data_dict['reactants']}\n\n"
    if data_dict.get("reagents"):
        user_message += f"### Reagents (SMILES format)\n{data_dict['reagents']}\n\n"
    if data_dict.get("product"):
        user_message += f"### Products (SMILES format)\n{data_dict['product']}\n\n"
    if data_dict.get("exist_precursor"):
        user_message += f"### Substructures present in the precursor\n{data_dict['exist_precursor']}\n\n"
    if data_dict.get("exist_reactants"):
        user_message += f"### Substructures present in the reactants\n{data_dict['exist_reactants']}\n\n"
    if data_dict.get("exist_reagents"):
        user_message += f"### Substructures present in the reagents\n{data_dict['exist_reagents']}\n\n"
    if data_dict.get("exist_product"):
        user_message += f"### Substructures present in the products\n{data_dict['exist_product']}\n\n"
    if data_dict.get("predicted_reaction"):
        user_message += f"### Explanation of the chemical reaction and its mechanism\nThe reaction is {data_dict['predicted_reaction']}. {data_dict['predicted_reaction_doc']}"

    body_dict = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": system_message,
            },
            {
                "role": "user",
                "content": user_message
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


files = [
    # ("data/presto_data/forward/train-00000-of-00001.json", "train", "forward"),
    # ("data/presto_data/forward/test-00000-of-00001.json", "test", "forward"),
    # ("data/presto_data/retro/train-00000-of-00001.json", "train", "retro"),
    # ("data/presto_data/retro/test-00000-of-00001.json", "test", "retro"),
    ("data/presto_data/reagent/train-00000-of-00001.json", "train", "reagent"),
    ("data/presto_data/reagent/test-00000-of-00001.json", "test", "reagent"),
    ("data/presto_data/reagent/dev-00000-of-00001.json", "valid", "reagent"),
    ("data/presto_data/catalyst/train-00000-of-00001.json", "train", "catalyst"),
    ("data/presto_data/catalyst/test-00000-of-00001.json", "test", "catalyst"),
    ("data/presto_data/catalyst/dev-00000-of-00001.json", "valid", "catalyst"),
    ("data/presto_data/solvent/train-00000-of-00001.json", "train", "solvent"),
    ("data/presto_data/solvent/test-00000-of-00001.json", "test", "solvent"),
    ("data/presto_data/solvent/dev-00000-of-00001.json", "valid", "solvent"),
]

# models = ["gpt-3.5-turbo", "gpt-4o", "gpt-4o-mini"]
model = "gpt-4o-mini"
total_price = 0.

for file_name, split, task in tqdm(files, desc="Loading data", total=len(files)):
    batch_i = 0
    request_list = []
    with open(file_name, 'r') as f:
        data = json.load(f)
    for i, d in tqdm(enumerate(data), total=len(data)):
        data_dict = {}
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

        body_dict = get_openai_request_body(data_dict, task, model=model)
        
        request_dict = {
            "custom_id": f"{task}_{split}_{i}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body_dict
        }
        request_list.append(request_dict)
        if len(request_list) == 15000:
            # Save the request_list as jsonl file in CoT_experiments/data/openai_batch/requests/
            with open(f"CoT_experiments/data/openai_batch/requests/{task}_{split}_batch_{batch_i}.jsonl", "w") as f:
                for req in request_list:
                    f.write(json.dumps(req) + "\n")
            request_list = []
            batch_i += 1
    if request_list:
        with open(f"CoT_experiments/data/openai_batch/requests/{task}_{split}_batch_{batch_i}.jsonl", "w") as f:
            for req in request_list:
                f.write(json.dumps(req) + "\n")
        batch_i += 1
    