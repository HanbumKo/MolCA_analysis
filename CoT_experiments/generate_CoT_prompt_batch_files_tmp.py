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


# def get_openai_request_body(data_dict, task_name, model="gpt-4o-mini"):
#     if task_name == "forward":
#         system_message = """You are a chemical reaction description generator, tasked with creating a step-by-step explanation of the process involved in a forward reaction prediction task. This task involves predicting possible products from precursors composed of reactants and reagents. Specifically, you should generate the explanation by following these steps:
#     1. Separate the precursor into reactants and reagents if possible.
#     2. Identify and describe the substructures involved in the reaction and their chemical characteristics that contribute to forming the product.
#     3. Explain the mechanism suggested by how the reagent acts or the reaction conditions.
#     4. If any major intermediates are expected, briefly describe their structure or reaction mechanism.
#     5. Derive the product in SMILES format.

# The user will provide the following information:
#     - Precursor (in SMILES format)
#     - Products (in SMILES format)
#     - Substructures present in the precursor
#     - Substructures present in the products
#     - A description of the chemical reaction and its mechanism

# It is important to relate the substructures present in the precursor to the chemical reaction mechanism. Using all the information provided by the user, synthesize a coherent, logical explanation in a single paragraph, following the steps mentioned above."""
#     elif task_name == "retro":
#         system_message = """You are a retrosynthesis process explanation generator, tasked with creating a text that explains how a product can be transformed back into its reactants. Users will provide the following information:

#     - Reactants in SMILES format
#     - Products in SMILES format
#     - Key substructures found in the reactants
#     - Key substructures found in the products
#     - A description of the chemical reaction and its mechanism

# Using this information, you must generate a step-by-step explanation of how to deduce the reactants, assuming only the product is known. Specifically, you should follow these steps to construct the explanation:

#     1. Analyze the functional groups in the product that are crucial for the chemical reaction.
#     2. Deduce the structural changes that might have occurred in the product.
#     3. Describe the type of chemical reaction and the mechanism that could lead to these changes.
#     4. Infer the reactants (in SMILES format) that could enable such transformations.

# While the user will provide extensive information, your generated text should simulate a reasoning process as if the deduction is made starting solely from the product. You should generate a natural and logical explanation in one concise paragraph without subheading."""

#     elif task_name == "reagent":
#         system_message = """You are a chemical reaction explanation text generator tasked with creating step-by-step reasoning for a reagent prediction task, where the goal is to infer possible reagents that enable the production of given products from specific reactants in a chemical reaction. Your explanation should follow these specific steps:
#     1. Compare the reactants and products to identify any functional groups or bonds that have changed.
#     2. Infer the general mechanism that could facilitate such transformations.
#     3. Describe the reagents and reaction conditions (e.g., acid/base, oxidizing/reducing agents, catalysts) required for the identified reaction type in a step-by-step manner.
#     4. Review whether the proposed reagents can feasibly enable the formation of the products through the suggested mechanism.
#     5. Represent the reagents in SMILES format.

# The user will provide the following information:
#     - Reactants (in SMILES format)
#     - Reagents (in SMILES format)
#     - Products (in SMILES format)
#     - Substructures present in the reactants
#     - Substructures present in the reagents
#     - Substructures present in the products
#     - A description of the chemical reaction and its mechanism

# Ensure the explanation logically correlates the substructures present in the reactants and products with the reaction mechanism. Synthesize all the provided information into a coherent, natural, and logical explanation in a single paragraph, adhering to the outlined steps."""
#     elif task_name == "catalyst":
#         system_message = """You are a chemical reaction explanation text generator tasked with creating step-by-step reasoning for a catalyst prediction task, where the goal is to infer possible catalyst that enable the production of given products from specific reactants in a chemical reaction. Your explanation should follow these specific steps:
#     1. Identify the reaction type and key conditions (temperature, pH, acid/base, etc.).
#     2. List potential catalyst candidates commonly used for the reaction (e.g., acid/base catalysts, metal complexes).
#     3. Briefly explain the role of the catalyst in the reaction mechanism.
#     4. Select a catalyst compatible with the reactants and propose reaction conditions.
#     5. Summarize the reasons for selecting the catalyst and provide the catalyst in SMILES format.

# The user will provide the following information:
#     - Reactants (in SMILES format)
#     - Catalyst (in SMILES format)
#     - Products (in SMILES format)
#     - Substructures present in the reactants
#     - Substructures present in the catalyst
#     - Substructures present in the products
#     - A description of the chemical reaction and its mechanism

# Ensure the explanation logically correlates the substructures present in the reactants and products with the reaction mechanism. Synthesize all the provided information into a coherent, natural, and logical explanation in a single paragraph, adhering to the outlined steps."""
#     elif task_name == "solvent":
#         system_message = """You are a chemical reaction explanation text generator tasked with creating step-by-step reasoning for a solvent prediction task, where the goal is to infer possible solvent that enable the production of given products from specific reactants in a chemical reaction. Your explanation should follow these specific steps:
#     1. Identify reaction mechanisms (acid/base, oxidation/reduction, etc.) and reaction sensitivities (heat, moisture, etc.).
#     2. Classify candidates based on solvent properties (polarity, boiling point, viscosity, etc.).
#     3. Consider the stability of reactants and products (water solubility, acid/base resistance, etc.).
#     4. Evaluate practical factors such as toxicity, cost, and flammability.
#     5. Summarize the rationale for selecting the optimal solvent (and co-solvent) and derive it in SMILES format.

# The user will provide the following information:
#     - Reactants (in SMILES format)
#     - Solvent (in SMILES format)
#     - Products (in SMILES format)
#     - Substructures present in the reactants
#     - Substructures present in the solvent
#     - Substructures present in the products
#     - A description of the chemical reaction and its mechanism

# Ensure the explanation logically correlates the substructures present in the reactants and products with the reaction mechanism. Synthesize all the provided information into a coherent, natural, and logical explanation in a single paragraph, adhering to the outlined steps."""
#     else:
#         raise ValueError(f"Invalid task: {task_name}")
    
#     user_message = ""
#     if data_dict.get("precursor"):
#         user_message += f"### Precursor (SMILES format)\n{data_dict['precursor']}\n\n"
#     if data_dict.get("reactants"):
#         user_message += f"### Reactants (SMILES format)\n{data_dict['reactants']}\n\n"
#     if data_dict.get("reagents"):
#         user_message += f"### Reagents (SMILES format)\n{data_dict['reagents']}\n\n"
#     if data_dict.get("catalyst"):
#         user_message += f"### Catalyst (SMILES format)\n{data_dict['catalyst']}\n\n"
#     if data_dict.get("solvent"):
#         user_message += f"### Solvent (SMILES format)\n{data_dict['solvent']}\n\n"
#     if data_dict.get("product"):
#         user_message += f"### Products (SMILES format)\n{data_dict['product']}\n\n"
#     if data_dict.get("exist_precursor"):
#         user_message += f"### Substructures present in the precursor\n{data_dict['exist_precursor']}\n\n"
#     if data_dict.get("exist_reactants"):
#         user_message += f"### Substructures present in the reactants\n{data_dict['exist_reactants']}\n\n"
#     if data_dict.get("exist_reagents"):
#         user_message += f"### Substructures present in the reagents\n{data_dict['exist_reagents']}\n\n"
#     if data_dict.get("exist_catalyst"):
#         user_message += f"### Substructures present in the catalyst\n{data_dict['exist_catalyst']}\n\n"
#     if data_dict.get("exist_solvent"):
#         user_message += f"### Substructures present in the solvent\n{data_dict['exist_solvent']}\n\n"
#     if data_dict.get("exist_product"):
#         user_message += f"### Substructures present in the products\n{data_dict['exist_product']}\n\n"
#     if data_dict.get("predicted_reaction"):
#         user_message += f"### Explanation of the chemical reaction and its mechanism\nThe reaction is {data_dict['predicted_reaction']}. {data_dict['predicted_reaction_doc']}"

#     body_dict = {
#         "model": model,
#         "messages": [
#             {
#                 "role": "system",
#                 "content": system_message,
#             },
#             {
#                 "role": "user",
#                 "content": user_message
#             }
#         ],
#         "temperature": 0.0,
#         "max_tokens": 1000
#     }
    
#     return body_dict

def get_request_body(data_dict, use_react_doc, use_subs, use_step_inst, text_len, task_name, model="gpt-4o-mini"):
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
    2. Identify and describe the substructures involved in the reaction and their chemical characteristics that contribute to forming the product.
    3. Infer the mechanism suggested by how the reagent acts or the reaction conditions.
    4. Derive the product in SMILES format.

"""
        system_message += f"""The user will provide the following information:
    - Precursor (in SMILES format)
    - Product (in SMILES format)"""
        if use_subs:
            system_message += """
    - Substructures present in the precursor
    - Substructures present in the product"""
        if use_react_doc:
            system_message += """
    - A description of the chemical reaction and its mechanism"""





    elif task_name == "retro":
        system_message = "You are a retrosynthesis process explanation generator, tasked with creating a text that explains how a product can be transformed back into its reactant."
        if use_step_inst:
            system_message += f""" Specifically, you should generate the explanation to follow following steps:
    1. Analyze the functional groups in the product that are crucial for the chemical reaction.
    2. Infer the type of chemical reaction and the mechanism given the functional groups in product.
    3. Infer the reactant (in SMILES format) that could enable such transformations.

"""
        system_message += f"""The user will provide the following information:
    - Reactant (in SMILES format)
    - Product (in SMILES format)"""
        if use_subs:
            system_message += """
    - Substructures present in the reactant
    - Substructures present in the product"""
        if use_react_doc:
            system_message += """
    - A description of the chemical reaction and its mechanism"""





    elif task_name == "reagent":
        system_message = "You are a chemical reaction explanation text generator tasked with creating reasoning for a reagent prediction task, where the goal is to infer possible reagent that enable the production of given product from specific reactant in a chemical reaction."
        if use_step_inst:
            system_message += f""" Specifically, you should generate the explanation to follow following steps:
    1. Compare the reactant and product to identify any functional groups or bonds that have changed.
    2. Infer the general mechanism that could facilitate such transformations.
    3. Infer the reagent (in SMILES format) that could enable such transformations.

"""
        system_message += f"""The user will provide the following information:
    - Reactant (in SMILES format)
    - Reagent (in SMILES format)
    - Product (in SMILES format)"""
        if use_subs:
            system_message += """
    - Substructures present in the reactant
    - Substructures present in the reagent
    - Substructures present in the product"""
        if use_react_doc:
            system_message += """
    - A description of the chemical reaction and its mechanism"""


    elif task_name == "catalyst":
        system_message = "You are a chemical reaction explanation text generator tasked with creating step-by-step reasoning for a catalyst prediction task, where the goal is to infer possible catalyst that enable the production of given product from specific reactant in a chemical reaction."
        if use_step_inst:
            system_message += f""" Specifically, you should generate the explanation to follow following steps:
    1. Compare the reactant and product to identify any functional groups or bonds that have changed.
    2. Infer the general mechanism that could facilitate such transformations.
    3. Infer the catalyst (in SMILES format) that could enable such transformations.

"""
        system_message += f"""The user will provide the following information:
    - Reactant (in SMILES format)
    - Catalyst (in SMILES format)
    - Product (in SMILES format)"""
        if use_subs:
            system_message += """
    - Substructures present in the reactant
    - Substructures present in the catalyst
    - Substructures present in the product"""
        if use_react_doc:
            system_message += """
    - A description of the chemical reaction and its mechanism"""


    elif task_name == "solvent":
        system_message = "You are a chemical reaction explanation text generator tasked with creating reasoning for a solvent prediction task, where the goal is to infer possible solvent that enable the production of given product from specific reactant in a chemical reaction."
        if use_step_inst:
            system_message += f""" Specifically, you should generate the explanation to follow following steps:
    1. Compare the reactant and product to identify any functional groups or bonds that have changed.
    2. Infer the general mechanism that could facilitate such transformations.
    3. Infer the solvent (in SMILES format) that could enable such transformations.

"""
        system_message += f"""The user will provide the following information:
    - Reactant (in SMILES format)
    - Solvent (in SMILES format)
    - Product (in SMILES format)"""
        if use_subs:
            system_message += """
    - Substructures present in the reactant
    - Substructures present in the solvent
    - Substructures present in the product"""
        if use_react_doc:
            system_message += """
    - A description of the chemical reaction and its mechanism"""

    else:
        raise ValueError(f"Invalid task: {task_name}")

    system_message += f"""

## Important rules:
    - Rely solely on information provided by the user.
    - Demonstrate your reasoning process step by step.
    - Ensure logical consistency in each step and clearly connect each reasoning step.
    - You must write assuming that {task_name_dict[task_name]} is not provided.
    - Never mention {task_name_dict[task_name]} directly, infer it.
    - Do not write content that includes phrases like 'the provided {task_name_dict[task_name]}.'"""
    if text_len == 1 or text_len == 2:
            system_message += f"""
    - Make the description{text_len_dict[text_len]}"""
    
    user_message = ""
    if data_dict.get("precursor"):
        user_message += f"### Precursor (SMILES format)\n{data_dict['precursor']}\n\n"
    if data_dict.get("reactants"):
        user_message += f"### Reactant (SMILES format)\n{data_dict['reactants']}\n\n"
    if data_dict.get("reagents"):
        user_message += f"### Reagent (SMILES format)\n{data_dict['reagents']}\n\n"
    if data_dict.get("catalyst"):
        user_message += f"### Catalyst (SMILES format)\n{data_dict['catalyst']}\n\n"
    if data_dict.get("solvent"):
        user_message += f"### Solvent (SMILES format)\n{data_dict['solvent']}\n\n"
    if data_dict.get("product"):
        user_message += f"### Product (SMILES format)\n{data_dict['product']}\n\n"
    
    if use_subs:
        if data_dict.get("exist_precursor"):
            user_message += f"### Substructures present in the precursor\n{data_dict['exist_precursor']}\n\n"
        if data_dict.get("exist_reactants"):
            user_message += f"### Substructures present in the reactant\n{data_dict['exist_reactants']}\n\n"
        if data_dict.get("exist_reagents"):
            user_message += f"### Substructures present in the reagent\n{data_dict['exist_reagents']}\n\n"
        if data_dict.get("exist_catalyst"):
            user_message += f"### Substructures present in the catalyst\n{data_dict['exist_catalyst']}\n\n"
        if data_dict.get("exist_solvent"):
            user_message += f"### Substructures present in the solvent\n{data_dict['exist_solvent']}\n\n"
        if data_dict.get("exist_product"):
            user_message += f"### Substructures present in the product\n{data_dict['exist_product']}\n\n"
    if use_react_doc:
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
        "temperature": 0.2,
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
    ("data/presto_data/forward/train.json", "train", "forward"),
    ("data/presto_data/forward/test.json", "test", "forward"),
    ("data/presto_data/forward/valid.json", "valid", "forward"),
    ("data/presto_data/retro/train.json", "train", "retro"),
    ("data/presto_data/retro/test.json", "test", "retro"),
    ("data/presto_data/retro/valid.json", "valid", "retro"),
    ("data/presto_data/reagent/train.json", "train", "reagent"),
    ("data/presto_data/reagent/test.json", "test", "reagent"),
    ("data/presto_data/reagent/valid.json", "valid", "reagent"),
    ("data/presto_data/catalyst/train.json", "train", "catalyst"),
    ("data/presto_data/catalyst/test.json", "test", "catalyst"),
    ("data/presto_data/catalyst/valid.json", "valid", "catalyst"),
    ("data/presto_data/solvent/train.json", "train", "solvent"),
    ("data/presto_data/solvent/test.json", "test", "solvent"),
    ("data/presto_data/solvent/valid.json", "valid", "solvent"),
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

        body_dict = get_request_body(data_dict, use_react_doc=True, use_subs=False, use_step_inst=True, text_len=2, task_name=task, model="gpt-4o-mini")
        
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
    