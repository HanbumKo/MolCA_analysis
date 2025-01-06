import re
import selfies as sf
import json
import os
import pandas as pd
import numpy as np

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


def get_openai_output(data_dict, task_name, model="gpt-4o-mini"):
    if task_name == "forward":
        system_message = """You are a chemical reaction description generator, tasked with creating a step-by-step explanation of the process involved in a forward reaction prediction task. This task involves predicting possible products from precursors composed of reactants and reagents. Specifically, you should generate the explanation by following these steps:
    1. Separate the precursor into reactants and reagents.
    2. Identify and describe the substructures involved in the reaction and their chemical characteristics that contribute to forming the product.
    3. Explain the mechanism suggested by how the reagent acts or the reaction conditions.
    4. If any major intermediates are expected, briefly describe their structure or reaction mechanism.
    5. Derive the product in SMILES format.

The user will provide the following information:
    - Precursor (in SMILES format)
    - Reactants (in SMILES format)
    - Reagents (in SMILES format)
    - Products (in SMILES format)
    - Substructures present in the reactants
    - Substructures present in the reagents
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
    - Reagents (in SMILES format)
    - Products (in SMILES format)
    - Substructures present in the reactants
    - Substructures present in the reagents
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
    else:
        raise ValueError(f"Invalid task: {task_name}")
    user_message = f"""### Precursor (SMILES format)
{data_dict['precursor']}

### Reactants (SMILES format)
{data_dict['reactants']}

### Reagents (SMILES format)
{data_dict['reagents']}

### Products (SMILES format)
{data_dict['products']}

### Substructures present in the reactants
{data_dict['exist_reactants']}

### Substructures present in the reagents
{data_dict['exist_reagents']}

### Substructures present in the products
{data_dict['exist_products']}

### Explanation of the chemical reaction and its mechanism
The reaction is {data_dict['predicted_reaction']}. {data_dict['predicted_reaction_doc']}
"""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": system_message,
            },
            {
                "role": "user",
                "content": user_message
            }
        ],
        temperature=0.0,
        max_tokens=1000
    )
    output = response.choices[0].message.content.strip()
    if model == "gpt-3.5-turbo":
        prompt_price_per_token = 0.000003
        completion_price_per_token = 0.000006
    elif model == "gpt-4o":
        prompt_price_per_token = 0.0000025
        completion_price_per_token = 0.00001
    elif model == "gpt-4o-mini":
        prompt_price_per_token = 0.00000015
        completion_price_per_token = 0.0000006
    elif model == "o1":
        prompt_price_per_token = 0.000015
        completion_price_per_token = 0.00006
    else:
        prompt_price_per_token = 0.
        completion_price_per_token = 0.
    prompt_tokens = response.usage.prompt_tokens
    completion_tokens = response.usage.completion_tokens
    price = prompt_tokens * prompt_price_per_token + completion_tokens * completion_price_per_token

    return output, price

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
    # Regression tasks
    # (glob(f"data/biot5_plus_data/tasks_plus/*_forward_reaction_prediction_molinst_mol_train.json")[0], "forward"),
    (glob(f"data/biot5_plus_data/tasks_plus/*_forward_reaction_prediction_molinst_mol_test.json")[0], "forward"),
    # (glob(f"data/biot5_plus_data/tasks_plus/*_retrosynthesis_molinst_mol_train.json")[0], "retro",),
    (glob(f"data/biot5_plus_data/tasks_plus/*_retrosynthesis_molinst_mol_test.json")[0], "retro",),
    # (glob(f"data/biot5_plus_data/tasks_plus/*_reagent_prediction_molinst_mol_train.json")[0] , "reagent"),
    (glob(f"data/biot5_plus_data/tasks_plus/*_reagent_prediction_molinst_mol_test.json")[0] , "reagent"),
]

# models = ["gpt-3.5-turbo", "gpt-4o", "gpt-4o-mini"]
model = "gpt-4o-mini"
total_price = 0.

for file_name, task in tqdm(files, desc="Loading data", total=len(files)):
    with open(file_name, 'r') as f:
        data = json.load(f)
    for i, d in tqdm(enumerate(data["Instances"]), total=len(data["Instances"])):
        canonical_rxn = d['canonical_rxn']
        reactants, reagents, products = canonical_rxn.split(">")
        precursor = reactants + "." + reagents
        reactants_mol = Chem.MolFromSmiles(reactants)
        reagents_mol = Chem.MolFromSmiles(reagents)
        products_mol = Chem.MolFromSmiles(products)
        if reactants_mol is None or reagents_mol is None or products_mol is None:
            raise ValueError(f"Invalid SMILES string.\nreactants: {reactants}\nreagents: {reagents}\nproducts: {products}")
        reactants_fp = list(MACCSkeys.GenMACCSKeys(reactants_mol))
        reagents_fp = list(MACCSkeys.GenMACCSKeys(reagents_mol))
        products_fp = list(MACCSkeys.GenMACCSKeys(products_mol))
        reactants_fp_nonzero = [i for i, v in enumerate(reactants_fp) if v]
        reagents_fp_nonzero = [i for i, v in enumerate(reagents_fp) if v]
        products_fp_nonzero = [i for i, v in enumerate(products_fp) if v]
        reactants_subs = [subs[i] for i in reactants_fp_nonzero]
        reagents_subs = [subs[i] for i in reagents_fp_nonzero]
        products_subs = [subs[i] for i in products_fp_nonzero]
        reactants_subs_str = "\n".join(reactants_subs)
        reagents_subs_str = "\n".join(reagents_subs)
        products_subs_str = "\n".join(products_subs)
        predicted_reaction = d['predicted_reaction']
        predicted_reaction_doc = mechanism_docs[predicted_reaction]
        data_dict = {
            "precursor": precursor,
            "reactants": reactants,
            "reagents": reagents,
            "products": products,
            "exist_reactants": reactants_subs_str,
            "exist_reagents": reagents_subs_str,
            "exist_products": products_subs_str,
            "predicted_reaction": predicted_reaction,
            "predicted_reaction_doc": predicted_reaction_doc
        }
        if task == "forward":
            result, price = get_openai_output(data_dict, task, model=model)
            d["reasoning"] = result
        elif task == "reagent":
            result, price = get_openai_output(data_dict, task, model=model)
            d["reasoning"] = result
        elif task == "retro":
            result, price = get_openai_output(data_dict, task, model=model)
            d["reasoning"] = result
        else:
            raise ValueError(f"Invalid task: {task}")
        total_price += price
        if i % 100 == 99:
            print(f"Total price: {total_price}")
            # break
    
    # Save the data to "CoT_experiments/data/biot5_plus_reasoning_data/"
    save_file_name = file_name.split("/")[-1].replace(".json", "_reasoning.json")
    save_file_path = f"CoT_experiments/data/biot5_plus_reasoning_data/{save_file_name}"
    with open(save_file_path, 'w') as f:
        json.dump(data, f, indent=4)

print(f"Total price: {total_price}")