import random
import json
import os
import numpy as np

from openai import OpenAI
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import MACCSkeys
from glob import glob


# import CoT_experiments/keys.txt
with open("CoT_experiments/keys.txt", "r") as f:
    keys = f.readlines()
    keys = [k.strip() for k in keys]
    for key in keys:
        env_name, env_value = key.split("=")
        os.environ[env_name] = env_value

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def get_manual_request_body(data_dict, task_name, model="gpt-4o-2024-11-20"):
    random_text_constrain = random.choice(["in a single paragraph.", "in two paragraphs.", "in three paragraphs.", "."])
    if task_name == "forward":
        system_message = f"""You are a chemical reaction description generator, tasked with creating a step-by-step explanation of the process involved in a forward reaction prediction task. This task involves predicting possible products from precursors composed of reactants and reagents.

The user will provide the following information:
    - Precursor (in SMILES format)
    - Product (in SMILES format)

You should generate an explanation assuming you don't know the correct product. Using all the information provided by the user, synthesize a coherent, logical explanation {random_text_constrain}"""
    elif task_name == "retro":
        system_message = f"""You are a retrosynthesis process explanation generator, tasked with creating a text that explains how a product can be transformed back into its reactant.

The user will provide the following information:
    - Reactant (in SMILES format)
    - Product (in SMILES format)

You should generate an explanation assuming you don't know the correct reactant. Using all the information provided by the user, synthesize a coherent, logical explanation {random_text_constrain}"""
    elif task_name == "reagent":
        system_message = f"""You are a chemical reaction explanation text generator tasked with creating reasoning for a reagent prediction task, where the goal is to infer possible reagents that enable the production of given products from specific reactants in a chemical reaction.

The user will provide the following information:
    - Reactants (in SMILES format)
    - Reagents (in SMILES format)
    - Products (in SMILES format)

You should generate an explanation assuming you don't know the correct reagent. Synthesize all the provided information into a coherent, natural, and logical explanation {random_text_constrain}"""
    elif task_name == "catalyst":
        system_message = f"""You are a chemical reaction explanation text generator tasked with creating a coherent explanation for a catalyst prediction task, where the goal is to infer possible catalyst that enable the production of given products from specific reactants in a chemical reaction.

The user will provide the following information:
    - Reactants (in SMILES format)
    - Catalyst (in SMILES format)
    - Products (in SMILES format)

You should generate an explanation assuming you don't know the correct catalyst. Synthesize all the provided information into a coherent, natural, and logical explanation {random_text_constrain}"""
    elif task_name == "solvent":
        system_message = f"""You are a chemical reaction explanation text generator tasked with creating reasoning for a solvent prediction task, where the goal is to infer possible solvent that enable the production of given products from specific reactants in a chemical reaction.

The user will provide the following information:
    - Reactants (in SMILES format)
    - Solvent (in SMILES format)
    - Products (in SMILES format)

You should generate an explanation assuming you don't know the correct solvent. Synthesize all the provided information into a coherent, natural, and logical explanation {random_text_constrain}"""
    else:
        raise ValueError(f"Invalid task: {task_name}")


    user_message = ""
    if data_dict.get("precursor"):
        user_message += f"### Precursor (SMILES format)\n{data_dict['precursor']}\n\n"
    if data_dict.get("reactants"):
        user_message += f"### Reactants (SMILES format)\n{data_dict['reactants']}\n\n"
    if data_dict.get("reagents"):
        user_message += f"### Reagents (SMILES format)\n{data_dict['reagents']}\n\n"
    if data_dict.get("catalyst"):
        user_message += f"### Catalyst (SMILES format)\n{data_dict['catalyst']}\n\n"
    if data_dict.get("solvent"):
        user_message += f"### Solvent (SMILES format)\n{data_dict['solvent']}\n\n"
    if data_dict.get("product"):
        user_message += f"### Products (SMILES format)\n{data_dict['product']}\n\n"
    # if data_dict.get("exist_precursor"):
    #     user_message += f"### Substructures present in the precursor\n{data_dict['exist_precursor']}\n\n"
    # if data_dict.get("exist_reactants"):
    #     user_message += f"### Substructures present in the reactants\n{data_dict['exist_reactants']}\n\n"
    # if data_dict.get("exist_reagents"):
    #     user_message += f"### Substructures present in the reagents\n{data_dict['exist_reagents']}\n\n"
    # if data_dict.get("exist_catalyst"):
    #     user_message += f"### Substructures present in the catalyst\n{data_dict['exist_catalyst']}\n\n"
    # if data_dict.get("exist_solvent"):
    #     user_message += f"### Substructures present in the solvent\n{data_dict['exist_solvent']}\n\n"
    # if data_dict.get("exist_product"):
    #     user_message += f"### Substructures present in the products\n{data_dict['exist_product']}\n\n"
    # if data_dict.get("predicted_reaction"):
    #     user_message += f"### Explanation of the chemical reaction and its mechanism\nThe reaction is {data_dict['predicted_reaction']}. {data_dict['predicted_reaction_doc']}"

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
        "temperature": random.uniform(0.0, 1.0),
        "max_tokens": 1000
    }
    
    return body_dict



def get_syntheticreact_request_body(data_dict, task_name, model="gpt-4o-mini"):
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
        system_message = """You are a retrosynthesis process explanation generator, tasked with creating a text that explains how a product can be transformed back into its reactants. Users will provide the following information:

    - Reactants in SMILES format
    - Products in SMILES format
    - Key substructures found in the reactants
    - Key substructures found in the products
    - A description of the chemical reaction and its mechanism

Using this information, you must generate a step-by-step explanation of how to deduce the reactants, assuming only the product is known. Specifically, you should follow these steps to construct the explanation:

    1. Analyze the functional groups in the product that are crucial for the chemical reaction.
    2. Deduce the structural changes that might have occurred in the product.
    3. Describe the type of chemical reaction and the mechanism that could lead to these changes.
    4. Infer the reactants (in SMILES format) that could enable such transformations.

While the user will provide extensive information, your generated text should simulate a reasoning process as if the deduction is made starting solely from the product. You should generate a natural and logical explanation in one concise paragraph without subheading."""

    elif task_name == "reagent":
        system_message = """You are a chemical reaction explanation text generator tasked with creating step-by-step reasoning for a reagent prediction task, where the goal is to infer possible reagents that enable the production of given products from specific reactants in a chemical reaction. Your explanation should follow these specific steps:
    1. Compare the reactants and products to identify any functional groups or bonds that have changed.
    2. Infer the general mechanism that could facilitate such transformations.
    3. Describe the reagents and reaction conditions required for the identified reaction type in a step-by-step manner.
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
    5. Summarize the reasons for selecting the catalyst and provide the catalyst in SMILES format.

The user will provide the following information:
    - Reactants (in SMILES format)
    - Catalyst (in SMILES format)
    - Products (in SMILES format)
    - Substructures present in the reactants
    - Substructures present in the catalyst
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
    - Solvent (in SMILES format)
    - Products (in SMILES format)
    - Substructures present in the reactants
    - Substructures present in the solvent
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
    if data_dict.get("catalyst"):
        user_message += f"### Catalyst (SMILES format)\n{data_dict['catalyst']}\n\n"
    if data_dict.get("solvent"):
        user_message += f"### Solvent (SMILES format)\n{data_dict['solvent']}\n\n"
    if data_dict.get("product"):
        user_message += f"### Products (SMILES format)\n{data_dict['product']}\n\n"
    if data_dict.get("exist_precursor"):
        user_message += f"### Substructures present in the precursor\n{data_dict['exist_precursor']}\n\n"
    if data_dict.get("exist_reactants"):
        user_message += f"### Substructures present in the reactants\n{data_dict['exist_reactants']}\n\n"
    if data_dict.get("exist_reagents"):
        user_message += f"### Substructures present in the reagents\n{data_dict['exist_reagents']}\n\n"
    if data_dict.get("exist_catalyst"):
        user_message += f"### Substructures present in the catalyst\n{data_dict['exist_catalyst']}\n\n"
    if data_dict.get("exist_solvent"):
        user_message += f"### Substructures present in the solvent\n{data_dict['exist_solvent']}\n\n"
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



def get_cleaning_text_request_body(origin_text, ground_truth, task_name, model="gpt-4o-2024-11-20"):
    if task_name == "forward":
        system_message = """You are an expert in chemistry and natural language processing. Your task is to revise the provided reasoning text for a forward reaction prediction problem. Ensure that the text no longer includes any direct mention of the correct product SMILES. You may retain references to other SMILES that are not the product, or you can rewrite the reasoning to maintain coherence without directly referring to the product SMILES. Ensure the text remains scientifically accurate and logical."""
        user_message = f"""The product’s SMILES string is "{ground_truth}". Here is a reasoning text for a forward reaction prediction problem. Please remove any direct mention of the correct product SMILES while ensuring the reasoning remains accurate and logical. You can refer to other SMILES in the reasoning if needed, but ensure the correct product SMILES is not explicitly mentioned or identifiable. Here is the text:

{origin_text}"""
    elif task_name == "retro":
        system_message = """You are an advanced language model specialized in chemical reasoning. Your task is to process retrosynthesis reasoning text. If the text explicitly mentions SMILES strings that refer to reactants (not products), you must remove those specific SMILES strings. You can retain or rewrite reasoning that involves SMILES strings of products. Ensure the rest of the explanation remains coherent and accurate while adhering to this instruction."""
        user_message = f"""The reactant’s SMILES string is "{ground_truth}". Here is a retrosynthesis reasoning text. Remove any SMILES strings that are directly mentioned as reactants, but allow the reasoning to retain or include product SMILES strings where necessary. Rewrite the reasoning text accordingly to make it smooth and logical. Here is the text:

{origin_text}"""
    elif task_name == "reagent":
        system_message = """You are an advanced language model specializing in chemical reasoning. Your task is to edit reasoning text related to reagent prediction. Specifically, you must identify and remove all direct mentions of the SMILES notation of reagents used in the prediction process. Mentions of SMILES related to reactants or products can remain unchanged. Ensure the revised text retains its logical flow and context without revealing the reagent SMILES."""
        user_message = f"""The reagent’s SMILES string is "{ground_truth}". Here is a reasoning text related to reagent prediction. Remove any sections where the SMILES notation of reagents is directly mentioned. Ensure SMILES notations of reactants and products remain intact, and maintain the coherence of the explanation. Here is the text:

{origin_text}"""
    elif task_name == "catalyst":
        system_message = """You are an AI model specializing in text analysis and editing. Your task is to analyze reasoning text related to catalyst prediction problems. When processing the text, exclude any mention of SMILES strings that are directly identified as the correct catalyst. However, retain SMILES strings that are identified as reactants or products. Ensure that the resulting text maintains coherence and logical flow, while adhering to these constraints."""
        user_message = f"""The catalyst’s SMILES string is "{ground_truth}". Given the following reasoning text, remove any reference to SMILES strings that are explicitly identified as the correct catalyst. SMILES strings referring to reactants or products should remain unchanged. Ensure the text remains logically consistent. Here is the text:

{origin_text}"""
    elif task_name == "solvent":
        system_message = """You are an advanced language model tasked with processing reasoning texts about solvent prediction. Your goal is to analyze the provided text and exclude any portion where the correct solvent’s SMILES representation is mentioned directly. Maintain the integrity of the reasoning process, ensuring that any SMILES representations used as reactants or products are preserved. Provide a refined output where solvent SMILES references are removed while keeping the rest of the content intact and clear."""
        user_message = f"""The solvent’s SMILES string is "{ground_truth}". Here is a reasoning text about solvent prediction. Please remove any part of the text that directly mentions the correct solvent’s SMILES representation. Ensure that SMILES representing reactants or products remain untouched and retain all other details necessary for understanding the reasoning. The text is as follows:

{origin_text}"""
    else:
        raise ValueError(f"Invalid task: {task_name}")

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
        "max_tokens": 1300
    }
    
    return body_dict




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
    3. Infer the reactant (in SMILES format) that could enable such transformations.

"""

    elif task_name == "reagent":
        system_message = "You are a chemical reaction explanation text generator tasked with creating reasoning for a reagent prediction task, where the goal is to infer possible reagent that enable the production of given product from specific reactant in a chemical reaction."
        if use_step_inst:
            system_message += f""" Specifically, you should generate the explanation to follow following steps:
    1. Compare the reactant and product to identify any functional groups or bonds that have changed.
    2. Infer the general mechanism that could facilitate such transformations.
    3. Infer the reagent (in SMILES format) that could enable such transformations.

"""

    elif task_name == "catalyst":
        system_message = "You are a chemical reaction explanation text generator tasked with creating step-by-step reasoning for a catalyst prediction task, where the goal is to infer possible catalyst that enable the production of given product from specific reactant in a chemical reaction."
        if use_step_inst:
            system_message += f""" Specifically, you should generate the explanation to follow following steps:
    1. Compare the reactant and product to identify any functional groups or bonds that have changed.
    2. Infer the general mechanism that could facilitate such transformations.
    3. Infer the catalyst (in SMILES format) that could enable such transformations.

"""

    elif task_name == "solvent":
        system_message = "You are a chemical reaction explanation text generator tasked with creating reasoning for a solvent prediction task, where the goal is to infer possible solvent that enable the production of given product from specific reactant in a chemical reaction."
        if use_step_inst:
            system_message += f""" Specifically, you should generate the explanation to follow following steps:
    1. Compare the reactant and product to identify any functional groups or bonds that have changed.
    2. Infer the general mechanism that could facilitate such transformations.
    3. Infer the solvent (in SMILES format) that could enable such transformations.

"""
    else:
        raise ValueError(f"Invalid task: {task_name}")
    
    system_message += f"""# Information about chemical reactions that users ask about:
"""
    if data_dict.get("precursor"):
        system_message += f"## Precursor (SMILES format)\n{data_dict['precursor']}\n\n"
    if data_dict.get("reactants"):
        system_message += f"## Reactant (SMILES format)\n{data_dict['reactants']}\n\n"
    if data_dict.get("reagents"):
        system_message += f"## Reagent (SMILES format)\n{data_dict['reagents']}\n\n"
    if data_dict.get("catalyst"):
        system_message += f"## Catalyst (SMILES format)\n{data_dict['catalyst']}\n\n"
    if data_dict.get("solvent"):
        system_message += f"## Solvent (SMILES format)\n{data_dict['solvent']}\n\n"
    if data_dict.get("product"):
        system_message += f"## Product (SMILES format)\n{data_dict['product']}\n\n"
    
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
        "temperature": 0.2,
        "max_tokens": 1000
    }
    
    return body_dict






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



with open('CoT_experiments/data/maccskeys/MACCSKeys_descriptions.json', 'r') as f:
    maccskeys_descriptions = json.load(f)
    subs = ["NONE"] + [d['substructure'] for d in maccskeys_descriptions.values()]

with open("CoT_experiments/data/fewshot_example_all.json", "r") as f:
    fewshot_example_all = json.load(f)

mechanism_docs = {}
for file_name in glob("CoT_experiments/data/reaction_docs/docs_chatgpt/*.txt"):
    mechanism_name = os.path.basename(file_name).replace(".txt", "").replace("_", " ")
    with open(file_name, 'r') as f:
        data = f.read()
    mechanism_docs[mechanism_name] = data


task_names = ["forward", "retro", "reagent", "catalyst", "solvent"]
# reasoning_texts_gpt4ostepbystep = {task_name: [[], [], [], []] for task_name in task_names}
# reasoning_texts_llasmolstepbystep = {task_name: [[], [], [], []] for task_name in task_names}
# reasoning_texts_prestostepbystep = {task_name: [[], [], [], []] for task_name in task_names}
# reasoning_texts_manual = {task_name: [[], [], [], []] for task_name in task_names}
# reasoning_texts_syntheticreact = {task_name: [[], [], [], []] for task_name in task_names}
# reasoning_texts_reactexplainer = {task_name: [[], [], [], []] for task_name in task_names}


system_prompt_all = fewshot_example_all["system_prompts"]
user_prompt_all = fewshot_example_all["user_prompts"]
ground_truth_all = fewshot_example_all["ground_truths"]



all_combinations = []
# for use_react_doc in [True, False]:
for use_react_doc in [True]:
    # for use_subs in [True, False]:
    for use_subs in [True]:
        # for use_step_inst in [True, False]:
        for use_step_inst in [True]:
            # for text_len in [0, 1, 2]:
            for text_len in [2]:
                all_combinations.append((use_react_doc, use_subs, use_step_inst, text_len))

for use_react_doc, use_subs, use_step_inst, text_len in all_combinations:
    reasoning_texts = {task_name: [[], [], [], []] for task_name in task_names}
    combination_str = f"{use_react_doc}_{use_subs}_{use_step_inst}_{text_len}"
    # if combination_str in fewshot_example_all:
    #     continue
    for task_name in task_names:
        file_name = f"CoT_experiments/data/presto_reasoning_data/{task_name}/train.json"
        with open(file_name, 'r') as f:
            data = json.load(f)
        for seed in range(2):
            for n_shot in range(5):
                system_prompt = system_prompt_all[task_name][seed][n_shot]
                user_prompt = user_prompt_all[task_name][seed][n_shot].replace("[START_I_SMILES]", "").replace("[END_I_SMILES]", "")
                ground_truth = ground_truth_all[task_name][seed][n_shot]
                d = data[n_shot_indices[task_name][seed][n_shot]]

                assert user_prompt == d['user_prompt'].replace("[START_I_SMILES]", "").replace("[END_I_SMILES]", "")

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

                if task_name == "forward":
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
                elif task_name == "retro":
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
                elif task_name == "reagent":
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
                elif task_name == "catalyst":
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
                elif task_name == "solvent":
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

                body_dict = get_request_body(data_dict, use_react_doc,  use_subs, use_step_inst, text_len, task_name, model="gpt-4o-mini")
                response = client.chat.completions.create(**body_dict)
                response_text = response.choices[0].message.content
                reasoning_texts[task_name][seed].append(response_text)
                print()
                print(f"combination: use_react_doc={use_react_doc}, use_subs={use_subs}, use_step_inst={use_step_inst}, text_len={text_len}")
                print(f"{task_name} {seed} {n_shot}")
                print("="*100)
                print(response_text)
                # print(body_dict["messages"][0]["content"])
                # print("-"*100)
                # print(body_dict["messages"][1]["content"])
                print("="*100)
                print()
    fewshot_example_all[combination_str] = reasoning_texts

    with open("CoT_experiments/data/fewshot_example_all.json", "w") as f:
        json.dump(fewshot_example_all, f, indent=4)



"""
################################ Step by step reasoning text generation ################################
if "reasoning_texts_gpt4ostepbystep" not in fewshot_example_all:
    for task_name in task_names:
        for seed in range(3):
            for n_shot in range(5):
                system_prompt = system_prompt_all[task_name][seed][n_shot]
                user_prompt = user_prompt_all[task_name][seed][n_shot].replace("[START_I_SMILES]", "").replace("[END_I_SMILES]", "")

                body_dict = {
                    "model": "gpt-4o-2024-11-20",
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
                            "content": "Let's think step by step."
                        }
                    ],
                    "temperature": 1.0,
                    "max_tokens": 1000
                }
                response = client.chat.completions.create(**body_dict)
                response_text = response.choices[0].message.content
                reasoning_texts_gpt4ostepbystep[task_name][seed].append(f"Let's think step by step. {response_text}")
                print("="*100)            
                print("reasoning_texts_gpt4ostepbystep")
                print(f"{task_name} {seed} {n_shot} done")
                # print(response_text)
                print()
                print()

    fewshot_example_all["reasoning_texts_gpt4ostepbystep"] = reasoning_texts_gpt4ostepbystep

    with open("CoT_experiments/data/fewshot_example_all.json", "w") as f:
        json.dump(fewshot_example_all, f, indent=4)
#############################################################################################################


##################################### Manual reasoning text generation #####################################
if "reasoning_texts_manual" not in fewshot_example_all:
    for task_name in task_names:
        file_name = f"CoT_experiments/data/presto_reasoning_data/{task_name}/train.json"
        with open(file_name, 'r') as f:
            data = json.load(f)
        for seed in range(3):
            for n_shot in range(5):
                system_prompt = system_prompt_all[task_name][seed][n_shot]
                user_prompt = user_prompt_all[task_name][seed][n_shot].replace("[START_I_SMILES]", "").replace("[END_I_SMILES]", "")
                ground_truth = ground_truth_all[task_name][seed][n_shot]
                d = data[n_shot_indices[task_name][seed][n_shot]]

                assert user_prompt == d['user_prompt'].replace("[START_I_SMILES]", "").replace("[END_I_SMILES]", "")

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

                body_dict = get_manual_request_body(data_dict, task_name)
                response = client.chat.completions.create(**body_dict)
                response_text = response.choices[0].message.content

                body_dict = get_cleaning_text_request_body(response_text, ground_truth, task_name)
                response = client.chat.completions.create(**body_dict)
                response_text = response.choices[0].message.content

                reasoning_texts_manual[task_name][seed].append(response_text)
                print("="*100)
                print("reasoning_texts_manual")
                print(f"{task_name} {seed} {n_shot} done")
                print("_"*100)
                # print(prev_response_text)
                print("_"*100)
                print(response_text)
                print()
                print()

    fewshot_example_all["reasoning_texts_manual"] = reasoning_texts_manual

    with open("CoT_experiments/data/fewshot_example_all.json", "w") as f:
        json.dump(fewshot_example_all, f, indent=4)
#############################################################################################################


################################# ReactExplainer reasoning text generation #################################
if "reasoning_texts_reactexplainer" not in fewshot_example_all:
    pass
#############################################################################################################


################################# SyntheticReact reasoning text generation #################################
if "reasoning_texts_syntheticreact" not in fewshot_example_all:
    for task_name in task_names:
        file_name = f"CoT_experiments/data/presto_reasoning_data/{task_name}/train.json"
        with open(file_name, 'r') as f:
            data = json.load(f)
        for seed in range(3):
            for n_shot in range(5):
                system_prompt = system_prompt_all[task_name][seed][n_shot]
                user_prompt = user_prompt_all[task_name][seed][n_shot].replace("[START_I_SMILES]", "").replace("[END_I_SMILES]", "")
                ground_truth = ground_truth_all[task_name][seed][n_shot]
                d = data[n_shot_indices[task_name][seed][n_shot]]

                assert user_prompt == d['user_prompt'].replace("[START_I_SMILES]", "").replace("[END_I_SMILES]", "")

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

                body_dict = get_syntheticreact_request_body(data_dict, task_name)
                response = client.chat.completions.create(**body_dict)
                response_text = response.choices[0].message.content

                body_dict = get_cleaning_text_request_body(response_text, ground_truth, task_name)
                response = client.chat.completions.create(**body_dict)
                response_text = response.choices[0].message.content

                reasoning_texts_syntheticreact[task_name][seed].append(response_text)
                print("="*100)            
                print("reasoning_texts_syntheticreact")
                print(f"{task_name} {seed} {n_shot} done")
                print("_"*100)
                # print(prev_response_text)
                print("_"*100)
                print(response_text)
                print()
                print()

    fewshot_example_all["reasoning_texts_syntheticreact"] = reasoning_texts_syntheticreact

    with open("CoT_experiments/data/fewshot_example_all.json", "w") as f:
        json.dump(fewshot_example_all, f, indent=4)
#############################################################################################################


##################################### LlaSMol reasoning text generation #####################################
if "reasoning_texts_llasmolstepbystep" not in fewshot_example_all:
    pass
#############################################################################################################


##################################### PRESTO reasoning text generation #####################################
if "reasoning_texts_prestostepbystep" not in fewshot_example_all:
    pass
#############################################################################################################
"""