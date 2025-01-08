import os
import json
import copy
import random
import pandas as pd
import numpy as np
import random

from openai import OpenAI
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from rxnfp.transformer_fingerprints import (
    RXNBERTFingerprintGenerator, get_default_model_and_tokenizer, generate_fingerprints
)
from rdkit import Chem
from rdkit.Chem import AllChem
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

def calcualte_accuracy(preds, expected):
    """
    """
    return sum([1 for p, e in zip(preds, expected) if p == e]) / len(expected)

def remove_atom_mapping(atommaped_reaction):
    """
    """
    if ">>" in atommaped_reaction:
        mol_list = []
        reactants, products = atommaped_reaction.split(">>")
        for mol in [reactants, products]:
            mol = Chem.MolFromSmiles(mol)
            if mol is None:
                raise ValueError("유효하지 않은 SMILES 문자열입니다.")
            for atom in mol.GetAtoms():
                atom.SetAtomMapNum(0)
            mol = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
            mol_list.append(mol)
        return f"{mol_list[0]}>>{mol_list[1]}"
    else:
        mol_list = []
        reactants, reagents, products = atommaped_reaction.split(">")
        precursors = reactants + "." + reagents
        for mol in [precursors, products]:
            mol = Chem.MolFromSmiles(mol)
            if mol is None:
                raise ValueError("유효하지 않은 SMILES 문자열입니다.")
            for atom in mol.GetAtoms():
                atom.SetAtomMapNum(0)
            mol = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
            mol_list.append(mol)
        return f"{mol_list[0]}>>{mol_list[1]}"


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

with open('CoT_experiments/data/rxnfp/rxnclass2id.json', 'r') as f:
    rxnclass2id = json.load(f)

with open('CoT_experiments/data/rxnfp/rxnclass2name.json', 'r') as f:
    rxnclass2name = json.load(f)
all_classes =sorted(rxnclass2id.keys())



df = pd.read_csv('CoT_experiments/data/rxnfp/schneider50k.tsv', sep='\t')
df['class_id'] = [rxnclass2id[c] for c in df.rxn_class]
df['class_name'] = [rxnclass2name[c] for c in df.rxn_class]
# train_df = df[df.split=='train']
# test_df = df[df.split=='test']
train_df = df
test_df = df
# df.head()

for model_name in ["bert_ft_10k_25s", "bert_pretrained", "bert_ft"]: # "bert_ft" is the best model
    model, tokenizer = get_default_model_and_tokenizer(model_name)
    rxnfp_generator = RXNBERTFingerprintGenerator(model, tokenizer)

    train_fingerprints = generate_fingerprints(train_df.rxn.values.tolist(), rxnfp_generator, batch_size=8)
    test_fingerprints = generate_fingerprints(test_df.rxn.values.tolist(), rxnfp_generator, batch_size=8)


    lr_cls =  LogisticRegression(max_iter=5000)
    lr_classifier_trained = lr_cls.fit(train_fingerprints, train_df.class_id.values.tolist())


    print(f"=== Model: {model_name} ===")
    preds = lr_classifier_trained.predict(train_fingerprints)
    predicted = [all_classes[x] for x in preds]
    expected = [all_classes[x] for x in train_df.class_id.values.tolist()]
    accuracy = calcualte_accuracy(predicted, expected)
    print(f"Train accuracy: {accuracy}")

    preds = lr_classifier_trained.predict(test_fingerprints)
    predicted = [all_classes[x] for x in preds]
    expected = [all_classes[x] for x in test_df.class_id.values.tolist()]
    accuracy = calcualte_accuracy(predicted, expected)
    print(f"Test accuracy: {accuracy}")
    print()


# Train/Test split accuracy
# === Model: bert_ft_10k_25s ===
# Train accuracy: 1.0
# Test accuracy: 0.984925

# === Model: bert_pretrained ===
# Train accuracy: 0.9909
# Test accuracy: 0.903425

# === Model: bert_ft ===
# Train accuracy: 0.9998
# Test accuracy: 0.994375


# All train accuracy
# === Model: bert_ft_10k_25s ===
# Train accuracy: 0.99784

# === Model: bert_pretrained ===
# Train accuracy: 0.97424

# === Model: bert_ft ===
# Train accuracy: 0.99946
