import json
import copy
import random
import pandas as pd
import numpy as np
from sklearn import metrics
import random
from sklearn.linear_model import LogisticRegression
from rxnfp.transformer_fingerprints import (
    RXNBERTFingerprintGenerator, get_default_model_and_tokenizer, generate_fingerprints
)
from rdkit import Chem
from rdkit.Chem import AllChem

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
                raise ValueError("Invalid SMILES string.")
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
                raise ValueError("Invalid SMILES string.")
            for atom in mol.GetAtoms():
                atom.SetAtomMapNum(0)
            mol = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
            mol_list.append(mol)
        return f"{mol_list[0]}>>{mol_list[1]}"



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
# train_df = df
# test_df = df

# open "CoT_experiments/data/rxnfp/fps_ft.npz" with np.load
all_fingerprints = np.load('CoT_experiments/data/rxnfp/fps_ft.npz')['fps']

lr_cls =  LogisticRegression(max_iter=100)
lr_classifier_trained = lr_cls.fit(all_fingerprints, df.class_id.values.tolist())


preds = lr_classifier_trained.predict(all_fingerprints)
predicted = [all_classes[x] for x in preds]
expected = [all_classes[x] for x in df.class_id.values.tolist()]
accuracy = calcualte_accuracy(predicted, expected)
print(f"Train accuracy: {accuracy}")

