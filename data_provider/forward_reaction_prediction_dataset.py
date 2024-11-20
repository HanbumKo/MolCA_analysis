import torch
import json
import pandas as pd
from torch_geometric.data import Dataset, InMemoryDataset, Data
import os
import selfies as sf
from rdkit import Chem
from ogb.utils import smiles2graph
from tqdm import tqdm


SPLIT_MARKER = f"SPL{1}T-TH{1}S-Pl3A5E"

def smiles2data(smiles):
    graph = smiles2graph(smiles)
    x = torch.from_numpy(graph['node_feat'])
    edge_index = torch.from_numpy(graph['edge_index'], )
    edge_attr = torch.from_numpy(graph['edge_feat'])
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data


def split_smiles_with_separator(smiles):
    result = "SPL1T-TH1S-Pl3A5E".join(smiles)
    return result


class ForwardReactionPrediction(InMemoryDataset):
    def __init__(self, path, text_max_len, prompt=None):
        super(ForwardReactionPrediction, self).__init__()
        # Load data pt file
        self.data_list = torch.load(path)

        self.prompt = ""
        self.perm = None

    def _selfies_to_smiles(self, selfies):
        # Convert SELFIES to SMILES
        smiles = sf.decoder(selfies)

        # Convert SMILES to RDKit molecule
        mol = Chem.MolFromSmiles(smiles)

        # Ensure the molecule is sanitized and has stereochemistry information
        Chem.SanitizeMol(mol)
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

        # Convert back to SMILES with stereochemistry (isomeric SMILES)
        isomeric_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        return isomeric_smiles

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, index):
        data = self.data_list[index]
        graph_list = []
        for batch_i, smiles in enumerate(data['smiles'].split('.')):
            graph = smiles2data(smiles)
            graph.instruction = data['instruction']
            graph.smiles = smiles
            graph.y = data['y']
            graph_list.append(graph)

        all_smiles = [graph.smiles for graph in graph_list]
        tagged_smiles = [f"[START_I_SMILES]{smile}[END_I_SMILES]" for smile in all_smiles]
        smiles_prompt = ".".join(tagged_smiles)
        smiles_prompt = smiles_prompt + "\n\nQuestion: " + data['instruction'] + "\n\nAnswer: "
        label_text = split_smiles_with_separator(data['y'])

        return graph_list, f"[START_I_SMILES]{label_text}[END_I_SMILES]\n", smiles_prompt


class USPTOForwardReactionPrediction(InMemoryDataset):
    def __init__(self, path, text_max_len, prompt=None):
        super(USPTOForwardReactionPrediction, self).__init__()
        # Load data pt file
        self.data_list = pd.read_csv(path)

        self.prompt_list = [
            "Please suggest a potential product based on the given reactants and reagents.",
            "Please provide a feasible product that could be formed using the given reactants and reagents.",
            "Based on the given reactants and reagents, what product could potentially be produced?",
            "Given the reactants and reagents provided, what is a possible product that can be formed?",
            "Using the provided reactants and reagents, can you propose a likely product?",
            "Based on the given reactants and reagents, suggest a possible product.",
            "With the provided reactants and reagents, propose a potential product.",
            "Given the reactants and reagents below, come up with a possible product.",
            "Given the following reactants and reagents, please provide a possible product.",
            "Using the listed reactants and reagents, offer a plausible product.",
        ]
        self.perm = None

    def _selfies_to_smiles(self, selfies):
        # Convert SELFIES to SMILES
        smiles = sf.decoder(selfies)

        # Convert SMILES to RDKit molecule
        mol = Chem.MolFromSmiles(smiles)

        # Ensure the molecule is sanitized and has stereochemistry information
        Chem.SanitizeMol(mol)
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

        # Convert back to SMILES with stereochemistry (isomeric SMILES)
        isomeric_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        return isomeric_smiles

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, index):
        data = self.data_list.iloc[index]
        reaction = data['reactions']
        reaction = reaction.split('>>')
        reactants = reaction[0]
        products = reaction[1]
        reactants = reactants.split('.')
        products = products.split('.')
        smiles = products[0]
        
        graph = smiles2data(smiles)
        graph.smiles = smiles

        graph_list = []
        for batch_i, smiles in enumerate(reactants):
            graph = smiles2data(smiles)
            graph.smiles = smiles
            graph_list.append(graph)

        all_smiles = [graph.smiles for graph in graph_list]
        tagged_smiles = [f"[START_I_SMILES]{smile}[END_I_SMILES]" for smile in all_smiles]
        smiles_prompt = ".".join(tagged_smiles)
        smiles_prompt = smiles_prompt + "\n\nQuestion: " + self.prompt_list[index % len(self.prompt_list)] + "\n\nAnswer: "
        label_text = split_smiles_with_separator(smiles)

        return graph_list, f"[START_I_SMILES]{label_text}[END_I_SMILES]\n", smiles_prompt


if __name__ == '__main__':
    dataset = ForwardReactionPrediction('data/property_prediction/train.pt', 128)
    print(dataset[0])