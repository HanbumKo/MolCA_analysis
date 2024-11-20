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


class Retrosynthesis(InMemoryDataset):
    def __init__(self, path, text_max_len, prompt=None):
        super(Retrosynthesis, self).__init__()
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
        smiles = data['smiles']
       
        graph = smiles2data(smiles)
        graph.instruction = data['instruction']
        graph.smiles = smiles
        graph.y = data['y']

        smiles_prompt = f"[START_I_SMILES]{smiles}[END_I_SMILES]"
        smiles_prompt = f"{smiles_prompt}\n\nQuestion: {data['instruction']}\n\nAnswer: "
        
        label_smiles = data['y'].split('.')
        all_processed_label_smiles = []
        for label_s in label_smiles:
            label_s_text = split_smiles_with_separator(label_s)
            all_processed_label_smiles.append(label_s_text)
        all_processed_label_smiles = [f"[START_I_SMILES]{label_s}[END_I_SMILES]" for label_s in all_processed_label_smiles]
        label_text = ".".join(all_processed_label_smiles)

        return graph, f"{label_text}\n", smiles_prompt


class USPTORetrosynthesis(InMemoryDataset):
    def __init__(self, path, text_max_len, prompt=None):
        # Load data csv file
        self.data_list = pd.read_csv(path)
        # The csv data has the structure:
        # id,class,reactions
        # US08329716B2,UNK,O=C(OC(=O)C(F)(F)F)C(F)(F)F.Nc1ccc(Oc2ccnc3[nH]ccc23)c(F)c1>>O=C(Nc1ccc(Oc2ccnc3[nH]ccc23)c(F)c1)C(F)(F)F
        # US06051718,UNK,CC(C)(C)OC(=O)OC(=O)OC(C)(C)C.COc1ccc(C(C)CN)cc1OC1Cc2ccccc2C1>>COc1ccc(C(C)CNC(=O)OC(C)(C)C)cc1OC1Cc2ccccc2C1
        
        self.prompt_list = [
            "Based on the given product, provide some plausible reactants that might have been utilized to prepare it.",
            "Which reactants could have been used to generate the given product?",
            "Please suggest potential reactants for the given product.",
            "Please suggest possible reactants for the given product.",
            "What reactants could lead to the production of the following product?",
            "Can you identify some reactants that might result in the given product?",
            "Given the product provided, propose some possible reactants that could have been employed in its formation.",
            "Given the following product, please provide possible reactants.",
            "Please suggest potential reactants used in the synthesis of the provided product.",
            "Provided the product below, propose some possible reactants that could have been used in the reaction.",
        ]
        self.perm = None
    
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
        
        smiles_prompt = f"[START_I_SMILES]{smiles}[END_I_SMILES]"
        smiles_prompt = f"{smiles_prompt}\n\nQuestion: {self.prompt_list[index % len(self.prompt_list)]}\n\nAnswer: "
        
        label_smiles = reactants
        all_processed_label_smiles = []
        for label_s in label_smiles:
            label_s_text = split_smiles_with_separator(label_s)
            all_processed_label_smiles.append(label_s_text)
        all_processed_label_smiles = [f"[START_I_SMILES]{label_s}[END_I_SMILES]" for label_s in all_processed_label_smiles]
        label_text = ".".join(all_processed_label_smiles)

        return graph, f"{label_text}\n", smiles_prompt



if __name__ == '__main__':
    dataset = Retrosynthesis('data/property_prediction/train.pt', 128)
    print(dataset[0])