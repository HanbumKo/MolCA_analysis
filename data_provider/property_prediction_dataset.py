import torch
import json
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


def split_float_with_separator(number):
    number_str = str(number)
    result = "SPL1T-TH1S-Pl3A5E".join(number_str)
    return result


class PropertyPrediction(InMemoryDataset):
    def __init__(self, path, text_max_len, prompt=None):
        super(PropertyPrediction, self).__init__()
        # Load data pt file
        self.data_list = torch.load(path)

        if not prompt:
            self.prompt = '[START_I_SMILES]{}[END_I_SMILES]'
        else:
            self.prompt = prompt
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
        

        if self.prompt.find('{}') >= 0:
            smiles_prompt = self.prompt.format(smiles[:128])
        else:
            smiles_prompt = self.prompt

        smiles_prompt = smiles_prompt + "\n\nQuestion: " + graph.instruction + "\n\nAnswer: "
        label_text = split_float_with_separator(graph.y)

        return graph, label_text + '\n', smiles_prompt


if __name__ == '__main__':
    dataset = PropertyPrediction('data/property_prediction/train.pt', 128)
    print(dataset[0])