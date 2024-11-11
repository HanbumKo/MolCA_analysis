import os
import pandas as pd
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


def split_smiles_with_separator(smiles):
    result = "SPL1T-TH1S-Pl3A5E".join(smiles)
    return result


class PubChemPlusPlus(InMemoryDataset):
    def __init__(self, path, text_max_len, prompt=None):
        super(PubChemPlusPlus, self).__init__()
        self.data_list = []
        # Read all csv files in the path
        for csv_name in os.listdir(path):
            file_path = os.path.join(path, csv_name)
            df = pd.read_csv(file_path)
            self.data_list.extend(df.to_dict('records'))
            print(f"Loaded {len(df)} records from {file_path}")

        if not prompt:
            self.prompt = 'The SMILES of this molecule is [START_I_SMILES]{}[END_I_SMILES]. '
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
        graph = smiles2data(data['SMILES'])
        graph.text = data['description']
        graph.smiles = data['SMILES']
        if self.prompt.find('{}') >= 0:
            smiles_prompt = self.prompt.format(graph.smiles[:128])
        else:
            smiles_prompt = self.prompt

        return graph, str(graph.text) + "\n", smiles_prompt # Need to clean up the text data


if __name__ == '__main__':
    dataset = PubChemPlusPlus('data/PubChem++', 128)
    print(dataset[0])