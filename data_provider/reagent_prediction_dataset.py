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


class ReagentPrediction(InMemoryDataset):
    def __init__(self, path, text_max_len, prompt=None):
        super(ReagentPrediction, self).__init__()
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
        left_smiles = [smiles for smiles in data['smiles'].split('>>')[0].split('.')]
        right_smiles = [smiles for smiles in data['smiles'].split('>>')[1].split('.')]
        all_smiles = left_smiles + right_smiles
        graph_list = []
        for batch_i, smiles in enumerate(all_smiles):
            graph = smiles2data(smiles)
            graph.instruction = data['instruction']
            graph.smiles = smiles
            graph.y = data['y']
            graph_list.append(graph)
 
        # all_smiles = [graph.smiles for graph in graph_list]
        # tagged_smiles = [f"[START_I_SMILES]{smile}[END_I_SMILES]" for smile in all_smiles]
        left_tagged_smiles = [f"[START_I_SMILES]{smile}[END_I_SMILES]" for smile in left_smiles]
        right_tagged_smiles = [f"[START_I_SMILES]{smile}[END_I_SMILES]" for smile in right_smiles]
        left_smiles_prompt = ".".join(left_tagged_smiles)
        right_smiles_prompt = ".".join(right_tagged_smiles)
        smiles_prompt = f"{left_smiles_prompt}>>{right_smiles_prompt}"
        smiles_prompt = f"{smiles_prompt}\n\nQuestion: {data['instruction']}\n\nAnswer: "
        
        label_smiles = data['y'].split('.')
        all_processed_label_smiles = []
        for label_s in label_smiles:
            label_s_text = split_smiles_with_separator(label_s)
            all_processed_label_smiles.append(label_s_text)
        all_processed_label_smiles = [f"[START_I_SMILES]{label_s}[END_I_SMILES]" for label_s in all_processed_label_smiles]
        label_text = ".".join(all_processed_label_smiles)

        return graph_list, f"{label_text}\n", smiles_prompt


if __name__ == '__main__':
    dataset = ReagentPrediction('data/property_prediction/train.pt', 128)
    print(dataset[0])