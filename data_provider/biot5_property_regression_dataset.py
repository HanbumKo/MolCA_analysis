import torch
import json
from torch_geometric.data import Dataset, InMemoryDataset, Data
import os
import selfies as sf
from rdkit import Chem
from ogb.utils import smiles2graph
from tqdm import tqdm
from glob import glob


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


class BioT5PropertyRegression(InMemoryDataset):
    def __init__(self, data_type):
        super(BioT5PropertyRegression, self).__init__()
        # Load data pt file
        # self.data_list = torch.load(path)

        files = [
            # Regression tasks
            (glob(f"data/biot5_plus_data/tasks_plus/*_property_prediction_molinst_mol_homo_{data_type}.json")[0], "homo_reg"),
            (glob(f"data/biot5_plus_data/tasks_plus/*_property_prediction_molinst_mol_lumo_{data_type}.json")[0], "lumo_reg",) ,
            (glob(f"data/biot5_plus_data/tasks_plus/*_property_prediction_molinst_mol_gap_{data_type}.json")[0] , "gap_reg"),
        ]

        self.data_type = data_type
        self.prompt = '[START_I_SMILES]{}[END_I_SMILES]'
        # self.task = 'property_prediction'
        # file_name = glob(f"data/biot5_plus_data/tasks_plus/*_property_prediction_molinst_mol_{data_type}.json")[0]
        self.data_list = []
        for file_name, task in tqdm(files, desc="Loading data", total=len(files)):
            with open(file_name, 'r') as f:
                dataset = json.load(f)["Instances"]
                for i, d in enumerate(dataset):
                    data = {
                        "instruction": d['instruction'],
                        'iupac': d['input'].split('<boi>')[1].split('<eoi>')[0],
                        "smiles": d['input'].split("[START_I_SMILES]")[1].split("[END_I_SMILES]")[0],
                        "y": d['output'][0],
                        "task": task
                    }
                    self.data_list.append(data)
                    # if i == 50:
                    #     break
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
        iupac = data['iupac']
        graph.smiles = smiles
        task = data['task']
        graph.y = data['y']
        

        if self.prompt.find('{}') >= 0:
            smiles_prompt = self.prompt.format(smiles[:128])
        else:
            smiles_prompt = self.prompt

        smiles_prompt = smiles_prompt + f"\n\nThe molecule's IUPAC name is {iupac}.\n\nQuestion: " + graph.instruction + "\n\nAnswer: "
        label_text = split_float_with_separator(graph.y)

        return graph, label_text + '\n', smiles_prompt, task


if __name__ == '__main__':
    dataset = BioT5PropertyRegression('data/property_prediction/train.pt', 128)
    print(dataset[0])