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


class BioT5ReactionCoT(InMemoryDataset):
    def __init__(self, data_type):
        super(BioT5ReactionCoT, self).__init__()
        # Load data pt file
        # self.data_list = torch.load(path)

        files = [
            # Regression tasks
            (glob(f"CoT_experiments/data/biot5_plus_reasoning_data/*_forward_reaction_prediction_molinst_mol_{data_type}.json")[0], "forward"),
            (glob(f"CoT_experiments/data/biot5_plus_reasoning_data/*_retrosynthesis_molinst_mol_{data_type}.json")[0], "retro",) ,
            (glob(f"CoT_experiments/data/biot5_plus_reasoning_data/*_reagent_prediction_molinst_mol_{data_type}.json")[0] , "reagent"),
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
                        "input_smiles": d['input'].split("[START_I_SMILES]")[1].split("[END_I_SMILES]")[0],
                        "output_smiles": d['output'][0].split("[START_I_SMILES]")[1].split("[END_I_SMILES]")[0],
                        "reasoning": d['reasoning'],
                        "task": task,
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
        instruction = data['instruction']
        input_smiles = data['input_smiles']
        output_smiles = data['output_smiles']
        reasoning_text = data['reasoning']
        task = data['task']
        if task == "reagent":
            left_smiles = [smiles for smiles in input_smiles.split('>>')[0].split('.')]
            right_smiles = [smiles for smiles in input_smiles.split('>>')[1].split('.')]
            left_tagged_smiles = [f"[START_I_SMILES]{smile}[END_I_SMILES]" for smile in left_smiles]
            right_tagged_smiles = [f"[START_I_SMILES]{smile}[END_I_SMILES]" for smile in right_smiles]
            left_smiles_prompt = ".".join(left_tagged_smiles)
            right_smiles_prompt = ".".join(right_tagged_smiles)
            smiles_prompt = f"{left_smiles_prompt}>>{right_smiles_prompt}"
            smiles_prompt = f"Question: {instruction}\n{smiles_prompt}\n<work>\n{reasoning_text}\n</work>\n\nAnswer: "
            all_input_smiles = left_smiles + right_smiles
        else:
            all_input_smiles = input_smiles.split('.')
            tagged_smiles = [f"[START_I_SMILES]{smile}[END_I_SMILES]" for smile in all_input_smiles]
            smiles_prompt = ".".join(tagged_smiles)
            smiles_prompt = f"Question: {instruction}\n{smiles_prompt}\n<work>\n{reasoning_text}\n</work>\n\nAnswer: "

        graph_list = []
        for i, smiles in enumerate(all_input_smiles):
            graph = smiles2data(smiles)
            graph_list.append(graph)

        label_text = split_float_with_separator(output_smiles)


        return graph_list, f"[START_I_SMILES]{label_text}[END_I_SMILES]\n", smiles_prompt, task


if __name__ == '__main__':
    dataset = BioT5ReactionCoT(data_type="test")
    print(dataset[0])