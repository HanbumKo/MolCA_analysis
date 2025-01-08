import re
import torch
import json
import random
from torch_geometric.data import Dataset, InMemoryDataset, Data
import os
import selfies as sf
from rdkit import Chem
from ogb.utils import smiles2graph
from tqdm import tqdm
from glob import glob


CUSTOM_SEQ_RE = re.compile(r"(\[START_(DNA|SMILES|I_SMILES|AMINO)])(.*?)(\[END_\2])")
SPLIT_MARKER = f"SPL{1}T-TH{1}S-Pl3A5E"

def smiles2data(smiles):
    graph = smiles2graph(smiles)
    x = torch.from_numpy(graph['node_feat'])
    edge_index = torch.from_numpy(graph['edge_index'], )
    edge_attr = torch.from_numpy(graph['edge_feat'])
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data


def _insert_split_marker(m: re.Match):
    """
    Applies split marker based on a regex match of special tokens such as
    [START_DNA].

    Parameters
    ----------
    n : str
        Input text to split

    Returns
    ----------
    str - the text with the split token added
    """
    start_token, _, sequence, end_token = m.groups()
    sequence = re.sub(r"(.)", fr"{SPLIT_MARKER}\1", sequence, flags=re.DOTALL)
    return f"{start_token}{sequence}{SPLIT_MARKER}{end_token}"


def escape_custom_split_sequence(text):
    """
    Applies custom splitting to the text for GALILEO's tokenization

    Parameters
    ----------
    text : str
        Input text to split

    Returns
    ----------
    str - the text with the split token added
    """
    return CUSTOM_SEQ_RE.sub(_insert_split_marker, text)


class PrestoReaction(InMemoryDataset):
    def __init__(self, data_type, shuffle=False, n_test_samples=99999999):
        super(PrestoReaction, self).__init__()
        # Load data pt file
        # self.data_list = torch.load(path)

        files = [
            # Regression tasks
            (f"data/presto_data/forward/{data_type}.json", "forward"),
            (f"data/presto_data/retro/{data_type}.json", "retro"),
            (f"data/presto_data/reagent/{data_type}.json", "reagent"),
            (f"data/presto_data/catalyst/{data_type}.json", "catalyst"),
            (f"data/presto_data/solvent/{data_type}.json", "solvent"),
        ]

        self.data_type = data_type
        self.data_list = []
        for file_name, task in tqdm(files, desc="Loading data", total=len(files)):
            with open(file_name, 'r') as f:
                dataset = json.load(f)
                for i, d in enumerate(dataset):
                    data = {
                        **d,
                        "task": task,
                    }
                    self.data_list.append(data)
                    if i == n_test_samples-1:
                        break
        if shuffle:
            random.shuffle(self.data_list)
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
        system_prompt = data['system_prompt']
        user_prompt = data['user_prompt'].replace(" .", ".").replace("[END_I_SMILES].[START_I_SMILES]", ".")
        input_prompt = f"{system_prompt}\n\nQuestion: {user_prompt}\n\n"
        task = data['task']

        if task == "forward":
            output_prompt = f"Answer: [START_I_SMILES]{data['product']}[END_I_SMILES]</s>"
        elif task == "retro":
            output_prompt = f"Answer: [START_I_SMILES]{data['reactants']}[END_I_SMILES]</s>"
        elif task == "reagent":
            output_prompt = f"Answer: [START_I_SMILES]{data['reagents']}[END_I_SMILES]</s>"
        elif task == "catalyst":
            output_prompt = f"Answer: [START_I_SMILES]{data['catalyst']}[END_I_SMILES]</s>"
        elif task == "solvent":
            output_prompt = f"Answer: [START_I_SMILES]{data['solvent']}[END_I_SMILES]</s>"
        else:
            raise ValueError(f"Unknown task: {task}")

        graph_list = []
        # for i, smiles in enumerate(all_input_smiles):
        #     graph = smiles2data(smiles)
        #     graph_list.append(graph)

        output_prompt = escape_custom_split_sequence(output_prompt)


        return graph_list, output_prompt, input_prompt, task


if __name__ == '__main__':
    dataset = PrestoReaction(data_type="test")
    print(dataset[0])