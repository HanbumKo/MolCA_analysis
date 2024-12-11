import torch
import os
import pandas as pd
import selfies as sf
from glob import glob

from torch_geometric.data import Dataset, InMemoryDataset, Data
from rdkit import Chem
from ogb.utils import smiles2graph

def smiles2data(smiles):
    graph = smiles2graph(smiles)
    x = torch.from_numpy(graph['node_feat'])
    edge_index = torch.from_numpy(graph['edge_index'], )
    edge_attr = torch.from_numpy(graph['edge_feat'])
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data


class MoleculeCaption(Dataset):
    def __init__(self, root, text_max_len, prompt=None, filtered_cid_path=None):
        super(MoleculeCaption, self).__init__(root)
        self.root = root
        self.text_max_len = text_max_len
        self.graph_name_list = os.listdir(root+'graph/')
        self.graph_name_list.sort()
        self.text_name_list = os.listdir(root+'text/')
        self.text_name_list.sort()
        self.smiles_name_list = os.listdir(root+'smiles/')
        self.smiles_name_list.sort()
        self.tokenizer = None
        
        if not prompt:
            self.prompt = 'The SMILES of this molecule is [START_I_SMILES]{}[END_I_SMILES]. '
        else:
            self.prompt = prompt

        if filtered_cid_path is not None:
            print('before filtering', len(self.graph_name_list), len(self.text_name_list), len(self.smiles_name_list))
            with open(filtered_cid_path, 'r') as f:
                self.filtered_cid_set = [line.strip() for line in f.readlines()]
                self.filtered_cid_set = set(self.filtered_cid_set)
            filtered_graph_name_list = []
            for g in self.graph_name_list:
                cid = g.split('_')[1][:-3]
                if cid in self.filtered_cid_set:
                    filtered_graph_name_list.append(g)
            self.graph_name_list = filtered_graph_name_list
            filtered_text_name_list = []
            for t in self.text_name_list:
                cid = t.split('_')[1][:-4]
                if cid in self.filtered_cid_set:
                    filtered_text_name_list.append(t)
            self.text_name_list = filtered_text_name_list
            filtered_smiles_name_list = []
            for s in self.smiles_name_list:
                cid = s.split('_')[1][:-4]
                if cid in self.filtered_cid_set:
                    filtered_smiles_name_list.append(s)
            self.smiles_name_list = filtered_smiles_name_list
            print('after filtering', len(self.graph_name_list), len(self.text_name_list), len(self.smiles_name_list))

    def get(self, index):
        return self.__getitem__(index)

    def len(self):
        return len(self)

    def __len__(self):
        return len(self.graph_name_list)

    def __getitem__(self, index):
        graph_name, text_name = self.graph_name_list[index], self.text_name_list[index]
        smiles_name = self.smiles_name_list[index]

        # load and process graph
        graph_path = os.path.join(self.root, 'graph', graph_name)
        data_graph = torch.load(graph_path)
        # load and process text
        text_path = os.path.join(self.root, 'text', text_name)
        
        text_list = []
        count = 0
        for line in open(text_path, 'r', encoding='utf-8'):
            count += 1
            text_list.append(line.strip('\n'))
            if count > 100:
                break
        text = ' '.join(text_list) + '\n'

        # load and process smiles
        smiles_path = os.path.join(self.root, 'smiles', smiles_name)
        with open(smiles_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            assert len(lines) == 1
            smiles = lines[0].strip()

        if self.prompt.find('{}') >= 0:
            smiles_prompt = self.prompt.format(smiles[:128])
        else:
            smiles_prompt = self.prompt
        return data_graph, text, smiles_prompt
    
    def tokenizer_text(self, text):
        sentence_token = self.tokenizer(text=text,
                                        truncation=True,
                                        padding='max_length',
                                        add_special_tokens=True,
                                        max_length=self.text_max_len,
                                        return_tensors='pt',
                                        return_attention_mask=True)
        return sentence_token


class MoleculeCaptionV2(InMemoryDataset):
    def __init__(self, path, text_max_len, prompt=None):
        super(MoleculeCaptionV2, self).__init__()
        self.data, self.slices = torch.load(path)

        self.path = path
        self.text_max_len = text_max_len
        
        if not prompt:
            self.prompt = 'The SMILES of this molecule is [START_I_SMILES]{}[END_I_SMILES]. '
        else:
            self.prompt = prompt
        self.perm = None

    def __getitem__(self, index):
        if self.perm is not None:
            index = self.perm[index]
        data = self.get(index)
        smiles = data.smiles
        assert len(smiles.split('\n')) == 1

        if self.prompt.find('{}') >= 0:
            smiles_prompt = self.prompt.format(smiles[:128])
        else:
            smiles_prompt = self.prompt
        text = data.text.split('\n')[:100]
        text = ' '.join(text) + '\n'
        return data, text, smiles_prompt
    
    def shuffle(self):
        self.perm = torch.randperm(len(self)).tolist()
        return self


class MolCapExtended(InMemoryDataset):
    def __init__(self, path, text_max_len, prompt=None):
        super(MolCapExtended, self).__init__()
        self.data_list = []
        # Read all csv files in the path
        # for file_path in glob(path + '*.csv'):
        # for csv_name in os.listdir(path):
        #     file_path = os.path.join(path, csv_name)
        df = pd.read_csv(path)
        self.data_list.extend(df.to_dict('records'))
        # print(f"Loaded {len(df)} records from {file_path}")

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
    dataset = MoleculeCaptionV2('./data/PubChem324kV2/pretrain.pt', 128)
    print(dataset[0])