import torch
from torch_geometric.data import Dataset, InMemoryDataset
import os
import re


def read_iupac(path):
    regex = re.compile('\[Compound\((\d+)\)\]')
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    cid2iupac = {}
    for line in lines:
        smiles, cid, iupac = line.split('\t')
        match = regex.match(cid)
        if match:
            cid = match.group(1)
            cid2iupac[cid] = iupac
    return cid2iupac


class IUPACDataset(InMemoryDataset):
    def __init__(self, path, text_max_len, prompt=None):
        super(IUPACDataset, self).__init__()
        self.data, self.slices = torch.load(path)

        self.path = path
        self.text_max_len = text_max_len
        self.tokenizer = None

        if not prompt:
            self.prompt = "[START_I_SMILES]{}[END_I_SMILES]The molecule's IUPAC name is "
        else:
            self.prompt = prompt
        self.perm = None

    def __getitem__(self, index):
        if self.perm is not None:
            index = self.perm[index]
        data = self.get(index)
        smiles = data.smiles
        assert len(smiles.split('\n')) == 1

        iupac = data.iupac
        
        if self.prompt.find('{}') >= 0:
            smiles_prompt = self.prompt.format(smiles[:128])
        else:
            smiles_prompt = self.prompt

        return data, iupac + '\n', smiles_prompt

    def shuffle(self):
        self.perm = torch.randperm(len(self)).tolist()
        return self


if __name__ == '__main__':
    dataset = IUPACDataset('../data/PubChemDataset_v4/test/', 128, )
    print(dataset[0], len(dataset))
    dataset = IUPACDataset('../data/PubChemDataset_v4/train/', 128, )
    print(dataset[0], len(dataset))
    dataset = IUPACDataset('../data/PubChemDataset_v4/valid/', 128, )
    print(dataset[0], len(dataset))