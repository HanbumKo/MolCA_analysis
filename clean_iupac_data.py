import torch
import pandas as pd


from torch_geometric.data import Data
from ogb.utils import smiles2graph
from tqdm import tqdm
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


def smiles2data(smiles):
    graph = smiles2graph(smiles)
    # x = torch.from_numpy(graph['node_feat'])
    # edge_index = torch.from_numpy(graph['edge_index'], )
    # edge_attr = torch.from_numpy(graph['edge_feat'])
    # data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return graph


valid_data, valid_slices = torch.load("data/PubChem324kV2/valid.pt")
test_data, test_slices = torch.load("data/PubChem324kV2/test.pt")

# import "data/PubChem324kV2/train_iupac_hard.csv"
df = pd.read_csv("data/PubChem324kV2/train_iupac_hard.csv")

valid_rows = []
for i, row in tqdm(df.iterrows(), total=len(df)):
    try:
        smiles2data(row['SMILES'])
        valid_rows.append(i)
    except:
        print(f"Error in row {i} - {row['SMILES']}")
        pass
    # if i == 100:
    #     break

cleaned_df = df.loc[valid_rows]
cleaned_df = cleaned_df[~cleaned_df['SMILES'].isin(valid_data.smiles)]
cleaned_df = cleaned_df[~cleaned_df['SMILES'].isin(test_data.smiles)]
cleaned_df.to_csv("data/PubChem324kV2/train_iupac_clean.csv", index=False)


