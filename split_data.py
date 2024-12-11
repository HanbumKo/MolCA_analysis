import pandas as pd

from tqdm import tqdm
from ogb.utils import smiles2graph

# 파일 경로
file_path = 'data/PubChem324kV2_Extended/5M_samples.csv'

# CSV 파일 불러오기
df = pd.read_csv(file_path)


n_data = len(df)

data_good = []
for i, data in tqdm(df.iterrows(), total=n_data):
    try:
        smiles2graph(data['SMILES'])
        data_good.append(data)
    except:
        print(data)

df = pd.DataFrame(data_good)
n_data = len(df)

# 데이터 분할
pretrain_df = df.iloc[:n_data-2000]
train_df = df.iloc[n_data-3000:n_data-2000]
valid_df = df.iloc[n_data-2000:n_data-1000]
test_df = df.iloc[n_data-1000:]

# 분할된 데이터 저장
pretrain_path = 'data/PubChem324kV2_Extended/pretrain.csv'
train_path = 'data/PubChem324kV2_Extended/train.csv'
valid_path = 'data/PubChem324kV2_Extended/valid.csv'
test_path = 'data/PubChem324kV2_Extended/test.csv'

pretrain_df.to_csv(pretrain_path, index=False)
train_df.to_csv(train_path, index=False)
valid_df.to_csv(valid_path, index=False)
test_df.to_csv(test_path, index=False)

pretrain_path, train_path, valid_path, test_path
