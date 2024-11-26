import deepchem as dc

# USPTO 데이터셋 로드
tasks, datasets, transformers = dc.molnet.load_uspto(
    featurizer='Raw',
    split='scaffold',
    data_dir='data/USPTO',
    subset="50K",
    reload=True
)

# 데이터셋 언패킹
train_dataset, valid_dataset, test_dataset = datasets

train_raw = train_dataset.ids

# 훈련 데이터 확인
print("훈련 데이터 특징 행렬 크기:", train_dataset.X.shape)
print("훈련 데이터 레이블 벡터 크기:", train_dataset.y.shape)

# 변환기 확인
print("적용된 변환기들:", transformers)
