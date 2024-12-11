import pandas as pd
import torch
import glob
import random
from rdkit import Chem
from tqdm import tqdm

def canonicalize(smiles):
    """
    SMILES 문자열을 정규화하는 함수.
    
    :param smiles: SMILES 문자열
    :return: 정규화된 SMILES 문자열
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
        # raise ValueError("유효하지 않은 SMILES 문자열입니다.")
    return Chem.MolToSmiles(mol, canonical=True)

def are_smiles_equivalent(smiles1, smiles2):
    """
    두 SMILES 문자열이 같은 분자를 나타내는지 확인하는 함수.
    
    :param smiles1: 첫 번째 SMILES 문자열
    :param smiles2: 두 번째 SMILES 문자열
    :return: 두 SMILES가 같은 분자를 나타내면 True, 그렇지 않으면 False
    """
    try:
        # SMILES 문자열을 RDKit 분자 객체로 변환
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)

        # 유효한 SMILES인지 확인
        if mol1 is None or mol2 is None:
            return False

        # Canonical SMILES를 생성하여 비교
        canonical_smiles1 = Chem.MolToSmiles(mol1, canonical=True)
        canonical_smiles2 = Chem.MolToSmiles(mol2, canonical=True)

        return canonical_smiles1 == canonical_smiles2

    except Exception as e:
        print(f"오류 발생: {e}")
        return False

# Load "data/ChEBI-20_data/train.txt"
# This consists of "CID	SMILES	description" columns
try:
    train_df = pd.read_csv("data/ChEBI-20_data/train.txt", sep="\t")
    CheBI_smiles = train_df["SMILES"].tolist()
    data, slices = torch.load("data/PubChem324kV2/train.pt")
    PubChem_smiles = data.smiles
except Exception as e:
    print(f"Error reading train.txt: {e}")
    raise e

chebi_pubchem_smiels_raw = CheBI_smiles + PubChem_smiles
chebi_pubchem_smiels = set([canonicalize(smiles) for smiles in tqdm(chebi_pubchem_smiels_raw, desc="Canonicalizing SMILES", total=len(chebi_pubchem_smiels_raw))])




# CSV 파일이 있는 디렉토리 경로
csv_folder_path = "data"  # CSV 파일들이 저장된 폴더 경로를 입력하세요.
exclude_file_path = "5M_samples.csv"  # 제외할 행이 포함된 파일 경로
output_file_path = "OOD_1K_samples.csv"  # 결과 파일 이름

# 제외할 데이터(A.csv)를 읽어오기
try:
    exclude_df = pd.read_csv(exclude_file_path)
except Exception as e:
    print(f"Error reading {exclude_file_path}: {e}")
    exclude_df = pd.DataFrame()

# CSV 파일 불러오기
csv_files = glob.glob(f"{csv_folder_path}/*.csv")

# 모든 데이터를 저장할 리스트
all_data = []

# 각 파일의 데이터를 읽어 리스트에 추가
for file in tqdm(csv_files, desc="Reading CSV files", total=len(csv_files)):
    # A.csv 파일 자체는 스킵
    if file == exclude_file_path:
        continue

    try:
        df = pd.read_csv(file)
        # "description" 컬럼에서 "Based on"으로 시작하는 행 제외
        if "description" in df.columns:
            df = df[~df["description"].str.startswith("Based on", na=False)]
        all_data.append(df)
    except Exception as e:
        print(f"Error reading {file}: {e}")

# 모든 데이터를 하나의 DataFrame으로 합치기
combined_data = pd.concat(all_data, ignore_index=True)


# A.csv 데이터와 중복된 행 제외
if not exclude_df.empty:
    combined_data = combined_data.merge(exclude_df, how="outer", indicator=True)
    combined_data = combined_data[combined_data["_merge"] == "left_only"].drop(columns=["_merge"])

# 데이터가 5M개 이상인지 확인
if len(combined_data) < 5000000:
    print("데이터가 5M개보다 적습니다. 데이터를 더 추가하세요.")
else:
    # 랜덤하게 5M개의 행 추출
    random_rows = combined_data.sample(n=1000, random_state=random.randint(1, 10000))

    # 결과를 CSV로 저장
    random_rows.to_csv(output_file_path, index=False)
    print(f"랜덤한 5M개의 행을 {output_file_path}에 저장했습니다.")
