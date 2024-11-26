import ftplib
import os

# PubChem FTP 서버 설정
FTP_HOST = "ftp.ncbi.nlm.nih.gov"
FTP_DIR = "/pubchem/Compound/CURRENT-Full/SDF/"  # Compound SDF 데이터 경로
LOCAL_DIR = "data/pubchem_dump"  # 로컬에 저장할 폴더명

# 로컬 디렉토리가 없으면 생성
if not os.path.exists(LOCAL_DIR):
    os.makedirs(LOCAL_DIR)

# FTP 서버에서 파일을 다운로드하는 함수
def download_file(ftp, filename):
    local_filepath = os.path.join(LOCAL_DIR, filename)
    with open(local_filepath, "wb") as f:
        ftp.retrbinary(f"RETR {filename}", f.write)
    print(f"{filename} 다운로드 완료")

# FTP 서버에 연결 및 데이터 다운로드
with ftplib.FTP(FTP_HOST) as ftp:
    ftp.login()  # 로그인 (익명 접속)
    ftp.cwd(FTP_DIR)  # Compound SDF 데이터 폴더로 이동
    filenames = ftp.nlst()  # 폴더 내 파일 목록 가져오기

    for filename in filenames:
        # 파일이 이미 존재하면 다운로드하지 않음
        local_filepath = os.path.join(LOCAL_DIR, filename)
        if os.path.exists(local_filepath):
            print(f"{filename} 이미 존재함")
        else:
            download_file(ftp, filename)
    print("모든 파일 다운로드 완료")