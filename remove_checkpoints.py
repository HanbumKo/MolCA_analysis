import pandas as pd
import os
import glob
import shutil


# ("ft_chebi20_from_noft", "meteor_score_val"),
# ("ft_property_from_noft", "mae_val"),
# ("ft_retrosynthesis_from_nostage2", "num_t1_exact_match_val"),


file_name = "ftstm_retrosynthesis_from_noft"

if "_property_" in file_name:
    metric_name = "mae_val"
elif "_retrosynthesis_" in file_name or "_forward_" in file_name or "_reagent_" in file_name:
    metric_name = "num_t1_exact_match_val"
elif "_chebi20_" in file_name or "_pubchem324k_" in file_name or "_iupac_" in file_name:
    metric_name = "meteor_score_val"
else:
    raise ValueError(f"Unknown file_name: {file_name}")
    

# Load metrics.csv
metrics_path = f"all_checkpoints/{file_name}/lightning_logs/version_0/metrics.csv"
metrics_df = pd.read_csv(metrics_path)

if metric_name != "mae_val":
    # Find row that has maximum metric_name value
    best_epoch = metrics_df.loc[metrics_df[metric_name].idxmax(), 'epoch']
else:
    # Find row that has minimum metric_name value
    best_epoch = metrics_df.loc[metrics_df[metric_name].idxmin(), 'epoch']


# best_epoch에 해당하는 체크포인트 파일 경로
best_ckpt_path = f"all_checkpoints/{file_name}/epoch={int(best_epoch):02d}.ckpt"
best_ckpt_dest = f"all_checkpoints/{file_name}/best.ckpt"

# best.ckpt로 파일 복사
print(f"Copying {best_ckpt_path} to {best_ckpt_dest}")
shutil.copyfile(best_ckpt_path, best_ckpt_dest)

# 나머지 체크포인트 파일 삭제
for ckpt_file in glob.glob(f"all_checkpoints/{file_name}/epoch=*.ckpt"):
    print(f"Removing {ckpt_file}")
    os.remove(ckpt_file)

print(f"Best checkpoint copied to {best_ckpt_dest}, other checkpoints removed.")
