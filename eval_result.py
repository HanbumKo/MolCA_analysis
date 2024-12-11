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
    metric_name_test = "mae_test"
elif "_retrosynthesis_" in file_name or "_forward_" in file_name or "_reagent_" in file_name:
    metric_name = "num_t1_exact_match_val"
    metric_name_test = "num_t1_exact_match_test"
elif "_chebi20_" in file_name or "_pubchem324k_" in file_name or "_iupac_" in file_name:
    metric_name = "meteor_score_val"
    metric_name_test = "meteor_score_test"
else:
    raise ValueError(f"Unknown file_name: {file_name}")


# Load metrics.csv
metrics_path = f"all_checkpoints/{file_name}/lightning_logs/version_0/metrics.csv"
metrics_df = pd.read_csv(metrics_path)

if metric_name != "mae_val":
    # Find row that has maximum metric_name value
    if metric_name == "num_t1_exact_match_val":
        best_test_metric = metrics_df.loc[metrics_df[metric_name].idxmax(), metric_name_test] / metrics_df.loc[metrics_df[metric_name].idxmax(), 'num_all_test']
    else:
        best_test_metric = metrics_df.loc[metrics_df[metric_name].idxmax(), metric_name_test]
else:
    # Find row that has minimum metric_name value
    best_test_metric = metrics_df.loc[metrics_df[metric_name].idxmin(), metric_name_test]


print(f"Best test metric: {best_test_metric}")