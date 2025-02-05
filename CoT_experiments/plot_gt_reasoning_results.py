import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import random
import json
import numpy as np


mpl.rcParams.update({
    'font.size': 20,        # 기본 글씨 크기
    'axes.labelsize': 25,   # x, y축 label 폰트 크기
    'axes.titlesize': 20,   # subplot title 폰트 크기
    'xtick.labelsize': 15,  # x축 tick 폰트 크기
    'ytick.labelsize': 20,  # y축 tick 폰트 크기
    'legend.fontsize': 20,  # 범례 폰트 크기
})


# Example task, reasoning, shot, seed, and metric lists
task_names = ["forward", "retro", "reagent", "catalyst", "solvent"]
# methods = ["GPT-4o", "LlaSMol", "PRESTO", "ReactReasoner(Ours)"]
method = "ReactReasoner(Ours)"
# n_shotcutoff_texts_texts = ["0.0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0", "before_gt_smiles"]
cutoff_texts = ["0.0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"]
metrics = ["exact_match", "bleu", "levenshtein", "rdk_sims", "maccs_sims", "morgan_sims", "validity"]


task_names_to_name = {
    "forward": "Forward",
    "retro": "Retrosynthesis",
    "reagent": "Reagent",
    "catalyst": "Catalyst",
    "solvent": "Solvent"
}

metric_to_name = {
    "exact_match": "EXACT↑",
    "bleu": "BLEU↑",
    "levenshtein": "LEVENSHTEIN↓",
    "rdk_sims": "RDK FTS↑",
    "maccs_sims": "MACCS FTS↑",
    "morgan_sims": "MORGAN FTS↑",
    "validity": "VALIDITY↑"
}


d = {}
base_path = "/home/hko/PRESTO/logs/full-ckpt-best/cutoff_test"
for task_name in task_names:
    d[task_name] = {}
    print(f"Task: {task_name}")
    for cutoff in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, "before_gt_smiles"]:
        cutoff = str(cutoff)
        d[task_name][cutoff] = {}
        if os.path.exists(f"{base_path}/{task_name}-{cutoff}-cutoff"):
            with open(f"{base_path}/{task_name}-{cutoff}-cutoff/score.json") as f:
                metrics = json.load(f)
                for key, val in metrics.items():
                    d[task_name][cutoff][key] = val


d_compare = {
    "LlaSMol": {
        "forward": {
            "exact_match": 0.4,
            "bleu": 0.817,
            "levenshtein": 5.83,
            "rdk_sims": 0.625,
            "maccs_sims": 0.785,
            "morgan_sims": 0.632,
            "validity": 1
        },
        "retro": {
            "exact_match": 0.3,
            "bleu": 0.802,
            "levenshtein": 14.43,
            "rdk_sims": 0.744,
            "maccs_sims": 0.84,
            "morgan_sims": 0.705,
            "validity": 1
        },
        "reagent": {
            "exact_match": 0,
            "bleu": 0.042,
            "levenshtein": 49.41,
            "rdk_sims": 0.018,
            "maccs_sims": 0.098,
            "morgan_sims": 0.024,
            "validity": 1
        },
        "catalyst": {
            "exact_match": 0,
            "bleu": 0.022,
            "levenshtein": 46.95,
            "rdk_sims": 0.014,
            "maccs_sims": 0.054,
            "morgan_sims": 0.017,
            "validity": 1
        },
        "solvent": {
            "exact_match": 0,
            "bleu": 0.034,
            "levenshtein": 43.97,
            "rdk_sims": 0.013,
            "maccs_sims": 0.105,
            "morgan_sims": 0.033,
            "validity": 1
        },
    },
    "PRESTO": {
        "forward": {
            "exact_match": 0.17,
            "bleu": 0.641,
            "levenshtein": 9.67,
            "rdk_sims": 0.411,
            "maccs_sims": 0.588,
            "morgan_sims": 0.446,
            "validity": 1
        },
        "retro": {
            "exact_match": 0.36,
            "bleu": 0.81,
            "levenshtein": 11.07,
            "rdk_sims": 0.696,
            "maccs_sims": 0.811,
            "morgan_sims": 0.694,
            "validity": 0.96
        },
        "reagent": {
            "exact_match": 0.45,
            "bleu": 0.668,
            "levenshtein": 6.4,
            "rdk_sims": 0.672,
            "maccs_sims": 0.682,
            "morgan_sims": 0.583,
            "validity": 1
        },
        "catalyst": {
            "exact_match": 0.74,
            "bleu": 0.698,
            "levenshtein": 1.98,
            "rdk_sims": 0.911,
            "maccs_sims": 0.888,
            "morgan_sims": 0.743,
            "validity": 1
        },
        "solvent": {
            "exact_match": 0.46,
            "bleu": 0.539,
            "levenshtein": 3.33,
            "rdk_sims": 0.547,
            "maccs_sims": 0.57,
            "morgan_sims": 0.532,
            "validity": 1
        },
    },
}

# method 별로 라인을 구분하기 위해 색상, 마커 등을 지정
method_styles = {
    "GPT-4o": {
        "color": "#1f77b4", 
        "marker": "o", 
        "label": "GPT-4o"
    },
    "LlaSMol": {
        "color": "#ff7f0e", 
        "marker": "o", 
        "label": "LlaSMol"
    },
    "PRESTO": {
        "color": "#2ca02c",
        "marker": "o",
        "label": "PRESTO"
    },
    "ReactReasoner(Ours)": {
        "color": "#d62728",
        "marker": "o",
        "label": "ReactReasoner"
    },
}



fig, axes = plt.subplots(nrows=5, ncols=7, figsize=(35/1.3, 25/1.3), sharey=False)
for i, task in enumerate(task_names):
    for j, metric in enumerate(metrics):
        ax = axes[i, j]
        
        # x축을 n_shot_list의 인덱스 값으로 사용하되, 레이블은 n_shot_list 자체
        x_vals = np.arange(len(cutoff_texts))
        
        # 각 method에 대해 라인으로 그림
        ax.plot(
            x_vals,
            [d[task][n_shot][metric] for n_shot in cutoff_texts],
            color=method_styles[method]["color"],
            marker=method_styles[method]["marker"],
            label=method_styles[method]["label"]
        )

        # Draw horizontal line for comparison
        ax.axhline(y=d_compare["LlaSMol"][task][metric], color=method_styles["LlaSMol"]["color"], linestyle='--', label="LlaSMol")
        ax.axhline(y=d_compare["PRESTO"][task][metric], color=method_styles["PRESTO"]["color"], linestyle='--', label="PRESTO")
        ax.axhline(y=d[task]["before_gt_smiles"][metric], color=method_styles["ReactReasoner(Ours)"]["color"], linestyle='--', label="ReactReasoner*")
        
        # x축 눈금과 레이블 설정
        ax.set_xticks(x_vals)
        ax.set_xticklabels(cutoff_texts, rotation=90, ha='right')
        
        # (변경 1) subplot title 제거
        # ax.set_title(f"{task} - {metric}")  # 제거
        
        # (변경 2) 맨 왼쪽 컬럼이면 row label(= task)
        if j == 0:
            ax.set_ylabel(task_names_to_name[task])
        
        # (변경 3) 맨 윗줄이면 column label(= metric)
        if i == 0:
            ax.set_title(metric_to_name[metric])


        # handles, labels = ax.get_legend_handles_labels()
        # ax.legend(
        #     handles, 
        #     labels, 
        #     loc='upper center', 
        #     bbox_to_anchor=(0.5, 1.25),  # 그래프 위에 배치
        #     fontsize='small', 
        #     ncol=1,  # 한 줄로 정렬
        #     frameon=False  # 범례 테두리 제거
        # )
# Legend 정보를 수집
all_handles = []
all_labels = []
for row in axes:
    for ax in row:
        handles, labels = ax.get_legend_handles_labels()
        all_handles.extend(handles)
        all_labels.extend(labels)

# 중복 라벨 제거
by_label = {label: handle for label, handle in zip(all_labels, all_handles)}

# 범례 추가: 플롯 상단에 한 번만 표시
fig.legend(by_label.values(),
           by_label.keys(),
           loc='upper center',  # 플롯 상단 중앙에 배치
           ncol=4,  # 범례 항목을 한 줄에 네 개씩 정렬
           bbox_to_anchor=(0.5, 0.99),  # 플롯 위로 약간 띄워서 배치
        #    fontsize='medium',
           frameon=True)

# Save and display
plt.tight_layout(rect=[0, 0, 1, 0.97])  # 전체 플롯 레이아웃 조정
# plt.tight_layout()  # 전체 플롯 레이아웃 조정
plt.savefig(f"CoT_experiments/results/cot_prompt_test/plots/GT_reasoning_line.png", dpi=200)
plt.savefig(f"CoT_experiments/results/cot_prompt_test/plots/GT_reasoning_line.pdf", dpi=200)
plt.clf()