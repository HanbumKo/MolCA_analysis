import matplotlib.pyplot as plt
import matplotlib as mpl
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
# task_names = ["forward"]
# use_reasoning_texts = ["w/o reasoning instruction", "w/ reasoning instruction"]
# reasoning = ["no", "generated", "manual"]
# reasoning = ["no", "generated", "zeroshotcot"]
reasoning = ["no", "generated"]
n_shot_texts = ["0-shot", "1-shot", "2-shot", "3-shot", "4-shot", "5-shot", "6-shot", "7-shot"]
seed_texts = ["seed0", "seed1", "seed2", "seed3"]
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

# for model in ["gpt-4o-mini", "gpt-3.5-turbo", "gpt-4o-2024-11-20"]:
for model in ["llasmol"]:
# for model in ["llasmol"]:
    with open(f"CoT_experiments/results/cot_prompt_test/eval_results/{model}.json", "r") as f:
        d = json.load(f)

    # Plot 5x7 subplots
    fig, axes = plt.subplots(nrows=5, ncols=7, figsize=(35, 25), sharey=False)

    for i, task in enumerate(task_names):
        for j, metric in enumerate(metrics):
            ax = axes[i, j]
            
            # n_shot 별로 w/o, w/ reasoning(각각 4개 seed) 데이터를 모아 박스플롯
            all_box_data = []
            positions = []
            
            # 그룹 간 간격
            group_gap = 2.0
            # w/o, w/ reasoning 두 박스를 x축에서 살짝 나누기 위한 offset
            offset = 0.3
            
            for idx_n, n_shot in enumerate(n_shot_texts):
                for reason in reasoning:
                    # 데이터
                    data = [
                        d[task][reason][n_shot][s][metric]
                        for s in seed_texts
                    ]
                    all_box_data.append(data)
                    positions.append(idx_n * group_gap)
                # # w/o reasoning 데이터
                # data_wo = [
                #     d[task]["w/o reasoning instruction"][n_shot][s][metric]
                #     for s in seed_texts
                # ]
                # # w/ reasoning 데이터
                # data_w = [
                #     d[task]["w/ reasoning instruction"][n_shot][s][metric]
                #     for s in seed_texts
                # ]
                
                # all_box_data.append(data_wo)
                # positions.append(idx_n * group_gap - offset)
                
                # all_box_data.append(data_w)
                # positions.append(idx_n * group_gap + offset)
            
            # 박스플롯 그리기
            bp = ax.boxplot(all_box_data, positions=positions, widths=0.5, patch_artist=True)
            
            # 박스 색상 설정
            colors = []
            for k in range(len(all_box_data)):
                if k % len(reasoning) == 0:
                    colors.append('#1f77b4')
                elif k % len(reasoning) == 1:
                    colors.append('#ff7f0e')
                elif k % len(reasoning) == 2:
                    colors.append('#2ca02c')
                else:
                    raise ValueError("Too many reasoning types")
            
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.5)
            
            # x축 tick 설정: n_shot_list 개수만큼
            ax.set_xticks([idx * group_gap for idx in range(len(n_shot_texts))])
            ax.set_xticklabels(n_shot_texts, rotation=45, ha='right')
            
            # (변경 1) subplot title 제거
            # ax.set_title(f"{task} - {metric}")  # ← 제거
            
            # (변경 2) 맨 왼쪽 컬럼이면 row label(= task)
            if j == 0:
                ax.set_ylabel(task_names_to_name[task])
            
            # (변경 3) 맨 윗줄이면 column label(= metric)
            if i == 0:
                ax.set_title(metric_to_name[metric])

    # 범례는 맨 위 왼쪽 subplot에 직접 달 수도 있고(fig.legend 등을 통해 전체로 뺄 수도 있음)
    # axes[0, 0].legend(
    #     [bp['boxes'][0], bp['boxes'][1], bp['boxes'][2]],
    #     ["w/o reasoning", "w/ reasoning(generated)", "w/ reasoning(zero-shot-cot)"],
    #     loc="upper left"
    # )
    axes[0, 0].legend(
        [bp['boxes'][0], bp['boxes'][1]],
        ["Few-shot", "Few-shot-CoT(Ours)"],
        loc="upper left"
    )

    plt.tight_layout()
    plt.savefig(f"CoT_experiments/results/cot_prompt_test/plots/{model}_test_box.png", dpi=200)
    plt.clf()






    fig, axes = plt.subplots(nrows=5, ncols=7, figsize=(35, 25), sharey=False)

    # use_reasoning별로 라인을 구분하기 위해 색상, 마커 등을 지정
    reasoning_styles = {
        "no": {
            "color": "#1f77b4", 
            "marker": "o", 
            "label": "Few-shot"
        },
        "generated": {
            "color": "#ff7f0e", 
            "marker": "o", 
            "label": "Few-shot-CoT(Ours)"
        },
        # "zeroshotcot": {
        #     "color": "#2ca02c",
        #     "marker": "o",
        #     "label": "w/ reasoning(zero-shot-cot)"
        # }
    }

    for i, task in enumerate(task_names):
        for j, metric in enumerate(metrics):
            ax = axes[i, j]
            
            # x축을 n_shot_list의 인덱스 값으로 사용하되, 레이블은 n_shot_list 자체
            x_vals = np.arange(len(n_shot_texts))
            
            # 각 use_reasoning(=w/o, w/)에 대해 4개 seed 평균을 라인으로 그림
            for ur in reasoning:
                y_means = []
                for ns in n_shot_texts:
                    # seed 4개의 값 평균
                    seed_vals = [
                        d[task][ur][ns][s][metric] for s in seed_texts
                    ]
                    y_means.append(np.mean(seed_vals))
                
                ax.plot(
                    x_vals,
                    y_means,
                    color=reasoning_styles[ur]["color"],
                    marker=reasoning_styles[ur]["marker"],
                    label=reasoning_styles[ur]["label"]
                )
            
            # x축 눈금과 레이블 설정
            ax.set_xticks(x_vals)
            ax.set_xticklabels(n_shot_texts, rotation=45, ha='right')
            
            # (변경 1) subplot title 제거
            # ax.set_title(f"{task} - {metric}")  # 제거
            
            # (변경 2) 맨 왼쪽 컬럼이면 row label(= task)
            if j == 0:
                ax.set_ylabel(task_names_to_name[task])
            
            # (변경 3) 맨 윗줄이면 column label(= metric)
            if i == 0:
                ax.set_title(metric_to_name[metric])

    # 범례: 전체 플롯에서 한 번에 보여주기 위해 fig.legend 사용
    # handles, labels를 가져와서 정리할 수도 있으나 여기서는 직접 reasoning_styles를 사용
    lines = [
        plt.Line2D([0], [0], color=reasoning_styles["no"]["color"],
                marker=reasoning_styles["no"]["marker"], label="Few-shot"),
        plt.Line2D([0], [0], color=reasoning_styles["generated"]["color"],
                marker=reasoning_styles["generated"]["marker"], label="Few-shot-CoT(Ours)"),
        # plt.Line2D([0], [0], color=reasoning_styles["zeroshotcot"]["color"],
        #         marker=reasoning_styles["zeroshotcot"]["marker"], label="w/ reasoning(zero-shot-cot)")
    ]
    # fig.legend(handles=lines, loc='upper center', ncol=2)
    axes[0, 0].legend(handles=lines, loc='upper left')

    plt.tight_layout()
    plt.savefig(f"CoT_experiments/results/cot_prompt_test/plots/{model}_test_line.png", dpi=200)
    plt.clf()
