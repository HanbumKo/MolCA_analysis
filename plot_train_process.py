import os
import pandas as pd
import matplotlib.pyplot as plt

# If "analysis_results/plots" directory does not exist, create it
if not os.path.exists('analysis_results/plots'):
    os.makedirs('analysis_results/plots')


column_name_to_metric_name = {    
    "bleu2_train": "BLEU-2(Train)",
    "bleu4_train": "BLEU-4(Train)",
    "meteor_score_train": "METEOR(Train)",
    "rouge_1_train": "ROUGE-1(Train)",
    "rouge_2_train": "ROUGE-2(Train)",
    "rouge_l_train": "ROUGE-L(Train)",
    "bleu2_val": "BLEU-2(Val)",
    "bleu4_val": "BLEU-4(Val)",
    "meteor_score_val": "METEOR(Val)",
    "rouge_1_val": "ROUGE-1(Val)",
    "rouge_2_val": "ROUGE-2(Val)",
    "rouge_l_val": "ROUGE-L(Val)",
    "bleu2_test": "BLEU-2(Test)",
    "bleu4_test": "BLEU-4(Test)",
    "meteor_score_test": "METEOR(Test)",
    "rouge_1_test": "ROUGE-1(Test)",
    "rouge_2_test": "ROUGE-2(Test)",
    "rouge_l_test": "ROUGE-L(Test)",
    
    "num_t1_exact_match_train": "Exact match(Train)",
    "num_t1_invalid_test": "Invalid(Test)",
    "num_t1_no_answer_train": "No answer(Train)",
    "t1_maccs_fps_train": "MACCS(Train)",
    "t1_morgan_fps_train": "Morgan(Train)",
    "t1_rdk_fps_train": "RDKit(Train)",
    "num_t1_exact_match_val": "Exact match(Val)",
    "num_t1_invalid_val": "Invalid(Val)",
    "num_t1_no_answer_val": "No answer(Val)",
    "t1_maccs_fps_val": "MACCS(Val)",
    "t1_morgan_fps_val": "Morgan(Val)",
    "t1_rdk_fps_val": "RDKit(Val)",
    "num_t1_exact_match_test": "Exact match(Test)",
    "num_t1_invalid_test": "Invalid(Test)",
    "num_t1_no_answer_test": "No answer(Test)",
    "t1_maccs_fps_test": "MACCS(Test)",
    "t1_morgan_fps_test": "Morgan(Test)",
    "t1_rdk_fps_test": "RDKit(Test)",
    
    "mae_train": "MAE(Train)",
    "mse_train": "MSE(Train)",
    "rmse_train": "RMSE(Train)",
    "validity_train": "Validity(Train)",
    "mae_val": "MAE(Val)",
    "mse_val": "MSE(Val)",
    "rmse_val": "RMSE(Val)",
    "validity_val": "Validity(Val)",
    "mae_test": "MAE(Test)",
    "mse_test": "MSE(Test)",
    "rmse_test": "RMSE(Test)",
    "validity_test": "Validity(Test)",
    
    "num_t1_intersection_train": "Intersection(Train)",
    "num_t1_subset_train": "Subset(Train)",
    "num_t1_intersection_val": "Intersection(Val)",
    "num_t1_subset_val": "Subset(Val)",
    "num_t1_intersection_test": "Intersection(Test)",
    "num_t1_subset_test": "Subset(Test)",    
}

colors = [
    "red",
    "blue",
    "olive",
    "lawngreen",
    "gold",
    "peru",
    "turquoise",
    "deepskyblue",
    "darkviolet",
    "pink",
    "black",
    "gray",
    "darkorange",
    "darkgreen",
]


def plot_with_drop_nan(ax, steps, values, **kwargs):
    # If there are NaN values in the values, remove them and remove same indices in steps
    nan_indices = [i for i, value in enumerate(values) if pd.isna(value)]
    steps = [step for i, step in enumerate(steps) if i not in nan_indices]
    values = [value for i, value in enumerate(values) if i not in nan_indices]
    ax.plot(steps, values, **kwargs)


def save_plot(file_name, version=0):
    if "_property_" in file_name:
        main_metric_name = "mae"
        main_y_metric_label = "MAE(↓)"
        optimal = 0.
        metric_list = ["mae", "mse", "rmse", "validity"]
    elif "_retrosynthesis_" in file_name or "_forward_" in file_name or "_reagent_" in file_name:
        main_metric_name = "num_t1_exact_match"
        main_y_metric_label = "Exact match(↑)"
        optimal = 100.
        metric_list = ["num_t1_exact_match", "num_t1_invalid", "num_t1_no_answer", "t1_maccs_fps", "t1_morgan_fps", "t1_rdk_fps"]
    elif "_chebi20_" in file_name or "_pubchem324k_" in file_name or "_iupac_" in file_name:
        main_metric_name = "meteor_score"
        main_y_metric_label = "METEOR(↑)"
        optimal = 100.
        metric_list = ["bleu2", "bleu4", "meteor_score", "rouge_1", "rouge_2", "rouge_l"]
    else:
        raise ValueError(f"Unknown file_name: {file_name}")

    if "_property_" in file_name:
        title_name = "property prediction"
        y_range = None
    elif "_retrosynthesis_" in file_name:
        title_name = "retrosynthesis"
        y_range = None
    elif "_forward_" in file_name:
        title_name = "forward prediction"
        y_range = None
    elif "_reagent_" in file_name:
        title_name = "reagent prediction"
        y_range = None
    elif "_chebi20_" in file_name:
        title_name = "molecule captioning(ChEBI20)"
        y_range = (50, 105)
    elif "_pubchem324k_" in file_name:
        title_name = "molecule captioning(PubChem324k)"
        y_range = None
    elif "_iupac_" in file_name:
        title_name = "IUPAC name prediction"
        y_range = (60, 105)
    else:
        raise ValueError(f"Unknown file_name: {file_name}")
    
    metric_csv_path = f"all_checkpoints/{file_name}/lightning_logs/version_{version}/metrics.csv"
    df = pd.read_csv(metric_csv_path)
    fig, ax1 = plt.subplots(figsize=(6, 4))

    # Plot the train and validation loss on the left y-axis
    plot_with_drop_nan(ax1, list(df['step']), list(df['molecule loss']), label='Loss(Train)', color=colors[1])
    plot_with_drop_nan(ax1, list(df['step']), list(df['val molecule loss/dataloader_idx_0']), label='Loss(Val)', color=colors[0])
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    ax1.legend(loc='upper left')

    # Plot the main metric on the right y-axis
    ax2 = ax1.twinx()
    if main_y_metric_label == "Exact match(↑)":
        plot_with_drop_nan(ax2, list(df['step']), list(df[main_metric_name+"_train"]/df["num_all_train"]*100), label=column_name_to_metric_name[main_metric_name+"_train"], linestyle='--', color=colors[2])
        plot_with_drop_nan(ax2, list(df['step']), list(df[main_metric_name+"_val"]/df["num_all_val"]*100), label=column_name_to_metric_name[main_metric_name+"_val"], linestyle='--', color=colors[3])
        # plot_with_drop_nan(ax2, list(df['step']), list(df[main_metric_name+"_test"]/df["num_all_test"]), label=column_name_to_metric_name[main_metric_name+"_test"], linestyle='--', color=colors[4])

    else:
        plot_with_drop_nan(ax2, list(df['step']), list(df[main_metric_name+"_train"]), label=column_name_to_metric_name[main_metric_name+"_train"], linestyle='--', color=colors[2])
        plot_with_drop_nan(ax2, list(df['step']), list(df[main_metric_name+"_val"]), label=column_name_to_metric_name[main_metric_name+"_val"], linestyle='--', color=colors[3])
        # plot_with_drop_nan(ax2, list(df['step']), list(df[main_metric_name+"_test"]), label=column_name_to_metric_name[main_metric_name+"_test"], linestyle='--', color=colors[4])
    # plot optimal
    ax2.axhline(y=optimal, linestyle=':', label='Optimal', color=colors[5])
    if y_range is not None:
        ax2.set_ylim(y_range)
    ax2.set_ylabel(main_y_metric_label)
    ax2.legend(loc='center right')
    
    plt.title(f"Learning curve of {title_name}")
    plt.tight_layout()
    plt.savefig(f"analysis_results/plots/{file_name}_main.png")
    plt.clf()

    # # Set up the second y-axis
    # ax2 = ax1.twinx()
    # for i, text_sim_metric in enumerate(text_sim_metrics):
    #     plot_with_drop_nan(ax2, list(df['step']), list(df[text_sim_metric]), label=column_name_to_metric_name[text_sim_metric], linestyle='--', color=colors[i+2])
    # ax2.set_ylabel('Text similarity score')
    # ax2.legend(loc='center right')


    # plt.title(title)
    # plt.savefig(f"analysis_results/plots/{save_name}.png")
    # plt.clf()
    

#######################################################################################################################################
file_names = [
    "ft_pubchem324k_from_origin",
    "ft_chebi20_from_origin",
    "ft_iupac_from_origin",
    "ft_property_from_origin",
    "ft_retrosynthesis_from_origin",
    "ft_forward_from_origin",
    
    "ft_pubchem324k_stringonly",
    "ft_chebi20_stringonly",
    "ft_iupac_stringonly",
    "ft_property_stringonly",
    "ft_retrosynthesis_stringonly",
    "ft_forward_stringonly",
    
    "ft_USPTO_forward_from_origin",
    "ft_USPTO_retrosynthesis_from_origin",
    "ft_USPTO_forward_stringonly",
    "ft_USPTO_retrosynthesis_stringonly",
]
for file_name in file_names:
    save_plot(
        file_name=file_name
    )
    print(f"Saved {file_name}_main.png")
