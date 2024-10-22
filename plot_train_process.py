import os
import pandas as pd
import matplotlib.pyplot as plt

# If "analysis_results/plots" directory does not exist, create it
if not os.path.exists('analysis_results/plots'):
    os.makedirs('analysis_results/plots')


column_name_to_metric_name = {
    # "bleu2": "BLEU-2",
    # "bleu4": "BLEU-4",
    # "meteor_score": "METEOR",
    # "rouge_1": "ROUGE-1",
    # "rouge_2": "ROUGE-2",
    # "rouge_l": "ROUGE-L",
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


def save_plot(metric_csv_path, title, save_name, text_sim_metrics):
    assert type(text_sim_metrics) == list
    # Read metrics.csv file
    df = pd.read_csv(metric_csv_path)

    # Set up the figure
    fig, ax1 = plt.subplots(figsize=(6, 4))

    # Plot the train and validation loss on the left y-axis
    plot_with_drop_nan(ax1, list(df['step']), list(df['val molecule loss/dataloader_idx_0']), label='Validation loss', color=colors[0])
    plot_with_drop_nan(ax1, list(df['step']), list(df['molecule loss']), label='Train loss', color=colors[1])
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    ax1.legend(loc='upper left')

    # Set up the second y-axis
    ax2 = ax1.twinx()
    for i, text_sim_metric in enumerate(text_sim_metrics):
        plot_with_drop_nan(ax2, list(df['step']), list(df[text_sim_metric]), label=column_name_to_metric_name[text_sim_metric], linestyle='--', color=colors[i+2])
    ax2.set_ylabel('Text similarity score')
    ax2.legend(loc='center right')

    # Save the figure
    plt.title(title)
    plt.savefig(f"analysis_results/plots/{save_name}.png")
    plt.clf()
    

#######################################################################################################################################
save_plot(
    'all_checkpoints/ft_pubchem324k_from_origin_test_overfit/lightning_logs/version_0/metrics.csv',
    'Learning curve of fine-tuning molecule captioning (PubChem324k)',
    'ft_pubchem324k_from_origin_test_overfit_meteor',
    ['meteor_score_val', 'meteor_score_train']
)
save_plot(
    'all_checkpoints/ft_chebi20_from_origin_test_overfit/lightning_logs/version_0/metrics.csv',
    'Learning curve of fine-tuning molecule captioning (ChEBI20)',
    'ft_chebi20_from_origin_test_overfit_meteor',
    ['meteor_score_val', 'meteor_score_train']
)
save_plot(
    'all_checkpoints/ft_iupac_from_origin_test_overfit/lightning_logs/version_0/metrics.csv',
    'Learning curve of fine-tuning IUPAC name prediction',
    'ft_iupac_from_origin_test_overfit_meteor',
    ['meteor_score_val', 'meteor_score_train']
)

save_plot(
    'all_checkpoints/ft_pubchem324k_from_origin_test_overfit/lightning_logs/version_0/metrics.csv',
    'Learning curve of fine-tuning molecule captioning (PubChem324k)',
    'ft_pubchem324k_from_origin_test_overfit_full',
    list(column_name_to_metric_name.keys())
)
save_plot(
    'all_checkpoints/ft_chebi20_from_origin_test_overfit/lightning_logs/version_0/metrics.csv',
    'Learning curve of fine-tuning molecule captioning (ChEBI20)',
    'ft_chebi20_from_origin_test_overfit_full',
    list(column_name_to_metric_name.keys())
)
save_plot(
    'all_checkpoints/ft_iupac_from_origin_test_overfit/lightning_logs/version_0/metrics.csv',
    'Learning curve of fine-tuning IUPAC name prediction',
    'ft_iupac_from_origin_test_overfit_full',
    list(column_name_to_metric_name.keys())
)