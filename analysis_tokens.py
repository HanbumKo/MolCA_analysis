import itertools
import torch
import matplotlib.pyplot as plt

from transformers import AutoTokenizer
from collections import Counter
from torch_geometric.data import InMemoryDataset


class PubChemDataset(InMemoryDataset):
    def __init__(self, path):
        super(PubChemDataset, self).__init__()
        self.data, self.slices = torch.load(path)
    
    def __getitem__(self, idx):
        return self.get(idx)


def plot_histogram(ngrams_freq, task, ns):
    for n in ns:
        title = f"{n}-gram Frequency Distribution ({task})"
        filename = f"analysis_results/plots/{n}-gram_distribution_{task}.png"
        plt.figure(figsize=(10, 6))
        for split in ngrams_freq.keys():
            ngram_freq = ngrams_freq[split][n]
            tokens = list(ngram_freq.keys())
            frequencies = list(ngram_freq.values())
            # plt.bar(tokens, frequencies, alpha=0.5, label=split)
            plt.hist(tokens, weights=frequencies, alpha=0.5, label=split, bins=300)
        plt.xlabel(f"{n}-grams")
        plt.ylabel('Frequency')
        plt.title(title)
        # plt.xticks(rotation=90)
        plt.xticks([])
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(filename)


# Tokenize and count tokens and n-grams
def tokenize_and_count(string_list):
    all_tokens = []
    all_ngrams = {n: [] for n in ns}

    for text in string_list:
        tokens = tokenizer.tokenize(text)
        tokens = [token.replace('Ġ', ' ') for token in tokens]
        all_tokens.extend(tokens)

        for n in ns:
            ngrams = list(zip(*[tokens[i:] for i in range(n)]))
            all_ngrams[n].extend([''.join(ngram) for ngram in ngrams])


    # Calculate token frequencies
    token_freq = Counter(all_tokens)

    # Calculate n-gram frequencies
    ngram_freqs = {n: Counter(all_ngrams[n]) for n in all_ngrams}
    
    # sort ngram_freqs by frequency
    for n in ngram_freqs:
        ngram_freqs[n] = dict(sorted(ngram_freqs[n].items(), key=lambda x: x[1], reverse=True))

    # sort token_freq by frequency
    token_freq = dict(sorted(token_freq.items(), key=lambda x: x[1], reverse=True))

    return ngram_freqs


pretrain_dataset = PubChemDataset('./data/PubChem324kV2/pretrain.pt')
train_dataset = PubChemDataset('./data/PubChem324kV2/train.pt')
valid_dataset = PubChemDataset('./data/PubChem324kV2/valid.pt')
test_dataset = PubChemDataset('./data/PubChem324kV2/test.pt')

pretrain_descriptions = pretrain_dataset.text
train_descriptions = train_dataset.text
valid_descriptions = valid_dataset.text
test_descriptions = test_dataset.text

# pretrain_iupac = pretrain_dataset.iupac
train_iupac = train_dataset.iupac
valid_iupac = valid_dataset.iupac
test_iupac = test_dataset.iupac

descriptions = {
    "train": train_descriptions,
    "val": valid_descriptions,
    "test": test_descriptions
}
iupacs = {
    "train": train_iupac,
    "val": valid_iupac,
    "test": test_iupac
}

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/galactica-1.3b")

ns = [2, 4, 6]


# Calculate token and n-gram frequencies
ngrams_descriptions = {split: tokenize_and_count(descriptions[split]) for split in descriptions }
ngrams_iupacs = {split: tokenize_and_count(iupacs[split]) for split in iupacs }
plot_histogram(ngrams_descriptions, "Molecule captioning", ns)
plot_histogram(ngrams_iupacs, "IUPAC name prediction", ns)
