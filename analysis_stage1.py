import os
import argparse
import warnings
import torch
import random
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

from model.blip2_stage1 import Blip2Stage1
from data_provider.stage1_dm import Stage1DM, TrainCollater
from data_provider.stage1_kvplm_dm import Stage1KVPLMDM
from torch.utils.data import DataLoader
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdMolDescriptors
from rdkit import Chem
from rdkit.Chem import Draw
from io import BytesIO
from PIL import Image
from tqdm import tqdm
import matplotlib.ticker as ticker



## for pyg bug
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
## for A5000 gpus
torch.set_float32_matmul_precision('medium') # can be medium (bfloat16), high (tensorfloat32), highest (float32)


def to_device(data, device):
    data_device = tuple([d.to(device) for d in data])
    return data_device


# Function to map a value to a color
def value_to_color(value):
    return (1.0 - value, 1.0, 1.0 - value) # The higher the value, the closer to


def visualize_attention_heatmaps_in_one(attention_maps, idx, root_dir):
    """
    attention_maps: numpy array of shape [batch_size, num_queries, num_keys]
    Visualizes all attention maps in one figure with subplots.
    """
    batch_size, num_queries, num_keys = attention_maps.shape
    for i in range(batch_size):
        instance_i = idx * batch_size + i
        attention_map = attention_maps[i]
        attention_map = attention_map[:, attention_map[0] != 0] # Drop zero values
        plt.figure(figsize=(3, 3))
        heatmap = plt.imshow(attention_map, cmap='viridis', aspect='auto')
        cbar = plt.colorbar(heatmap)
        cbar.set_label('Attention score', fontsize=6)
        cbar.ax.tick_params(labelsize=6)
        plt.title(f'Instance #{instance_i}', fontsize=8)
        plt.xlabel('Graph node index', fontsize=6)
        plt.ylabel('Query index', fontsize=6)
        plt.tick_params(axis='both', which='major', labelsize=6)
        ax = plt.gca()
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        xticks = ax.get_xticks().astype(int)
        xtick_labels = [str(x) for x in xticks]
        if 0 in xticks:
            xtick_labels[xticks.tolist().index(0)] = 'GR'
        ax.set_xticklabels(xtick_labels, fontsize=6)
        plt.tight_layout()
        plt.savefig(f"{root_dir}/cross_attention_maps/attention_maps_{instance_i}.png", dpi=300)
        plt.close()


def visulize_molecule_graphs(attention_scores, graphs, idx, root_dir):
    batch_size = len(graphs)
    for graph_i in range(batch_size):
        graph = graphs[graph_i]
        mol = Chem.MolFromSmiles(graph.smiles)
        drawer = rdMolDraw2D.MolDraw2DCairo(1000, 1000)
        attention_score = attention_scores[graph_i]
        attention_score = attention_score[attention_score != 0]# Drop zero values
        # print(sum(attention_score))
        print()
        print("="*100)
        print(attention_score.max() - attention_score.min())
        print(len(graph.text))
        print("="*100)
        print()
        # Rescale the attention score to [0, 1]
        attention_score -= attention_score.min()
        attention_score /= attention_score.max()
        atom_colors = {}
        for atom_i, atom in enumerate(mol.GetAtoms()):
            # print(f"node feature {atom_i}: {graph.x[atom_i]}")
            # atom.SetProp("molAtomMapNumber", str(atom.GetIdx()))
            atom.SetProp("atomNote", str(atom.GetIdx()+1))
            atom_colors[atom.GetIdx()] = value_to_color(attention_score[atom_i])
        drawer.DrawMolecule(mol, highlightAtoms=list(atom_colors.keys()), highlightAtomColors=atom_colors, highlightBonds=False)
        drawer.FinishDrawing()
        img = drawer.GetDrawingText()
        img = Image.open(BytesIO(img))
        img.save(f"{root_dir}/molecule_highlights/molecule_{idx*batch_size+graph_i}.png")


def scatter_attention_textlen(attention_scores, graphs, idx):
    attscore_diffs = []
    text_lens = []
    batch_size = len(graphs)
    for graph_i in range(batch_size):
        graph = graphs[graph_i]
        attention_score = attention_scores[graph_i]
        attention_score = attention_score[attention_score != 0]# Drop zero values
        attscore_diffs.append(attention_score.max() - attention_score.min())
        text_lens.append(len(graph.text))

    return attscore_diffs, text_lens




def main(args):
    model = Blip2Stage1.load_from_checkpoint(args.checkpoint, device=args.devices, args=args, map_location="cuda")
    model.eval()

    checkpoint_name = args.checkpoint.split("/")[-2]
    analysis_root_dir = f"analysis_results/{checkpoint_name}"
    analysis_dirs = [
        f"{analysis_root_dir}/cross_attention_maps",
        f"{analysis_root_dir}/molecule_highlights",
        f"{analysis_root_dir}/scatter_plots",
    ]
    os.makedirs("analysis_results", exist_ok=True)
    os.makedirs(analysis_root_dir, exist_ok=True)
    for analysis_dir in analysis_dirs:
        os.makedirs(analysis_dir, exist_ok=True)

    blip2qformer = model.blip2qformer
    tokenizer = model.blip2qformer.tokenizer
    device = blip2qformer.gtm_head.bias.device
    
    if args.root.find('kv') >= 0:
        dm = Stage1KVPLMDM(args.num_workers, args.batch_size, args.root, args.text_max_len, args.graph_aug, args)
    else:
        dm = Stage1DM(args.num_workers, args.batch_size, args.root, args.text_max_len, args.graph_aug, tokenizer,
                      args)
    train_dataset = dm.train_dataset
    val_dataset = dm.val_dataset
    test_dataset = dm.test_dataset

    train_loader = DataLoader(train_dataset, batch_size=args.match_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False, drop_last=False, persistent_workers=True, collate_fn=TrainCollater(tokenizer, args.text_max_len))

    val_loader = DataLoader(val_dataset, batch_size=args.match_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False, drop_last=False, persistent_workers=True, collate_fn=TrainCollater(tokenizer, args.text_max_len))

    test_loader = DataLoader(test_dataset, batch_size=args.match_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False, drop_last=False, persistent_workers=True, collate_fn=TrainCollater(tokenizer, args.text_max_len))
    
    attscore_diffs_all = []
    text_lens_all = []

    for i, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
        batch = to_device(batch, device)
        graph, text, mask = batch
        batch_node, batch_mask = blip2qformer.graph_encoder(graph) # Graph Encoder in the paper
        if not blip2qformer.tune_gnn:
            batch_node = batch_node.detach()
        batch_size = batch_node.shape[0]

        batch_node = blip2qformer.ln_graph(batch_node, batch_mask) # [batch_size, 1+maximum_node_num, hidden_size 300]
        query_tokens = blip2qformer.query_tokens.expand(batch_node.shape[0], -1, -1)
        query_output = blip2qformer.Qformer.bert(
            query_embeds=query_tokens, # [batch_size, query_len, hidden_size 768]
            encoder_hidden_states=batch_node, # [batch_size, 1+maximum_node_num, hidden_size 300]
            encoder_attention_mask=batch_mask, # fixme: check whether this mask is correct | [batch_size, 1+maximum_node_num]
            use_cache=True,
            return_dict=True,
            output_attentions=True,
        )
        cross_attentions = query_output['cross_attentions'] # Cross attention and past key values are mixed in this tuple
        cross_attentions = [cross_attentions[i] for i in range(0, len(query_output['cross_attentions']), 2)]
        cross_attentions = torch.stack(cross_attentions).transpose(0, 1) # [batch_size, num_cross_attentions, num_heads, query_len, 1+maximum_node_num]
        mean_cross_attentions = cross_attentions.mean(dim=1).mean(1) # [batch_size, query_len, 1+maximum_node_num]
        attention_scores = mean_cross_attentions.mean(1)[:, 1:] # Drop the first token, which is graph representation token | [batch_size, maximum_node_num]

        if i < 10:
            # 1. Visualize attention heatmaps for all query tokens
            visualize_attention_heatmaps_in_one(mean_cross_attentions.cpu().detach().numpy(), i, analysis_root_dir)
            
            # 2. Visualize attention scores on molecule graphs
            visulize_molecule_graphs(attention_scores.cpu().detach().numpy(), graph, i, analysis_root_dir)
        
        # 3. Scatter plot of attention score differences and text lengths
        attscore_diffs, text_lens = scatter_attention_textlen(attention_scores.cpu().detach().numpy(), graph, i)
        attscore_diffs_all.extend(attscore_diffs)
        text_lens_all.extend(text_lens)
        
        
        
        if i == 100:
            break
    
    
    plt.scatter(text_lens_all, attscore_diffs_all)
    plt.xlabel("Text length")
    plt.ylabel("Attention score difference")
    plt.savefig(f"{analysis_root_dir}/scatter_plots/scatter_plot.png")









    print()




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, default="stage1_test")
    # GPU
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    # MM settings
    parser.add_argument('--gtm', action='store_true', help='use graph-text matching or not', default=True)
    parser.add_argument('--lm', action='store_true', help='use language modeling or not', default=True)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--devices', type=str, default='0')
    parser.add_argument('--precision', type=str, default='bf16-mixed')
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1)
    # parser.add_argument('--save_every_n_epochs', type=int, default=1)
    # parser = Trainer.add_argparse_args(parser)
    parser = Blip2Stage1.add_model_specific_args(parser)  # add model args
    parser = Stage1DM.add_model_specific_args(parser)
    # parser.set_defaults(accelerator='gpu',
    #                     devices='0,1,2,3',
    #                     precision='bf16',
    #                     max_epochs=50,
    #                     check_val_every_n_epoch=1)
    args = parser.parse_args()

    args.match_batch_size = 16
    args.root = "data/PubChem324kV2/"
    args.devices = "0"
    args.gtm = True
    args.lm = True
    args.mode = "train"
    args.filename = "stage1"
    args.rerank_cand_num = 128
    args.num_query_token = 8
    args.tune_gnn = True
    args.checkpoint = "all_checkpoints/stage1_keyword_random/last.ckpt"

    print("=========================================")
    for k, v in sorted(vars(args).items()):
        print(k, '=', v)
    print("=========================================")
    main(args)


