import os
import argparse
import warnings
import torch
import matplotlib.pyplot as plt

from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdMolDescriptors
from rdkit import Chem
from io import BytesIO
from PIL import Image
from tqdm import tqdm
from data_provider.stage2_dm import Stage2DM, TrainCollater
from data_provider.stage2_chebi_dm import Stage2CheBIDM
from model.blip2_stage2 import Blip2Stage2
import matplotlib.ticker as ticker
from bertviz import head_view, model_view


## for pyg bug
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
## for A5000 gpus
torch.set_float32_matmul_precision('medium') # can be medium (bfloat16), high (tensorfloat32), highest (float32)


def to_device(data, device, tokenizer):
    # data[0] = data[0].to(device)
    data[0].x = data[0].x.to(device, dtype=torch.long)
    data[0].edge_index = data[0].edge_index.to(device, dtype=torch.long)
    data[0].edge_attr = data[0].edge_attr.to(device, dtype=torch.long)
    data[0].batch = data[0].batch.to(device, dtype=torch.long)
    for k, v in data[1].items():
        data[1][k] = v.to(device)
    if type(data[2]) == tuple: # Assume tuple of texts
        data = list(data)
        data[2] = tokenizer.batch_encode_plus(data[2], padding=True, return_tensors='pt')
        data = tuple(data)
    for k, v in data[2].items():
        data[2][k] = v.to(device)
    # data_device = tuple([d.to(device, dtype=torch.bfloat16) for d in data])
    return data


# Function to map a value to a color
def value_to_color(value):
    return (1.0 - value, 1.0, 1.0 - value) # The higher the value, the closer to


def visualize_cross_attention_matplot(attention_maps, graphs, idx, root_dir, checkpoint_name):
    """
    attention_maps: numpy array of shape [batch_size, num_queries, num_keys]
    Visualizes all attention maps in one figure with subplots.
    """
    batch_size, num_queries, num_keys = attention_maps.shape
    for i in range(batch_size):
        graph = graphs[i]
        mol = Chem.MolFromSmiles(graph.smiles)
        symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
        instance_i = idx * batch_size + i
        attention_map = attention_maps[i]
        attention_map = attention_map[:, attention_map[0] != 0] # Drop zero values
        plt.figure(figsize=(3, 3))
        heatmap = plt.imshow(attention_map, cmap='viridis', aspect='auto', vmin=0, vmax=0.2)
        cbar = plt.colorbar(heatmap)
        cbar.set_label('Attention score', fontsize=6)
        cbar.ax.tick_params(labelsize=6)
        plt.title(f'{checkpoint_name}, instance #{instance_i}', fontsize=8)
        plt.xlabel('Graph node atom', fontsize=6)
        plt.ylabel('Query index', fontsize=6)
        plt.tick_params(axis='both', which='major', labelsize=6)
        ax = plt.gca()
        ax.set_xticks(ticks=list(range(len(symbols)+1)), labels=["g"] + symbols, fontsize=6)
        plt.tight_layout()
        plt.savefig(f"{root_dir}/cross_attention_maps/attention_maps_{instance_i}.png", dpi=300)
        plt.close()


def visualize_cross_attention_bertviz(cross_attention_maps, graphs, idx, root_dir, checkpoint_name):
    batch_size, num_layers, num_heads, num_queries, _ = cross_attention_maps.shape
    for i in range(batch_size):
        graph = graphs[i]
        mol = Chem.MolFromSmiles(graph.smiles)
        symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
        instance_i = idx * batch_size + i
        cross_attention_map = cross_attention_maps[i:i+1, :, :, :, cross_attention_maps[i, 0, 0, 0]!=0] # Drop zero values
        num_keys = cross_attention_map.shape[-1]
        num_nodes = num_keys - 1
        bertviz_cross_attention = list(cross_attention_map.transpose(0, 1))
        decoder_tokens = [f"{i}" for i in range(num_queries)]
        encoder_tokens = ["GR"] + symbols
        html_obj_headview = head_view(cross_attention=bertviz_cross_attention, decoder_tokens=decoder_tokens, encoder_tokens=encoder_tokens, html_action="return")
        html_obj_modelview = model_view(cross_attention=bertviz_cross_attention, decoder_tokens=decoder_tokens, encoder_tokens=encoder_tokens, display_mode='light', html_action="return")
        with open(f"{root_dir}/cross_attention_headview/{instance_i}.html", "w", encoding="utf-8") as f:
            f.write(html_obj_headview.data)
        with open(f"{root_dir}/cross_attention_modelview/{instance_i}.html", "w", encoding="utf-8") as f:
            f.write(html_obj_modelview.data)


def visulize_molecule_graphs(attention_scores, graphs, idx, root_dir, checkpoint_name):
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


def visualize_generation_attention_heatmaps(tokenizer, gen_attentions, idx, batch, target_start_idx, root_dir):
    _, prompt_tokens, text_tokens = batch
    batch_size = gen_attentions.size(0)
    for i in range(batch_size):
        instance_i = idx * batch_size + i
        prompt_token_ids = prompt_tokens.input_ids[i]
        text_token_ids = text_tokens.input_ids[i]
        
        # Get the first non-1 token index in the prompt tokens
        prompt_start_idx = (prompt_token_ids != 1).nonzero(as_tuple=True)[0].min().item()
        text_end_idx = (text_token_ids != 1).nonzero(as_tuple=True)[0].max().item()
        # gen_attention = gen_attentions[i, :text_end_idx+1, prompt_start_idx:] # [output_len, input_len + output_len]
        gen_attention = gen_attentions[i, :text_end_idx+1, prompt_start_idx:target_start_idx] # [output_len, input_len]

        input_text = tokenizer.decode(prompt_token_ids[prompt_start_idx:])
        gen_text = tokenizer.decode(text_token_ids[:text_end_idx+1])
        input_text_list = tokenizer.convert_ids_to_tokens(prompt_token_ids[prompt_start_idx:])
        gen_text_list = tokenizer.convert_ids_to_tokens(text_token_ids[:text_end_idx+1])
        input_text_list = [t.replace('Ġ', '') for t in input_text_list]
        gen_text_list = [t.replace('Ġ', '') for t in gen_text_list]
        
        max_text_len = 30
        gen_attention = gen_attention[:max_text_len, :]
        gen_text_list = gen_text_list[:max_text_len]

        plt.figure(figsize=(6, 6))
        # heatmap = plt.imshow(gen_attention.transpose(0, 1).cpu().float().detach().numpy(), cmap='viridis', aspect='auto')
        heatmap = plt.imshow(gen_attention[:, :].transpose(0, 1).cpu().float().detach().numpy(), cmap='viridis', aspect='auto')
        cbar = plt.colorbar(heatmap)
        cbar.set_label('Attention score', fontsize=6)
        plt.title(f'Instance #{instance_i}', fontsize=8)
        plt.xlabel("Generated text tokens", fontsize=6)
        plt.ylabel("Prompt tokens", fontsize=6)
        # Set xticks as gen_text_list
        plt.xticks(range(len(gen_text_list)), gen_text_list, fontsize=3, rotation=90)
        plt.yticks(range(len(input_text_list)), input_text_list, fontsize=6)
        plt.tight_layout()
        plt.savefig(f"{root_dir}/gen_attention_maps/attention_maps_{instance_i}.png", dpi=300)
        plt.close()
        
        
    


def main(args):
    model = Blip2Stage2(args)
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'], strict=False)

    # model = Blip2Stage2.load_from_checkpoint(args.checkpoint, strict=False, args=args, map_location="cuda")
    model.eval()
    # model.to(torch.bfloat16).to('cuda')
    model.to('cuda')

    checkpoint_name = args.checkpoint.split("/")[-2]
    analysis_root_dir = f"analysis_results/{checkpoint_name}"
    analysis_dirs = [
        f"{analysis_root_dir}/cross_attention_headview",
        f"{analysis_root_dir}/cross_attention_modelview",
        f"{analysis_root_dir}/cross_attention_maps",
        f"{analysis_root_dir}/gen_attention_maps",
        f"{analysis_root_dir}/molecule_highlights",
        f"{analysis_root_dir}/scatter_plots",
    ]
    os.makedirs("analysis_results", exist_ok=True)
    os.makedirs(analysis_root_dir, exist_ok=True)
    for analysis_dir in analysis_dirs:
        os.makedirs(analysis_dir, exist_ok=True)
    
    blip2opt = model.blip2opt
    tokenizer = model.blip2opt.opt_tokenizer
    device = blip2opt.opt_proj.bias.device
    
    if args.root.lower().find('chebi') >= 0:
        dm = Stage2CheBIDM(args.mode, args.num_workers, args.batch_size, args.root, args.text_max_len, tokenizer, args)
    else:
        dm = Stage2DM(args.mode, args.num_workers, args.batch_size, args.root, args.text_max_len, tokenizer, args)
    
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()

    attscore_diffs_all = []
    text_lens_all = []

    for i, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
        batch = to_device(batch, device, tokenizer)
        graph, prompt_tokens, text_tokens = batch
        res = blip2opt(batch)
        cross_attentions = res['cross_attentions']
        target_start_idx = res['target_start_idx']
        gen_attentions = torch.stack(res['generation_attentions']).mean(0).mean(1) # [batch_size, query_len, key_len], Mean over all attention layers and heads
        gen_attentions = gen_attentions[:, target_start_idx:, :] # Only consider the attention scores from the generated text to the input SMILES and query tokens

        cross_attentions = [cross_attentions[i] for i in range(0, len(cross_attentions), 2)]
        cross_attentions = torch.stack(cross_attentions).transpose(0, 1) # [batch_size, num_cross_attentions, num_heads, query_len, 1+maximum_node_num]
        mean_cross_attentions = cross_attentions.mean(dim=1).mean(1) # [batch_size, query_len, 1+maximum_node_num]
        attention_scores = mean_cross_attentions.mean(1)[:, 1:] # Drop the first token, which is graph representation token | [batch_size, maximum_node_num]
        
        

        if i < 10:
            # 1. Visualize attention heatmaps for all query tokens
            visualize_cross_attention_matplot(mean_cross_attentions.cpu().detach().numpy(), graph, i, analysis_root_dir, checkpoint_name)

            # 2. Visualize attention scores on molecule graphs
            visulize_molecule_graphs(attention_scores.cpu().detach().numpy(), graph, i, analysis_root_dir, checkpoint_name)
            
            # visualize_generation_attention_heatmaps(tokenizer, gen_attentions, i, batch, target_start_idx, analysis_root_dir)

            visualize_cross_attention_bertviz(cross_attentions.cpu().detach(), graph, i, analysis_root_dir, checkpoint_name)

        # 3. Scatter plot of attention score differences and text lengths
        attscore_diffs, text_lens = scatter_attention_textlen(attention_scores.cpu().detach().numpy(), batch[0], i)
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
    parser.add_argument('--filename', type=str, default="stage2_test")
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    # MM settings
    parser.add_argument('--mode', type=str, default='pretrain')
    parser.add_argument('--strategy_name', type=str, default=None)
    parser.add_argument('--iupac_prediction', action='store_true', default=False)
    parser.add_argument('--ckpt_path', type=str, default=None)
    # parser = Trainer.add_argparse_args(parser)
    parser = Blip2Stage2.add_model_specific_args(parser)  # add model args
    parser = Stage2DM.add_model_specific_args(parser)
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--devices', type=str, default='0')
    parser.add_argument('--precision', type=str, default='bf16-mixed')
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1)
    args = parser.parse_args()

    args.batch_size = 16
    args.root = "data/PubChem324kV2/"
    args.devices = "0"
    args.filename = "stage2"
    args.checkpoint = "all_checkpoints/stage2_Extended/last.ckpt"
    args.opt_model = "facebook/galactica-1.3b"
    args.max_epochs = 10
    args.mode = "pretrain"
    args.prompt = "[START_I_SMILES]{}[END_I_SMILES]."
    # args.prompt = "Here is a SMILES formula:\n[START_I_SMILES]{}[END_I_SMILES]."
    args.tune_gnn = True
    args.llm_tune = "lora"
    args.inference_batch_size = 16

    print("=========================================")
    for k, v in sorted(vars(args).items()):
        print(k, '=', v)
    print("=========================================")
    main(args)


