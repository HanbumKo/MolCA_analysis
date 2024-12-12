"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging
import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType, PeftModel
from ogb.utils import smiles2graph
from torch_geometric.loader.dataloader import Collater
from torch_geometric.data import Data
import numpy as np
from lavis.models.blip2_models.blip2 import (
    # Blip2Base,
    disabled_train,
)
from model.blip2 import Blip2Base
from transformers import AutoTokenizer
from transformers import OPTForCausalLM
# from opendelta import LoraModel
# from opendelta.delta_models.lora import LoraConfig
# from opendelta.delta_configs

opt_model_list = [
    "facebook/galactica-125m",
    "facebook/galactica-1.3b",
    "facebook/galactica-6.7b",
    "facebook/galactica-30b",
]

def mask_by_len(input, lens, fill_value=0):
    '''
    input: shape = [N, D]
    lens: shape = [N]
    '''
    mask = torch.arange(input.shape[1], device=input.device).reshape(1, -1)
    mask = mask < lens.reshape(-1, 1)
    input[mask] = fill_value
    return input


def smiles2data(smiles):
    graph = smiles2graph(smiles)
    x = torch.from_numpy(graph['node_feat'])
    edge_index = torch.from_numpy(graph['edge_index'], )
    edge_attr = torch.from_numpy(graph['edge_feat'])
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data

import re
SPLIT_MARKER = f"SPL{1}T-TH{1}S-Pl3A5E"

CUSTOM_SEQ_RE = re.compile(r"(\[START_(DNA|SMILES|I_SMILES|AMINO)])(.*?)(\[END_\2])")


def _insert_split_marker(m: re.Match):
    """
    Applies split marker based on a regex match of special tokens such as
    [START_DNA].

    Parameters
    ----------
    n : str
        Input text to split

    Returns
    ----------
    str - the text with the split token added
    """
    start_token, _, sequence, end_token = m.groups()
    sequence = re.sub(r"(.)", fr"{SPLIT_MARKER}\1", sequence, flags=re.DOTALL)
    return f"{start_token}{sequence}{SPLIT_MARKER}{end_token}"

def escape_custom_split_sequence(text):
    """
    Applies custom splitting to the text for GALILEO's tokenization

    Parameters
    ----------
    text : str
        Input text to split

    Returns
    ----------
    str - the text with the split token added
    """
    return CUSTOM_SEQ_RE.sub(_insert_split_marker, text)

def smiles_handler(text, mol_ph):
    smiles_list = []
    for match in CUSTOM_SEQ_RE.finditer(text):
        smiles = match.group(3)
        smiles_list.append(smiles)
    
    text = CUSTOM_SEQ_RE.sub(r'\1\3\4%s' % (mol_ph), text)
    text = escape_custom_split_sequence(text)
    return text, smiles_list


class Blip2OPT(Blip2Base):
    """
    BLIP2 first-stage model with Q-former and ViT.
    Supported model types:
        - pretrained: pretrained model with vit-g
        - pretrain_vitL: pretrained model with vit-large
        - coco: fintuned model on coco
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2", "pretrain")
    """
    def __init__(
        self,
        bert_name,
        gin_num_layers,
        gin_hidden_dim,
        gin_drop_ratio,
        tune_gnn=False,
        num_query_token=32,
        cross_attention_freq=2,
        llm_tune='freeze',
        peft_dir='',
        opt_model="facebook/galactica-1.3b",
        prompt="",
        args=None,
    ):
        super().__init__()
        self.args = args

        self.graph_encoder, self.ln_graph = self.init_graph_encoder(gin_num_layers, gin_hidden_dim, gin_drop_ratio, self.args.gnn_type)
        self.tune_gnn = tune_gnn
        if not tune_gnn:
            for name, param in self.graph_encoder.named_parameters():
                param.requires_grad = False
            self.graph_encoder = self.graph_encoder.eval()
            self.graph_encoder.train = disabled_train
            logging.info("freeze graph encoder")
        
        self.num_query_token = num_query_token
        
        if args.projector == 'qformer':
            self.Qformer, self.query_tokens = self.init_Qformer(bert_name, num_query_token, self.graph_encoder.num_features, cross_attention_freq)
            ### remove the unused parameters
            self.Qformer.cls = None
            self.Qformer.bert.embeddings.word_embeddings = None
            self.Qformer.bert.embeddings.position_embeddings = None
            for layer in self.Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
        elif args.projector == 'mlp':
            self.projector = nn.Sequential(
                nn.Linear(self.graph_encoder.num_features, self.args.bert_hidden_dim),
                nn.ReLU(),
                nn.Linear(self.args.bert_hidden_dim, self.args.bert_hidden_dim),
                nn.ReLU(),
                nn.Linear(self.args.bert_hidden_dim, self.args.bert_hidden_dim),
            ).to(torch.bfloat16)
        else:
            raise NotImplementedError("projector should be either 'qformer' or 'mlp'")

        ## initialize opt model
        self.opt_tokenizer = AutoTokenizer.from_pretrained(opt_model, use_fast=False, padding_side='right')
        self.opt_tokenizer.add_special_tokens({'pad_token': '<pad>'})
        self.opt_tokenizer.add_tokens('<mol>') # molecule placeholder
        self.mol_token = '<mol>'
        self.opt_tokenizer.mol_token_id = self.opt_tokenizer("<mol>", add_special_tokens=False).input_ids[0]

        self.collater = Collater([], [])
        
        if opt_model == 'facebook/galactica-125m':
            self.opt_model = OPTForCausalLM.from_pretrained(opt_model, torch_dtype=torch.bfloat16)
        else:
            if torch.cuda.is_bf16_supported():
                self.opt_model = OPTForCausalLM.from_pretrained(opt_model, torch_dtype=torch.bfloat16)
            else:
                self.opt_model = OPTForCausalLM.from_pretrained(opt_model, torch_dtype=torch.float16)
        self.opt_model.resize_token_embeddings(len(self.opt_tokenizer)) ## this will cause bug when full fine-tuning the opt model

        if args.stage2_path:
            ckpt = torch.load(args.stage2_path, map_location='cpu')
            new_state_dict = {}
            for key, val in ckpt['state_dict'].items():
                key = key.replace('blip2opt.', '')
                new_state_dict[key] = val
            self.load_state_dict(new_state_dict, strict=False)

        self.llm_tune = llm_tune
        if llm_tune == 'lora':
            if peft_dir:
                self.opt_model = PeftModel.from_pretrained(self.opt_model, peft_dir, is_trainable=True)
            else:
                if self.args.peft_config:
                    peft_config = LoraConfig(**LoraConfig.from_json_file(self.args.peft_config))
                else:
                    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout)
                self.peft_config = peft_config
                self.opt_model = get_peft_model(self.opt_model, peft_config)
                self.opt_model.print_trainable_parameters()
        elif llm_tune == 'freeze':
            for name, param in self.opt_model.named_parameters():
                param.requires_grad = False
        elif llm_tune == 'full':
            pass
        else:
            raise NotImplementedError()

        ## fixme: this is different from the original BLIP2
        self.eos_token_id = self.opt_tokenizer(
            "\n", add_special_tokens=False
        ).input_ids[0]

        if self.args.projector == 'qformer':
            self.opt_proj = nn.Linear(
                self.Qformer.config.hidden_size, self.opt_model.config.hidden_size
            )
        elif self.args.projector == 'mlp':
            self.opt_proj = nn.Linear(
                self.args.bert_hidden_dim, self.opt_model.config.hidden_size
            )
        if args.stage2_path:
            self.opt_proj.weight.data = new_state_dict['opt_proj.weight']
            self.opt_proj.bias.data = new_state_dict['opt_proj.bias']
            print(f"loaded stage2 model from {args.stage2_path}")
        
        ## fixme: no prompt yet
        self.prompt = prompt
        # prompt_tokens = self.opt_tokenizer(self.prompt, return_tensors="pt")
        # self.prompt_length = prompt_tokens.attention_mask.sum(1)

    def forward(self, batch):
        graphs, prompt_tokens, text_tokens = batch
        if len(graphs) == 0:
            device = prompt_tokens['input_ids'].device
        else:
            mol_tokens_list = self.forward_graph_list(graphs, prompt_tokens)
            device = prompt_tokens['input_ids'].device
        # elif self.args.root.lower().find('forward') >= 0: # forward reaction prediction
        #     mol_tokens_list = self.forward_graph_list(graphs, prompt_tokens)
        #     device = mol_tokens_list[0].device
        # elif self.args.root.lower().find('reagent_prediction') >= 0: # reagent prediction
        #     mol_tokens_list = self.forward_graph_list(graphs)
        #     device = mol_tokens_list[0].device
        # else:
        #     graph_embeds, graph_masks = self.graph_encoder(graphs) # graph_masks: (batch_size, maximum_num_node)
        #     if not self.tune_gnn:
        #         graph_embeds = graph_embeds.detach()
        #     graph_embeds = self.ln_graph(graph_embeds, graph_masks) # graph_embeds: (batch_size, maximum_num_node, 300)
        #     device = graph_embeds.device
        #     if self.args.projector == 'qformer':
        #         query_tokens = self.query_tokens.expand(graph_embeds.shape[0], -1, -1)
        #         query_output = self.Qformer.bert(
        #             query_embeds=query_tokens,
        #             encoder_hidden_states=graph_embeds,
        #             encoder_attention_mask=graph_masks, # fixme: check whether this mask is correct
        #             return_dict=True,
        #             output_attentions=True,
        #         )
        #         if self.args.query_index != -1:
        #             num_query = query_output.last_hidden_state.size(1)
        #             assert self.args.query_index < num_query, f"query_index should be less than {num_query}"
        #             new_hidden_state = torch.zeros_like(query_output.last_hidden_state)
        #             new_hidden_state[:, self.args.query_index, :] = query_output.last_hidden_state[:, self.args.query_index, :]
        #             query_output.last_hidden_state = new_hidden_state
        #         if self.args.shuffle_query: # Shuffle the query tokens between queries
        #             # query_output.last_hidden_state: (batch_size, num_query_token, D)
        #             print("shuffle query")
        #             query_output.last_hidden_state = query_output.last_hidden_state[:, torch.randperm(query_output.last_hidden_state.size(1)), :]
        #         if self.args.zero_query:
        #             print("zero query")
        #             query_output.last_hidden_state = torch.zeros_like(query_output.last_hidden_state)
        #         mol_tokens = self.opt_proj(query_output.last_hidden_state)
        #     elif self.args.projector == 'mlp':
        #         query_output = self.projector(graph_embeds)
        #         mol_tokens = self.opt_proj(query_output)
        #         prompt_tokens = self.expand_prompt_token(prompt_tokens, graph_masks)
        #     # mol_tokens = self.opt_proj(query_output.last_hidden_state)
        
        empty_targets = torch.ones(prompt_tokens.attention_mask.shape, dtype=torch.long).to(device).fill_(-100)
        targets = text_tokens.input_ids.masked_fill(
            text_tokens.input_ids == self.opt_tokenizer.pad_token_id, -100
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        # (prompt_tokens.input_ids == 22).nonzero(as_tuple=True)[1]

        prompt_embeds = self.opt_model.get_input_embeddings()(prompt_tokens.input_ids)
        if len(graphs) == 0:
            pass
        else:
            for i, mol_tokens in enumerate(mol_tokens_list):
                if mol_tokens is None:
                    continue
                prompt_embeds[i][prompt_tokens.is_mol_token[i]] = mol_tokens.flatten(0, 1)
                # prompt_embeds[prompt_tokens.is_mol_token] = torch.concat(mol_tokens_list, dim=0).flatten(0, 1).to(dtype=torch.bfloat16)
        # elif self.args.root.lower().find('forward') >= 0: # forward reaction prediction
        #     for i, mol_tokens in enumerate(mol_tokens_list):
        #         prompt_embeds[prompt_tokens.is_mol_token] = torch.concat(mol_tokens_list, dim=0).flatten(0, 1).to(dtype=torch.bfloat16)
        # elif self.args.root.lower().find('reagent_prediction') >= 0: # reagent prediction
        #     for i, mol_tokens in enumerate(mol_tokens_list):
        #         prompt_embeds[prompt_tokens.is_mol_token] = torch.concat(mol_tokens_list, dim=0).flatten(0, 1).to(dtype=torch.bfloat16)
        # else:
        #     if self.args.projector == 'qformer':
        #         prompt_embeds[prompt_tokens.is_mol_token] = mol_tokens.flatten(0, 1).to(dtype=torch.bfloat16)
        #     elif self.args.projector == 'mlp':
        #         for batch_idx in range(prompt_embeds.size(0)):
        #             prompt_embeds[batch_idx, prompt_tokens.is_mol_token[batch_idx]] = mol_tokens[batch_idx, graph_masks[batch_idx]].to(dtype=torch.bfloat16)
        inputs_embeds = self.opt_model.get_input_embeddings()(text_tokens.input_ids)
        inputs_embeds = torch.cat((prompt_embeds, inputs_embeds), dim=1)
        attention_mask = torch.cat([prompt_tokens.attention_mask, text_tokens.attention_mask], dim=1)
        
        outputs = self.opt_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
            output_attentions=True,
        )
        loss = outputs.loss
        generation_attentions = outputs.attentions # (batch_size, num_heads, num_query_token, num_text_token)
        target_start_idx = (targets != -100).nonzero(as_tuple=True)[1].min().item() # Same for all samples in the batch
        
        
        # ###============== Attention score regularization ===================###
        # att_cos = 0
        # att_kl = 0
        # att_l2 = 0
        # cross_attentions = query_output['cross_attentions'] # Cross attention and past key values are mixed in this tuple
        # cross_attentions = [cross_attentions[i] for i in range(0, len(query_output['cross_attentions']), 2)]
        # cross_attentions = torch.stack(cross_attentions).transpose(0, 1) # [batch_size, num_cross_attentions, num_heads, query_len, 1+maximum_node_num]
        # mean_cross_attentions = cross_attentions.mean(dim=1).mean(1) # [batch_size, query_len, 1+maximum_node_num]
        # masked_feat = mean_cross_attentions * graph_masks.unsqueeze(1).float()

        # att_cos = self.compute_avg_cosine_similarity(masked_feat)
        # att_kl = self.compute_average_symmetrised_kl_divergence(masked_feat)
        # att_l2 = self.compute_mean_l2_distance(masked_feat)
        
        return {
            "loss": loss,
            # "cross_attentions": query_output.cross_attentions,
            "generation_attentions": generation_attentions,
            "target_start_idx": target_start_idx,
            # "att_cos": att_cos.mean(),
            # "att_kl": att_kl.mean(),
            # "att_l2": att_l2.mean(),
        }

    def forward_graph_list(self, graph_list_all, prompt_tokens=None):
        mol_tokens_list = []
        for i, graph_list in enumerate(graph_list_all):
            if len(graph_list) == 0:
                # Append empty tensor
                mol_tokens_list.append(None)
                continue
            graphs = self.collater(graph_list)
            graph_embeds, graph_masks = self.graph_encoder(graphs)
            if not self.tune_gnn:
                graph_embeds = graph_embeds.detach()
            graph_embeds = self.ln_graph(graph_embeds, graph_masks)
            device = graph_embeds.device
            if self.args.projector == 'qformer':
                query_tokens = self.query_tokens.expand(graph_embeds.shape[0], -1, -1)
                query_output = self.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=graph_embeds,
                    encoder_attention_mask=graph_masks, # fixme: check whether this mask is correct
                    return_dict=True,
                    output_attentions=True,
                )
                if self.args.query_index != -1:
                    num_query = query_output.last_hidden_state.size(1)
                    assert self.args.query_index < num_query, f"query_index should be less than {num_query}"
                    new_hidden_state = torch.zeros_like(query_output.last_hidden_state)
                    new_hidden_state[:, self.args.query_index, :] = query_output.last_hidden_state[:, self.args.query_index, :]
                    query_output.last_hidden_state = new_hidden_state
                if self.args.shuffle_query: # Shuffle the query tokens between queries
                    # query_output.last_hidden_state: (batch_size, num_query_token, D)
                    print("shuffle query")
                    query_output.last_hidden_state = query_output.last_hidden_state[:, torch.randperm(query_output.last_hidden_state.size(1)), :]
                if self.args.zero_query:
                    print("zero query")
                    query_output.last_hidden_state = torch.zeros_like(query_output.last_hidden_state)
                mol_tokens = self.opt_proj(query_output.last_hidden_state)
            elif self.args.projector == 'mlp':
                query_output = self.projector(graph_embeds)
                mol_tokens = self.opt_proj(query_output)
                prompt_tokens_batch = self.expand_prompt_token_multiple(prompt_tokens, graph_masks, i)
            mol_tokens_list.append(mol_tokens)
        return mol_tokens_list

    @torch.no_grad()
    def generate(
        self,
        samples,
        do_sample=False,
        num_beams=5,
        max_length=128,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        graphs = samples['graphs']
        prompt_tokens = samples['prompt_tokens']
        # prompt_lens = samples['prompt_lens']
        # with self.maybe_autocast():
        if self.args.root.lower().find('forward') >= 0: # forward reaction prediction
            mol_tokens_list = self.forward_graph_list(graphs)
            device = mol_tokens_list[0].device
        elif self.args.root.lower().find('reagent_prediction') >= 0: # reagent prediction
            mol_tokens_list = self.forward_graph_list(graphs)
            device = mol_tokens_list[0].device
        else:
            graph_embeds, graph_masks = self.graph_encoder(graphs)
            graph_embeds = self.ln_graph(graph_embeds)

            device = graph_embeds.device
            if self.args.projector == 'qformer':
                query_tokens = self.query_tokens.expand(graph_embeds.shape[0], -1, -1)
                query_output = self.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=graph_embeds,
                    encoder_attention_mask=graph_masks, # fixme: check whether this mask is correct
                    return_dict=True,
                    output_attentions=True,
                )
                if self.args.query_index != -1:
                    num_query = query_output.last_hidden_state.size(1)
                    assert self.args.query_index < num_query, f"query_index should be less than {num_query}"
                    new_hidden_state = torch.zeros_like(query_output.last_hidden_state)
                    new_hidden_state[:, self.args.query_index, :] = query_output.last_hidden_state[:, self.args.query_index, :]
                    query_output.last_hidden_state = new_hidden_state
                if self.args.shuffle_query: # Shuffle the query tokens between queries
                    # query_output.last_hidden_state: (batch_size, num_query_token, D)
                    print("shuffle query")
                    query_output.last_hidden_state = query_output.last_hidden_state[:, torch.randperm(query_output.last_hidden_state.size(1)), :]
                if self.args.zero_query:
                    print("zero query")
                    query_output.last_hidden_state = torch.zeros_like(query_output.last_hidden_state)
                mol_tokens = self.opt_proj(query_output.last_hidden_state)
            elif self.args.projector == 'mlp':
                query_output = self.projector(graph_embeds)
                mol_tokens = self.opt_proj(query_output)
                prompt_tokens = self.expand_prompt_token(prompt_tokens, graph_masks)
        
        prompt_embeds = self.opt_model.get_input_embeddings()(prompt_tokens.input_ids)
        if self.args.root.lower().find('forward') >= 0: # forward reaction prediction
            for i, mol_tokens in enumerate(mol_tokens_list):
                prompt_embeds[prompt_tokens.is_mol_token] = torch.concat(mol_tokens_list, dim=0).flatten(0, 1)
        elif self.args.root.lower().find('reagent_prediction') >= 0: # reagent prediction
            for i, mol_tokens in enumerate(mol_tokens_list):
                prompt_embeds[prompt_tokens.is_mol_token] = torch.concat(mol_tokens_list, dim=0).flatten(0, 1).to(dtype=torch.bfloat16)
        else:
            if self.args.projector == 'qformer':
                prompt_embeds[prompt_tokens.is_mol_token] = mol_tokens.flatten(0, 1).to(dtype=torch.bfloat16)
            elif self.args.projector == 'mlp':
                for batch_idx in range(prompt_embeds.size(0)):
                    prompt_embeds[batch_idx, prompt_tokens.is_mol_token[batch_idx]] = mol_tokens[batch_idx, graph_masks[batch_idx]].to(dtype=torch.bfloat16)


        outputs = self.opt_model.generate(
            inputs_embeds=prompt_embeds,
            attention_mask=prompt_tokens.attention_mask,
            do_sample=do_sample,
            top_p=top_p,
            temperature=temperature,
            num_beams=num_beams,
            max_length=max_length,
            min_length=min_length,
            # pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            num_return_sequences=num_captions,
            # use_cache=False,
        )
        output_text = self.opt_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        output_text = [text.strip() for text in output_text]
        return output_text

    def expand_prompt_token(self, prompt_tokens, graph_masks):
        # Expand the mol tokens to match the number of nodes in the graph, the number of graph is different for each sample
        # prompt_tokens: {input_ids: (batch_size, prompt_len), token_type_ids: (batch_size, prompt_len), attention_mask: (batch_size, prompt_len), is_mol_token: (batch_size, prompt_len)}
        # graph_masks: (batch_size, maximum_num_node)
        batch_size, prompt_len = prompt_tokens.input_ids.shape
        device = prompt_tokens.input_ids.device
        
        # Calculate number of nodes in each graph
        node_counts = graph_masks.sum(dim=1) # shape: (batch_size)

        expanded_input_ids = []
        expanded_attention_mask = []
        expanded_token_type_ids = []
        expanded_is_mol_token = []
        
        for batch_idx in range(batch_size):
            # Find the start and end indices of the '1' values in the current batch
            mol_start_idx = (prompt_tokens.is_mol_token[batch_idx] == 1).nonzero(as_tuple=True)[0][0] # 첫 번째 '1'의 시작 인덱스
            mol_end_idx = (prompt_tokens.is_mol_token[batch_idx] == 1).nonzero(as_tuple=True)[0][-1] + 1 # 마지막 '1'의 끝 인덱스 + 1

            # Expand the '1' section to match the number of nodes in the graph
            num_nodes = node_counts[batch_idx].item()

            expanded_section = torch.ones(num_nodes).to(device) * 50000
            new_input_ids = torch.cat([
                prompt_tokens.input_ids[batch_idx][:mol_start_idx], # Prior to '1'
                expanded_section.int(), # Expanded '1' part
                prompt_tokens.input_ids[batch_idx][mol_end_idx:] # After '1'
            ])
            
            expanded_section = torch.ones(num_nodes).to(device)
            new_attention_mask = torch.cat([
                prompt_tokens.attention_mask[batch_idx][:mol_start_idx], # Prior to '1'
                expanded_section.int(), # Expanded '1' part
                prompt_tokens.attention_mask[batch_idx][mol_end_idx:] # After '1'
            ])
            
            expanded_section = torch.zeros(num_nodes).to(device)
            new_token_type_ids = torch.cat([
                prompt_tokens.token_type_ids[batch_idx][:mol_start_idx], # Prior to '1'
                expanded_section.int(), # Expanded '1' part
                prompt_tokens.token_type_ids[batch_idx][mol_end_idx:] # After '1'
            ])

            expanded_section = torch.ones(num_nodes).to(device) # Generate '1' for the number of nodes
            # Concatenate the original is_mol_token tensor with the expanded section
            new_is_mol_token = torch.cat([
                prompt_tokens.is_mol_token[batch_idx][:mol_start_idx], # Prior to '1'
                expanded_section.int(), # Expanded '1' part
                prompt_tokens.is_mol_token[batch_idx][mol_end_idx:] # After '1'
            ])

            expanded_input_ids.append(new_input_ids)
            expanded_attention_mask.append(new_attention_mask)
            expanded_token_type_ids.append(new_token_type_ids)
            expanded_is_mol_token.append(new_is_mol_token)

        max_length = max(tensor.size(0) for tensor in expanded_input_ids)
        padded_input_ids = [
            F.pad(tensor, (max_length - tensor.size(0), 0), mode='constant', value=1) for tensor in expanded_input_ids
        ]
        padded_attention_mask = [
            F.pad(tensor, (max_length - tensor.size(0), 0), mode='constant', value=0) for tensor in expanded_attention_mask
        ]
        padded_token_type_ids = [
            F.pad(tensor, (max_length - tensor.size(0), 0), mode='constant', value=0) for tensor in expanded_token_type_ids
        ]
        padded_is_mol_token = [
            F.pad(tensor, (max_length - tensor.size(0), 0), mode='constant', value=0) for tensor in expanded_is_mol_token
        ]

        prompt_tokens.input_ids = torch.stack(padded_input_ids)
        prompt_tokens.attention_mask = torch.stack(padded_attention_mask)
        prompt_tokens.token_type_ids = torch.stack(padded_token_type_ids)
        prompt_tokens.is_mol_token = torch.stack(padded_is_mol_token).bool()
        
        prompt_tokens['input_ids'] = torch.stack(padded_input_ids)
        prompt_tokens['attention_mask'] = torch.stack(padded_attention_mask)
        prompt_tokens['token_type_ids'] = torch.stack(padded_token_type_ids)
        prompt_tokens['is_mol_token'] = torch.stack(padded_is_mol_token)

        return prompt_tokens

    def expand_prompt_token_multiple(self, prompt_tokens, graph_masks, batch_i):
        # Expand the mol tokens to match the number of nodes in the graph, the number of graph is different for each sample
        # prompt_tokens: {input_ids: (batch_size, prompt_len), token_type_ids: (batch_size, prompt_len), attention_mask: (batch_size, prompt_len), is_mol_token: (batch_size, prompt_len)}
        # graph_masks: (batch_size, maximum_num_node)
        batch_size, _ = graph_masks.shape
        device = prompt_tokens.input_ids.device
        
        # Calculate number of nodes in each graph
        node_counts = graph_masks.sum(dim=1) # shape: (batch_size)
        
        current_input_ids = prompt_tokens.input_ids[batch_i]
        current_attention_mask = prompt_tokens.attention_mask[batch_i]
        current_token_type_ids = prompt_tokens.token_type_ids[batch_i]
        current_is_mol_token = prompt_tokens.is_mol_token[batch_i]

        expanded_input_ids = []
        expanded_attention_mask = []
        expanded_token_type_ids = []
        expanded_is_mol_token = []
        
        false_lenghts = self.count_false_sequences(current_is_mol_token)
        
        for batch_idx in range(batch_size):
            # Find the start and end indices of the '1' values in the current batch
            # mol_start_idx = (prompt_tokens.is_mol_token[batch_idx] == 1).nonzero(as_tuple=True)[0][0] # 첫 번째 '1'의 시작 인덱스
            # mol_end_idx = (prompt_tokens.is_mol_token[batch_idx] == 1).nonzero(as_tuple=True)[0][-1] + 1 # 마지막 '1'의 끝 인덱스 + 1
            current_is_mol_token = prompt_tokens.is_mol_token[batch_i]

            # Expand the '1' section to match the number of nodes in the graph
            num_nodes = node_counts[batch_idx].item()
            
            one_indices = (current_is_mol_token == 1).nonzero(as_tuple=True)[0]
            expanded_tensor = torch.zeros_like(current_is_mol_token)
            if len(one_indices) > 0:
                # 연속된 구간 계산
                start_idx = one_indices[0]
                for i in range(1, len(one_indices)):
                    if one_indices[i] != one_indices[i - 1] + 1: # 연속이 끊긴 경우
                        end_idx = one_indices[i - 1] + 1
                        expanded_tensor[start_idx:start_idx + num_nodes] = 1
                        start_idx = one_indices[i]

                # 마지막 구간 처리
                expanded_tensor[start_idx:start_idx + num_nodes] = 1

            expanded_is_mol_token.append(expanded_tensor)
            

            expanded_section = torch.ones(num_nodes).to(device) * 50000
            new_input_ids = torch.cat([
                prompt_tokens.input_ids[batch_idx][:mol_start_idx], # Prior to '1'
                expanded_section.int(), # Expanded '1' part
                prompt_tokens.input_ids[batch_idx][mol_end_idx:] # After '1'
            ])
            
            expanded_section = torch.ones(num_nodes).to(device)
            new_attention_mask = torch.cat([
                prompt_tokens.attention_mask[batch_idx][:mol_start_idx], # Prior to '1'
                expanded_section.int(), # Expanded '1' part
                prompt_tokens.attention_mask[batch_idx][mol_end_idx:] # After '1'
            ])
            
            expanded_section = torch.zeros(num_nodes).to(device)
            new_token_type_ids = torch.cat([
                prompt_tokens.token_type_ids[batch_idx][:mol_start_idx], # Prior to '1'
                expanded_section.int(), # Expanded '1' part
                prompt_tokens.token_type_ids[batch_idx][mol_end_idx:] # After '1'
            ])

            expanded_section = torch.ones(num_nodes).to(device) # Generate '1' for the number of nodes
            # Concatenate the original is_mol_token tensor with the expanded section
            new_is_mol_token = torch.cat([
                prompt_tokens.is_mol_token[batch_idx][:mol_start_idx], # Prior to '1'
                expanded_section.int(), # Expanded '1' part
                prompt_tokens.is_mol_token[batch_idx][mol_end_idx:] # After '1'
            ])

            expanded_input_ids.append(new_input_ids)
            expanded_attention_mask.append(new_attention_mask)
            expanded_token_type_ids.append(new_token_type_ids)
            expanded_is_mol_token.append(new_is_mol_token)

        max_length = max(tensor.size(0) for tensor in expanded_input_ids)
        padded_input_ids = [
            F.pad(tensor, (max_length - tensor.size(0), 0), mode='constant', value=1) for tensor in expanded_input_ids
        ]
        padded_attention_mask = [
            F.pad(tensor, (max_length - tensor.size(0), 0), mode='constant', value=0) for tensor in expanded_attention_mask
        ]
        padded_token_type_ids = [
            F.pad(tensor, (max_length - tensor.size(0), 0), mode='constant', value=0) for tensor in expanded_token_type_ids
        ]
        padded_is_mol_token = [
            F.pad(tensor, (max_length - tensor.size(0), 0), mode='constant', value=0) for tensor in expanded_is_mol_token
        ]

        prompt_tokens.input_ids = torch.stack(padded_input_ids)
        prompt_tokens.attention_mask = torch.stack(padded_attention_mask)
        prompt_tokens.token_type_ids = torch.stack(padded_token_type_ids)
        prompt_tokens.is_mol_token = torch.stack(padded_is_mol_token).bool()
        
        prompt_tokens['input_ids'] = torch.stack(padded_input_ids)
        prompt_tokens['attention_mask'] = torch.stack(padded_attention_mask)
        prompt_tokens['token_type_ids'] = torch.stack(padded_token_type_ids)
        prompt_tokens['is_mol_token'] = torch.stack(padded_is_mol_token)

        return prompt_tokens

    def count_false_sequences(self, bool_tensor):
        # Convert to integers (False -> 0, True -> 1)
        int_tensor = bool_tensor.to(dtype=torch.int)
        
        # Compute the difference between consecutive elements
        diff = torch.diff(int_tensor, prepend=torch.tensor([1]))
        
        # Identify the start and end of False sequences
        false_starts = (diff == -1).nonzero(as_tuple=True)[0]
        false_ends = (diff == 1).nonzero(as_tuple=True)[0]
        
        # If the tensor ends with False, append the last index + 1 to false_ends
        if len(false_ends) == 0 or false_starts[-1] > false_ends[-1]:
            false_ends = torch.cat([false_ends, torch.tensor([len(bool_tensor)])])
        
        # Calculate lengths of each False sequence
        false_lengths = (false_ends - false_starts).tolist()
        
        return false_lengths

    def compute_avg_cosine_similarity(self, X):
        # X has shape (batch_size, num_query, feature_dim)
        # Normalize the vectors along the feature_dim dimension
        normalized_X = X / X.norm(dim=-1, keepdim=True)

        # Compute cosine similarity matrices
        cos_sim_matrix = torch.bmm(normalized_X, normalized_X.transpose(1, 2))  # Shape: (batch_size, num_query, num_query)

        # Exclude self-similarities by creating a mask
        batch_size, num_query, _ = cos_sim_matrix.shape
        mask = torch.eye(num_query, device=X.device).bool().unsqueeze(0).expand(batch_size, -1, -1)
        cos_sim_matrix = cos_sim_matrix.masked_fill(mask, 0)

        # Compute the average cosine similarity over all pairs excluding self-pairs
        sum_cos_sim = cos_sim_matrix.sum(dim=(1, 2))  # Shape: (batch_size,)
        num_pairs = num_query * (num_query - 1)
        avg_cos_sim = sum_cos_sim / num_pairs

        return avg_cos_sim  # Shape: (batch_size,)

    def compute_average_symmetrised_kl_divergence(self, x):
        """
        Computes the average symmetrised KL divergence between all pairs of queries
        in a tensor of shape (batch_size, num_query, feature_dim).

        Parameters:
        x (torch.Tensor): Input tensor of shape (batch_size, num_query, feature_dim).

        Returns:
        torch.Tensor: Tensor of shape (batch_size) containing the average symmetrised KL divergence
                    for each batch.
        """
        batch_size, num_query, feature_dim = x.shape

        # Ensure the distributions sum to 1 along the feature_dim
        x = F.softmax(x, dim=-1)  # Shape: (batch_size, num_query, feature_dim)

        # Create tensors for all pairs of queries
        x1 = x.unsqueeze(2)  # Shape: (batch_size, num_query, 1, feature_dim)
        x2 = x.unsqueeze(1)  # Shape: (batch_size, 1, num_query, feature_dim)

        # Broadcast x1 and x2 to shape (batch_size, num_query, num_query, feature_dim)
        # Compute KL divergences D(x1 || x2) and D(x2 || x1)
        kl1 = (x1 * (x1.log() - x2.log())).sum(dim=-1)  # Shape: (batch_size, num_query, num_query)
        kl2 = (x2 * (x2.log() - x1.log())).sum(dim=-1)

        # Compute symmetrised KL divergence
        skl = kl1 + kl2  # Shape: (batch_size, num_query, num_query)

        # Create a mask to exclude self-divergences (diagonal elements)
        mask = 1 - torch.eye(num_query, device=x.device).unsqueeze(0)  # Shape: (1, num_query, num_query)
        mask = mask.expand(batch_size, -1, -1)  # Shape: (batch_size, num_query, num_query)

        # Apply the mask
        skl = skl * mask

        # Compute the average over all pairs (excluding self-pairs)
        num_pairs = num_query * (num_query - 1)
        skl_mean = skl.sum(dim=(1, 2)) / num_pairs  # Shape: (batch_size)

        return skl_mean

    def compute_mean_l2_distance(self, x):
        """
        x: Tensor of shape (batch_size, num_query, feature_dim)
        Returns a Tensor of shape (batch_size,) containing the mean L2 distance between all pairs of queries.
        """
        batch_size, num_query, feature_dim = x.shape

        # Compute squared norms of each query
        x_norm_squared = (x ** 2).sum(dim=2)  # Shape: (batch_size, num_query)

        # Compute the dot product between all pairs of queries
        prod = x @ x.transpose(1, 2)  # Shape: (batch_size, num_query, num_query)

        # Compute squared distances using the formula: ||u - v||^2 = ||u||^2 - 2*u^T*v + ||v||^2
        dist_squared = x_norm_squared.unsqueeze(2) - 2 * prod + x_norm_squared.unsqueeze(1)  # Shape: (batch_size, num_query, num_query)

        # Take the square root to get L2 distances
        dist = torch.sqrt(dist_squared + 1e-12)  # Add a small value to avoid sqrt(0)

        # Create a mask to exclude self-distances (diagonal elements)
        mask = ~torch.eye(num_query, device=x.device).bool().unsqueeze(0)  # Shape: (1, num_query, num_query)

        # Apply the mask to zero out diagonal elements
        dist = dist * mask.float()

        # Sum over all distances and compute the mean
        sum_dist = dist.sum(dim=(1, 2))  # Shape: (batch_size,)
        num_pairs = num_query * (num_query - 1)
        mean_dist = sum_dist / num_pairs

        return mean_dist
