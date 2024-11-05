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

        self.graph_encoder, self.ln_graph = self.init_graph_encoder(gin_num_layers, gin_hidden_dim, gin_drop_ratio)
        self.tune_gnn = tune_gnn
        if not tune_gnn:
            for name, param in self.graph_encoder.named_parameters():
                param.requires_grad = False
            self.graph_encoder = self.graph_encoder.eval()
            self.graph_encoder.train = disabled_train
            logging.info("freeze graph encoder")
        
        self.num_query_token = num_query_token
        self.Qformer, self.query_tokens = self.init_Qformer(bert_name, num_query_token, self.graph_encoder.num_features, cross_attention_freq)
        ### remove the unused parameters
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

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

        self.opt_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.opt_model.config.hidden_size
        )
        if args.stage2_path:
            self.opt_proj.weight.data = new_state_dict['opt_proj.weight']
            self.opt_proj.bias.data = new_state_dict['opt_proj.bias']
            print(f"loaded stage2 model from {args.stage2_path}")
        
        ## fixme: no prompt yet
        self.prompt = prompt
        # prompt_tokens = self.opt_tokenizer(self.prompt, return_tensors="pt")
        # self.prompt_length = prompt_tokens.attention_mask.sum(1)

    def forward_old(self, batch):
        graphs, text_tokens, prompt_lens = batch
        graph_embeds, graph_masks = self.graph_encoder(graphs)
        if not self.tune_gnn:
            graph_embeds = graph_embeds.detach()
        graph_embeds = self.ln_graph(graph_embeds, graph_masks)
        device = graph_embeds.device
        query_tokens = self.query_tokens.expand(graph_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=graph_embeds,
            encoder_attention_mask=graph_masks, # fixme: check whether this mask is correct
            return_dict=True,
        )
        inputs_opt = self.opt_proj(query_output.last_hidden_state)
        atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(device)
        targets = text_tokens.input_ids.masked_fill(
            text_tokens.input_ids == self.opt_tokenizer.pad_token_id, -100
        )
        if self.prompt:
            targets = mask_by_len(targets, prompt_lens, -100) # do not apply loss to the prompt
            # targets[:, : self.prompt_length] = -100  # do not apply loss to the prompt
        
        empty_targets = (
            torch.ones(atts_opt.size(), dtype=torch.long).to(device).fill_(-100)
        )
        targets = torch.cat([empty_targets, targets], dim=1)
        
        inputs_embeds = self.opt_model.get_input_embeddings()(text_tokens.input_ids)
        inputs_embeds = torch.cat([inputs_opt, inputs_embeds], dim=1)
        attention_mask = torch.cat([atts_opt, text_tokens.attention_mask], dim=1)
        
        outputs = self.opt_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
        )
        loss = outputs.loss
        return {"loss": loss}
    
    def forward(self, batch):
        graphs, prompt_tokens, text_tokens = batch
        graph_embeds, graph_masks = self.graph_encoder(graphs)
        if not self.tune_gnn:
            graph_embeds = graph_embeds.detach()
        graph_embeds = self.ln_graph(graph_embeds, graph_masks)
        device = graph_embeds.device
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
        
        empty_targets = torch.ones(prompt_tokens.attention_mask.shape, dtype=torch.long).to(device).fill_(-100)
        targets = text_tokens.input_ids.masked_fill(
            text_tokens.input_ids == self.opt_tokenizer.pad_token_id, -100
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        # (prompt_tokens.input_ids == 22).nonzero(as_tuple=True)[1]

        prompt_embeds = self.opt_model.get_input_embeddings()(prompt_tokens.input_ids)
        prompt_embeds[prompt_tokens.is_mol_token] = mol_tokens.flatten(0, 1).to(dtype=torch.bfloat16)
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
        
        
        ###============== Attention score regularization ===================###
        att_cos = 0
        att_kl = 0
        att_l2 = 0
        cross_attentions = query_output['cross_attentions'] # Cross attention and past key values are mixed in this tuple
        cross_attentions = [cross_attentions[i] for i in range(0, len(query_output['cross_attentions']), 2)]
        cross_attentions = torch.stack(cross_attentions).transpose(0, 1) # [batch_size, num_cross_attentions, num_heads, query_len, 1+maximum_node_num]
        mean_cross_attentions = cross_attentions.mean(dim=1).mean(1) # [batch_size, query_len, 1+maximum_node_num]
        masked_feat = mean_cross_attentions * graph_masks.unsqueeze(1).float()

        att_cos = self.compute_avg_cosine_similarity(masked_feat)
        att_kl = self.compute_average_symmetrised_kl_divergence(masked_feat)
        att_l2 = self.compute_mean_l2_distance(masked_feat)
        
        return {
            "loss": loss,
            "cross_attentions": query_output.cross_attentions,
            "generation_attentions": generation_attentions,
            "target_start_idx": target_start_idx,
            "att_cos": att_cos.mean(),
            "att_kl": att_kl.mean(),
            "att_l2": att_l2.mean(),
        }

    def forward_reaction(self, batch):
        reaction_tokens, notes_tokens, graphs = batch
        graph_embeds, graph_masks = self.graph_encoder(graphs)
        if not self.tune_gnn:
            graph_embeds = graph_embeds.detach()
        graph_embeds = self.ln_graph(graph_embeds, graph_masks)
        device = graph_embeds.device
        query_tokens = self.query_tokens.expand(graph_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=graph_embeds,
            encoder_attention_mask=graph_masks, # fixme: check whether this mask is correct
            return_dict=True,
        )
        mol_tokens = self.opt_proj(query_output.last_hidden_state) # shape = [mol_num, num_query_token, D]

        if False:
            if self.llm_tune:
                react_embeds = self.opt_model.model.get_decoder().embed_tokens(reaction_tokens.input_ids) # shape = [B, max_len, D]
                notes_embeds = self.opt_model.model.get_decoder().embed_tokens(notes_tokens.input_ids)
            else:
                react_embeds = self.opt_model.model.decoder.embed_tokens(reaction_tokens.input_ids) # shape = [B, max_len, D]
                notes_embeds = self.opt_model.model.decoder.embed_tokens(notes_tokens.input_ids) # shape = [B, max_len, D]
        else:
            react_embeds = self.opt_model.get_input_embeddings()(reaction_tokens.input_ids)
            notes_embeds = self.opt_model.get_input_embeddings()(notes_tokens.input_ids)

        react_embeds[reaction_tokens.is_ph_token] = mol_tokens.flatten(0, 1)
        inputs_embeds = torch.cat((react_embeds, notes_embeds), dim=1)

        targets = notes_tokens.input_ids.masked_fill(
            notes_tokens.input_ids == self.opt_tokenizer.pad_token_id, -100
        )
        empty_targets = (
            torch.ones(reaction_tokens.attention_mask.shape, dtype=torch.long).to(device).fill_(-100)
        )
        targets = torch.cat([empty_targets, targets], dim=1)
        attention_mask = torch.cat([reaction_tokens.attention_mask, notes_tokens.attention_mask], dim=1)

        outputs = self.opt_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
        )
        loss = outputs.loss
        return {"loss": loss}

    @torch.no_grad()
    def generate_old(
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
        with self.maybe_autocast():
            graph_embeds, graph_masks = self.graph_encoder(graphs)
            graph_embeds = self.ln_graph(graph_embeds)

            query_tokens = self.query_tokens.expand(graph_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=graph_embeds,
                encoder_attention_mask=graph_masks,
                return_dict=True,
            )

            device = graph_embeds.device
            inputs_opt = self.opt_proj(query_output.last_hidden_state)
            atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long, device=device)

            attention_mask = torch.cat([atts_opt, prompt_tokens.attention_mask], dim=1)
            
            inputs_embeds = self.opt_model.get_input_embeddings()(prompt_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_opt, inputs_embeds], dim=1)

            outputs = self.opt_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
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
        graph_embeds, graph_masks = self.graph_encoder(graphs)
        graph_embeds = self.ln_graph(graph_embeds)

        query_tokens = self.query_tokens.expand(graph_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=graph_embeds,
            encoder_attention_mask=graph_masks,
            return_dict=True,
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
        
        prompt_embeds = self.opt_model.get_input_embeddings()(prompt_tokens.input_ids)
        prompt_embeds[prompt_tokens.is_mol_token] = mol_tokens.flatten(0, 1)

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

    @torch.no_grad()
    def blip_qa(
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
        output_scores=False,
        ):

        device = next(self.parameters()).device
        
        ## data processing
        prompts = samples['prompts'] # assume list of strings
        prepared_prompts = []
        mol_list = []
        for p in prompts:
            text, smiles = smiles_handler(p, self.mol_token * self.num_query_token)
            prepared_prompts.append(text)
            mol_list.extend([smiles2data(s) for s in smiles])
        
        prompt_tokens = self.opt_tokenizer(prepared_prompts,
                                           truncation=False,
                                           padding='longest',
                                           add_special_tokens=True,
                                        #    max_length=self.args.max_len[],
                                           return_tensors='pt',
                                           return_attention_mask=True).to(device)
        
        ## forward function
        prompt_embeds = self.opt_model.get_input_embeddings()(prompt_tokens.input_ids)
        
        if len(mol_list) > 0:
            graphs = self.collater(mol_list).to(device)
            is_mol_token = (prompt_tokens.input_ids == self.mol_token) # shape = [B, max_len]
            ## graph forward
            graph_embeds, graph_masks = self.graph_encoder(graphs)
            graph_embeds = self.ln_graph(graph_embeds, graph_masks)
            query_tokens = self.query_tokens.expand(graph_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=graph_embeds,
                encoder_attention_mask=graph_masks, # fixme: check whether this mask is correct
                return_dict=True,
            )
            mol_tokens = self.opt_proj(query_output.last_hidden_state) # shape = [mol_num, num_query_token, D]
            ## replace mol tokens
            prompt_embeds[is_mol_token] = mol_tokens.flatten(0, 1)
        
        if output_scores:
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
                    output_scores=True,
                    return_dict_in_generate=True
                    # use_cache=False,
            )
            return outputs
        else:
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
    
    @torch.no_grad()
    def opt_qa(
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
        output_scores=False,
        ):

        device = next(self.parameters()).device
        ## data processing
        prompts = samples['prompts'] # assume list of strings
        prompts = [escape_custom_split_sequence(p) for p in prompts]
        
        prompt_tokens = self.opt_tokenizer(prompts,
                                           truncation=False,
                                           padding='longest',
                                           add_special_tokens=True,
                                        #    max_length=self.args.max_len[],
                                           return_tensors='pt',
                                           return_attention_mask=True).to(device)
        
        prompt_embeds = self.opt_model.get_input_embeddings()(prompt_tokens.input_ids)

        if output_scores:
            ## forward function
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
                    output_scores=True,
                    return_dict_in_generate=True
                )
            return outputs
        else:
            ## forward function
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
        
    @torch.no_grad()
    def probe_qformer(
        self, 
        batch,
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
        with self.maybe_autocast():
            device = next(self.parameters()).device
            
            graphs, smiles_prompt_tokens, texts = batch
            graphs = graphs.to(device)
            ## graph forward
            graph_embeds, graph_masks = self.graph_encoder(graphs)
            graph_embeds = self.ln_graph(graph_embeds, graph_masks)
            query_tokens = self.query_tokens.expand(graph_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=graph_embeds,
                encoder_attention_mask=graph_masks, # fixme: check whether this mask is correct
                return_dict=True,
            )
            mol_tokens = self.opt_proj(query_output.last_hidden_state) # shape = [mol_num, num_query_token, D]
            B, num_q, D = mol_tokens.shape
            
            ## 
            embed_func = self.opt_model.get_input_embeddings()
            embed_weight = embed_func.weight # shape = [vocab_size, D]
            
            dis_metric = 'cos'
            topk = 10
            if dis_metric == 'cos':
                mol_tokens = F.normalize(mol_tokens, dim=-1, p=2)
                embed_weight = F.normalize(embed_weight, dim=-1, p=2)
                sim = mol_tokens.flatten(0, 1) @ embed_weight.T # shape = [mol_num * num_query_token, vocab_size]
            elif dis_metric == 'euc':
                sim = - torch.cdist(mol_tokens.flatten(0, 1), embed_weight, p=2)
                assert sim.shape == (B * num_q, embed_weight.shape[0])
            else:
                raise NotImplementedError()
            _, topk_ids = torch.topk(sim, k=topk, dim=-1) # shape = [mol_num * num_query_token, k]
            knn_decode_strings = self.opt_tokenizer.batch_decode(topk_ids.flatten())
            knn_decode_strings = np.asarray(knn_decode_strings).reshape(B, num_q, topk).tolist() # shape = [mol_num, num_query_token, topk]
            knn_decode_strings = [[' '.join(ii) for ii in i] for i in knn_decode_strings] # shape = [mol_num, num_query_token]
            if False:
                ### print for presentation
                assert len(knn_decode_strings) == len(texts)
                for predict, text in zip(knn_decode_strings, texts):
                    print('----------------------------')
                    print(predict)
                    print(text)
            return knn_decode_strings

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
