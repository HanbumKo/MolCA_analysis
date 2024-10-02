"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging
import os
import torch
import torch.distributed as dist
from torch import Tensor
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F
from typing import Any, Iterable, Iterator, List, Optional, Sized, Tuple, Union, Dict
from model.help_funcs import pad_and_concat

# from lavis.common.registry import registry
# from lavis.models.base_model import all_gather_with_grad, concat_all_gather
from lavis.models.blip2_models.blip2 import (
    disabled_train,
)
from lavis.models.blip_models.blip_outputs import BlipOutput
from lavis.common.dist_utils import is_dist_avail_and_initialized
from model.blip2 import Blip2Base
# from pytorch_lightning.utilities import distributed

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    # if use distributed training
    if not is_dist_avail_and_initialized():
        return tensor

    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    print('running here')
    return output

# @torch.no_grad()
# def pl_concat_all_gather(tensor):
#     """
#     Performs all_gather operation on the provided tensors.
#     *** Warning ***: torch.distributed.all_gather has no gradient.
#     """
#     # if use distributed training
#     if not is_dist_avail_and_initialized():
#         return tensor

#     tensors_gather = distributed.gather_all_tensors(tensor)
#     output = torch.cat(tensors_gather, dim=0)
#     return output

@torch.no_grad()
def pl_concat_all_gather(tensor, padding=False, fill_value=0):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    # if use distributed training
    if not is_dist_avail_and_initialized():
        return tensor

    tensors_gather = gather_all_tensors(tensor)
    if padding:
        output = pad_and_concat(tensors_gather, fill_value=fill_value).detach()
    else:
        output = torch.cat(tensors_gather, dim=0)
    return output


def gather_all_tensors(*args: Any, **kwargs: Any) -> Any:
    return _gather_all_tensors(*args, **kwargs)

def _gather_all_tensors(result: Tensor, group: Optional[Any] = None) -> List[Tensor]:
    """Function to gather all tensors from several DDP processes onto a list that is broadcasted to all processes.

    Works on tensors that have the same number of dimensions, but where each dimension may differ. In this case
    tensors are padded, gathered and then trimmed to secure equal workload for all processes.

    Args:
        result: The value to sync
        group: The process group to gather results from. Defaults to all processes (world)

    Return:
        gathered_result: List with size equal to the process group where
            gathered_result[i] corresponds to result tensor from process i
    """
    if group is None:
        group = torch.distributed.group.WORLD

    # Convert tensors to contiguous format
    result = result.contiguous()

    world_size = torch.distributed.get_world_size(group)
    torch.distributed.barrier(group=group)

    # If the tensor is scalar, things are easy
    if result.ndim == 0:
        return _simple_gather_all_tensors(result, group, world_size)

    # 1. Gather sizes of all tensors
    local_size = torch.tensor(result.shape, device=result.device)
    local_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    torch.distributed.all_gather(local_sizes, local_size, group=group)
    max_size = torch.stack(local_sizes).max(dim=0).values
    all_sizes_equal = all(all(ls == max_size) for ls in local_sizes)

    # 2. If shapes are all the same, then do a simple gather:
    if all_sizes_equal:
        return _simple_gather_all_tensors(result, group, world_size)

    # 3. If not, we need to pad each local tensor to maximum size, gather and then truncate
    pad_dims = []
    pad_by = (max_size - local_size).detach().cpu()
    for val in reversed(pad_by):
        pad_dims.append(0)
        pad_dims.append(val.item())
    result_padded = F.pad(result, pad_dims)
    gathered_result = [torch.zeros_like(result_padded) for _ in range(world_size)]
    torch.distributed.all_gather(gathered_result, result_padded, group)
    for idx, item_size in enumerate(local_sizes):
        slice_param = [slice(dim_size) for dim_size in item_size]
        gathered_result[idx] = gathered_result[idx][slice_param]
    return gathered_result

def _simple_gather_all_tensors(result: Tensor, group: Any, world_size: int) -> List[Tensor]:
    gathered_result = [torch.zeros_like(result) for _ in range(world_size)]
    torch.distributed.all_gather(gathered_result, result, group)
    return gathered_result

# @registry.register_model("blip2")
# @registry.register_model("blip2_feature_extractor")
class Blip2Qformer(Blip2Base):
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
        gtm,
        lm,
        bert_name,
        temperature,
        gin_num_layers,
        gin_hidden_dim,
        gin_drop_ratio,
        att_reg,
        att_reg_method,
        att_reg_lambda,
        tune_gnn=False,
        num_query_token=32,
        cross_attention_freq=2,
        embed_dim=256,
    ):
        super().__init__()
        self.gtm = gtm
        self.lm = lm
        self.att_reg = att_reg
        self.att_reg_method = att_reg_method
        
        self.tokenizer = self.init_tokenizer()

        self.graph_encoder, self.ln_graph = self.init_graph_encoder(gin_num_layers, gin_hidden_dim, gin_drop_ratio)
        self.tune_gnn = tune_gnn
        if not tune_gnn:
            for name, param in self.graph_encoder.named_parameters():
                param.requires_grad = False
            self.graph_encoder = self.graph_encoder.eval()
            self.graph_encoder.train = disabled_train
            logging.info("freeze graph encoder")
        
        self.Qformer, self.query_tokens = self.init_Qformer(bert_name, num_query_token, self.graph_encoder.num_features, cross_attention_freq)
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])

        self.graph_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)

        self.gtm_head = nn.Linear(self.Qformer.config.hidden_size, 2)

        self.temperature = temperature

    
    def contrast(self, features_graph, features_text, return_sim=False):
        '''
        features_graph: shape = [B, num_qs, D]
        features_text: shape = [B, D]
        '''
        batch_size = features_graph.size(0)

        # normalized features
        features_graph = F.normalize(features_graph, dim=-1)
        features_text = F.normalize(features_text, dim=-1)

        # cosine similarity as logits
        sim_q2t = (features_graph.unsqueeze(1) @ features_text.unsqueeze(-1)).squeeze() # shape = [B, 1, num_qs, D]; shape = [B, D, 1]; output shape = [B, B, num_qs]
        sim_g2t, _ = sim_q2t.max(-1) # shape = [B, B]

        logits_per_graph = sim_g2t / self.temperature
        logits_per_text = logits_per_graph.t()

        labels = torch.arange(batch_size, dtype=torch.long, device=self.device)  # 大小为B
        loss_graph = F.cross_entropy(logits_per_graph, labels)
        loss_text = F.cross_entropy(logits_per_text, labels)
        loss = (loss_graph + loss_text) / 2

        if return_sim:
            return logits_per_graph, logits_per_text, loss
        else:
            return loss

    def contrast_global(self, features_graph, features_text, features_graph_all, features_text_all, return_sim=False):
        '''
        features_graph: shape = [B, num_qs, D]
        features_text: shape = [B, D]
        features_text_all: shape = [B * num_gpus, D]
        features_graph_all: shape = [B * num_gpus, num_qs, D]
        '''
        bs = features_graph.size(0)

        # cosine similarity as logits
        sim_q2t = (features_graph.unsqueeze(1) @ features_text_all.unsqueeze(-1)).squeeze() # shape = [B, 1, num_qs, D]; shape = [B * num_gpus, D, 1]; output shape = [B, B * num_gpus, num_qs]
        sim_g2t, _ = sim_q2t.max(-1) # shape = [B, B * num_gpus]

        logits_per_graph = sim_g2t / self.temperature
    

        sim_t2q = (features_text.unsqueeze(1).unsqueeze(1) @ features_graph_all.permute(0, 2, 1)).squeeze() # shape = [B, 1, 1, D]; [B*num_gpus, D, num_qs]; output shape = [B, B*num_gpus, 1, num_qs]
        sim_t2g, _ = sim_t2q.max(-1)
        logits_per_text = sim_t2g / self.temperature

        # labels = torch.arange(bs, dtype=torch.long, device=self.device)
        # rank = dist.get_rank()
        rank = 0
        labels = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(self.device)

        loss_graph = F.cross_entropy(logits_per_graph, labels)
        loss_text = F.cross_entropy(logits_per_text, labels)
        loss = (loss_graph + loss_text) / 2

        if return_sim:
            return logits_per_graph[:, rank*bs:rank*bs+bs], logits_per_text[:, rank*bs:rank*bs+bs], loss
        else:
            return loss
        
    def forward_old(self, batch):
        ## v1: not gather results from all gpus
        ###============== Image-text Contrastive ===================###
        graph, text, mask = batch
        batch_node, batch_mask = self.graph_encoder(graph)
        batch_node = batch_node.detach()
        batch_size = batch_node.shape[0]

        batch_node = self.ln_graph(batch_node, batch_mask)
        query_tokens = self.query_tokens.expand(batch_node.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=batch_node,
            encoder_attention_mask=batch_mask, # fixme: check whether this mask is correct
            use_cache=True,
            return_dict=True,
        )
        graph_feats = self.graph_proj(query_output.last_hidden_state) # shape = [B, num_q, D]
        text_output = self.Qformer.bert(text, attention_mask=mask, return_dict=True) # shape = [B, n_max, D]
        text_feats = self.text_proj(text_output.last_hidden_state[:, 0, :])
        sim_g2t, sim_t2g, loss_gtc = self.contrast(graph_feats, text_feats, return_sim=True)


        ###============== Image-text Matching ===================###
        loss_gtm = 0
        if self.gtm:
            g_emb = batch_node
            g_mask = batch_mask
            text_ids = text.clone()
            with torch.no_grad():
                weights_t2g = F.softmax(sim_t2g, dim=1) + 1e-4
                weights_t2g.fill_diagonal_(0)
                weights_g2t = F.softmax(sim_g2t, dim=1) + 1e-4
                weights_g2t.fill_diagonal_(0)

            # select a negative graph for each text
            graph_embeds_neg = []
            graph_mask_neg = []
            for b in range(batch_size):
                neg_idx = torch.multinomial(weights_t2g[b], 1).item()
                graph_embeds_neg.append(g_emb[neg_idx])
                graph_mask_neg.append(g_mask[neg_idx])
            
            graph_embeds_neg = torch.stack(graph_embeds_neg, dim=0)
            graph_mask_neg = torch.stack(graph_mask_neg, dim=0)

            # select a negative text for each image
            text_ids_neg = []
            text_atts_neg = []
            for b in range(batch_size):
                neg_idx = torch.multinomial(weights_g2t[b], 1).item()
                text_ids_neg.append(text_ids[neg_idx])
                text_atts_neg.append(mask[neg_idx])

            text_ids_neg = torch.stack(text_ids_neg, dim=0)
            text_atts_neg = torch.stack(text_atts_neg, dim=0)

            text_ids_all = torch.cat(
                [text_ids, text_ids, text_ids_neg], dim=0
            )  # pos, pos, neg
            text_atts_all = torch.cat(
                [mask, mask, text_atts_neg],
                dim=0,
            )

            query_tokens_itm = self.query_tokens.expand(text_ids_all.shape[0], -1, -1)
            query_atts_itm = torch.ones(query_tokens_itm.size()[:-1], dtype=torch.long, device=text.device)
            attention_mask_all = torch.cat([query_atts_itm, text_atts_all], dim=1)

            graph_embeds_all = torch.cat([g_emb, graph_embeds_neg, g_emb], dim=0)  # pos, neg, pos
            graph_atts_all = torch.cat([g_mask, graph_mask_neg, g_mask], dim=0)

            output_itm = self.Qformer.bert(
                text_ids_all,
                query_embeds=query_tokens_itm,
                attention_mask=attention_mask_all,
                encoder_hidden_states=graph_embeds_all,
                encoder_attention_mask=graph_atts_all,
                return_dict=True,
            )

            vl_embeddings = output_itm.last_hidden_state[:, : query_tokens_itm.size(1), :] # keep query tokens only
            vl_output = self.gtm_head(vl_embeddings)
            logits = vl_output.mean(dim=1)

            itm_labels = torch.cat(
                [torch.ones(batch_size, dtype=torch.long), torch.zeros(2 * batch_size, dtype=torch.long)],
                dim=0,
            ).to(text.device)
            loss_gtm = F.cross_entropy(logits, itm_labels)

        ##================= Image Captioning ========================##
        loss_lm = 0
        if self.lm:
            decoder_input_ids = text.clone()
            decoder_input_ids[:, 0] = self.tokenizer.bos_token_id
            labels = decoder_input_ids.masked_fill(
                decoder_input_ids == self.tokenizer.pad_token_id, -100
            )
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long, device=text.device)
            
            attention_mask = torch.cat([query_atts, mask], dim=1)
            lm_output = self.Qformer(
                decoder_input_ids,
                attention_mask=attention_mask,
                past_key_values=query_output.past_key_values,
                return_dict=True,
                labels=labels,
            )

            loss_lm = lm_output.loss

        return BlipOutput(
            loss=loss_gtc + loss_gtm + loss_lm,
            loss_itc=loss_gtc,
            loss_itm=loss_gtm,
            loss_lm=loss_lm,
        )


    def forward(self, batch):
        ## v2: gather results from all gpus
        ###============== Image-text Contrastive ===================###
        graph, text, mask = batch
        batch_node, batch_mask = self.graph_encoder(graph) # Graph Encoder in the paper
        if not self.tune_gnn:
            batch_node = batch_node.detach()
        batch_size = batch_node.shape[0]

        batch_node = self.ln_graph(batch_node, batch_mask)
        query_tokens = self.query_tokens.expand(batch_node.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=batch_node,
            encoder_attention_mask=batch_mask, # fixme: check whether this mask is correct
            use_cache=True,
            return_dict=True,
            output_attentions=True,
        )
        graph_feats = self.graph_proj(query_output.last_hidden_state) # shape = [B, num_q, D]
        text_output = self.Qformer.bert(text, attention_mask=mask, return_dict=True, output_attentions=True) # shape = [B, n_max, D]
        text_feats = self.text_proj(text_output.last_hidden_state[:, 0, :])
        
        text_feats, graph_feats = F.normalize(text_feats, p=2, dim=-1), F.normalize(graph_feats, p=2, dim=-1)
        text_feats_all, graph_feats_all = pl_concat_all_gather(text_feats), pl_concat_all_gather(graph_feats) # shape = [B * num_gpus, D]
        sim_g2t, sim_t2g, loss_gtc = self.contrast_global(graph_feats, text_feats, graph_feats_all, text_feats_all, return_sim=True)


        ###============== Image-text Matching ===================###
        loss_gtm = 0
        if self.gtm:
            ## not aggregate global tensor because of their different shapes
            g_emb_world = batch_node
            g_mask_world = batch_mask
            text_ids_world = text
            text_mask_world = mask
            with torch.no_grad():
                weights_t2g = F.softmax(sim_t2g, dim=1) + 1e-4
                weights_t2g.fill_diagonal_(0)
                weights_g2t = F.softmax(sim_g2t, dim=1) + 1e-4
                weights_g2t.fill_diagonal_(0)

            # select a negative graph for each text
            graph_embeds_neg = []
            graph_mask_neg = []
            for b in range(batch_size):
                neg_idx = torch.multinomial(weights_t2g[b], 1).item() # sample a negative graph based on the similarity
                graph_embeds_neg.append(g_emb_world[neg_idx]) # append the negative graph embedding
                graph_mask_neg.append(g_mask_world[neg_idx]) # append the negative graph mask
            
            graph_embeds_neg = torch.stack(graph_embeds_neg, dim=0) # stack the negative graph embeddings | shape = [B, num_qs, D]
            graph_mask_neg = torch.stack(graph_mask_neg, dim=0) # stack the negative graph masks | shape = [B, num_qs]

            # select a negative text for each image
            text_ids_neg = []
            text_atts_neg = []
            for b in range(batch_size):
                neg_idx = torch.multinomial(weights_g2t[b], 1).item()
                text_ids_neg.append(text_ids_world[neg_idx])
                text_atts_neg.append(text_mask_world[neg_idx])

            text_ids_neg = torch.stack(text_ids_neg, dim=0)
            text_atts_neg = torch.stack(text_atts_neg, dim=0)

            text_ids_all = torch.cat(
                [text, text, text_ids_neg], dim=0
            )  # pos, pos, neg
            text_atts_all = torch.cat(
                [mask, mask, text_atts_neg],
                dim=0,
            )

            query_tokens_itm = self.query_tokens.expand(text_ids_all.shape[0], -1, -1)
            query_atts_itm = torch.ones(query_tokens_itm.size()[:-1], dtype=torch.long, device=text.device)
            attention_mask_all = torch.cat([query_atts_itm, text_atts_all], dim=1)

            graph_embeds_all = torch.cat([batch_node, graph_embeds_neg, batch_node], dim=0)  # pos, neg, pos
            graph_atts_all = torch.cat([batch_mask, graph_mask_neg, batch_mask], dim=0)

            output_itm = self.Qformer.bert(
                text_ids_all,
                query_embeds=query_tokens_itm,
                attention_mask=attention_mask_all,
                encoder_hidden_states=graph_embeds_all,
                encoder_attention_mask=graph_atts_all,
                return_dict=True,
                output_attentions=True,
            )
            # output_itm.last_hidden_state.shape = [batch_size * 3, num_query + max_text_len, hidden_dim]
            vl_embeddings = output_itm.last_hidden_state[:, : query_tokens_itm.size(1), :] # keep query tokens only | shape = [batch_size * 3, num_query, hidden_dim]
            vl_output = self.gtm_head(vl_embeddings)
            logits = vl_output.mean(dim=1) # mean pooling for all query tokens | shape = [batch_size * 3, 2]

            itm_labels = torch.cat(
                [torch.ones(batch_size, dtype=torch.long), torch.zeros(2 * batch_size, dtype=torch.long)],
                dim=0,
            ).to(text.device)
            loss_gtm = F.cross_entropy(logits, itm_labels)

        ##================= Image Captioning ========================##
        loss_lm = 0
        if self.lm:
            decoder_input_ids = text.clone()
            decoder_input_ids[:, 0] = self.tokenizer.bos_token_id
            labels = decoder_input_ids.masked_fill(
                decoder_input_ids == self.tokenizer.pad_token_id, -100
            )
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long, device=text.device)
            
            attention_mask = torch.cat([query_atts, mask], dim=1)
            lm_output = self.Qformer(
                decoder_input_ids,
                attention_mask=attention_mask,
                past_key_values=query_output.past_key_values,
                return_dict=True,
                labels=labels,
            )

            loss_lm = lm_output.loss


        ###============== Attention score regularization ===================###
        att_cos = 0
        att_kl = 0
        att_l2 = 0
        cross_attentions = query_output['cross_attentions'] # Cross attention and past key values are mixed in this tuple
        cross_attentions = [cross_attentions[i] for i in range(0, len(query_output['cross_attentions']), 2)]
        cross_attentions = torch.stack(cross_attentions).transpose(0, 1) # [batch_size, num_cross_attentions, num_heads, query_len, 1+maximum_node_num]
        mean_cross_attentions = cross_attentions.mean(dim=1).mean(1) # [batch_size, query_len, 1+maximum_node_num]
        masked_feat = mean_cross_attentions * batch_mask.unsqueeze(1).float()

        att_cos = self.compute_avg_cosine_similarity(masked_feat)
        att_kl = self.compute_average_symmetrised_kl_divergence(masked_feat)
        att_l2 = self.compute_mean_l2_distance(masked_feat)

        return BlipOutput(
            loss=loss_gtc + loss_gtm + loss_lm,
            loss_itc=loss_gtc,
            loss_itm=loss_gtm,
            loss_lm=loss_lm,
        ), att_cos.mean(), att_kl.mean(), att_l2.mean()
    
    def forward_v3(self, batch):
        ## v3: use smiles instruction
        ###============== Image-text Contrastive ===================###
        graphs, text_tokens, prompt_tokens = batch
        graph_embeds, graph_masks = self.graph_encoder(graphs)
        if not self.tune_gnn:
            graph_embeds = graph_embeds.detach()
        graph_embeds = self.ln_graph(graph_embeds, graph_masks)

        device = text_tokens.input_ids.device
        batch_size = graph_embeds.shape[0]
        
        ## 
        query_tokens = self.query_tokens.expand(graph_embeds.shape[0], -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long, device=device)
        attention_mask_gtc = torch.cat([query_atts, prompt_tokens.attention_mask], dim=1)
        query_output = self.Qformer.bert(
            input_ids=prompt_tokens,
            query_embeds=query_tokens,
            attention_mask=attention_mask_gtc,
            encoder_hidden_states=graph_embeds,
            encoder_attention_mask=graph_masks, # fixme: check whether this mask is correct
            use_cache=True,
            return_dict=True,
        )

        query_output = query_output.last_hidden_state[:, : query_tokens.size(1), :] # keep query tokens only
        graph_feats = self.graph_proj(query_output) # shape = [B, num_q, D]
        text_output = self.Qformer.bert(text_tokens.input_ids, attention_mask=text_tokens.attention_mask, return_dict=True) # shape = [B, n_max, D]
        text_feats = self.text_proj(text_output.last_hidden_state[:, 0, :])
        
        text_feats, graph_feats = F.normalize(text_feats, p=2, dim=-1), F.normalize(graph_feats, p=2, dim=-1)
        text_feats_all, graph_feats_all = pl_concat_all_gather(text_feats), pl_concat_all_gather(graph_feats) # shape = [B * num_gpus, D]
        sim_g2t, sim_t2g, loss_gtc = self.contrast_global(graph_feats, text_feats, graph_feats_all, text_feats_all, return_sim=True)


        ###============== Image-text Matching ===================###
        loss_gtm = 0
        if self.gtm:
            ## not aggregate global tensor because of their different shapes
            g_emb_world = graph_embeds
            g_mask_world = graph_masks
            text_ids_world = text_tokens.input_ids
            text_mask_world = text_tokens.attention_mask
            with torch.no_grad():
                weights_t2g = F.softmax(sim_t2g, dim=1) + 1e-4
                weights_t2g.fill_diagonal_(0)
                weights_g2t = F.softmax(sim_g2t, dim=1) + 1e-4
                weights_g2t.fill_diagonal_(0)

            # select a negative graph for each text
            graph_embeds_neg = []
            graph_mask_neg = []
            for b in range(batch_size):
                neg_idx = torch.multinomial(weights_t2g[b], 1).item()
                graph_embeds_neg.append(g_emb_world[neg_idx])
                graph_mask_neg.append(g_mask_world[neg_idx])
            
            graph_embeds_neg = torch.stack(graph_embeds_neg, dim=0)
            graph_mask_neg = torch.stack(graph_mask_neg, dim=0)

            # select a negative text for each image
            text_ids_neg = []
            text_atts_neg = []
            for b in range(batch_size):
                neg_idx = torch.multinomial(weights_g2t[b], 1).item()
                text_ids_neg.append(text_ids_world[neg_idx])
                text_atts_neg.append(text_mask_world[neg_idx])

            text_ids_neg = torch.stack(text_ids_neg, dim=0)
            text_atts_neg = torch.stack(text_atts_neg, dim=0)

            text_ids_all = torch.cat(
                [text_tokens.input_ids, text_tokens.input_ids, text_ids_neg], dim=0
            )  # pos, pos, neg
            text_atts_all = torch.cat(
                [text_tokens.attention_mask, text_tokens.attention_mask, text_atts_neg],
                dim=0,
            )

            query_tokens_itm = self.query_tokens.expand(text_ids_all.shape[0], -1, -1)
            query_atts_itm = torch.ones(query_tokens_itm.size()[:-1], dtype=torch.long, device=text_tokens.input_ids.device)
            attention_mask_all = torch.cat([query_atts_itm, text_atts_all], dim=1)

            graph_embeds_all = torch.cat([graph_embeds, graph_embeds_neg, graph_embeds], dim=0)  # pos, neg, pos
            graph_atts_all = torch.cat([graph_masks, graph_mask_neg, graph_masks], dim=0)

            output_itm = self.Qformer.bert(
                text_ids_all,
                query_embeds=query_tokens_itm,
                attention_mask=attention_mask_all,
                encoder_hidden_states=graph_embeds_all,
                encoder_attention_mask=graph_atts_all,
                return_dict=True,
            )

            vl_embeddings = output_itm.last_hidden_state[:, : query_tokens_itm.size(1), :] # keep query tokens only
            vl_output = self.gtm_head(vl_embeddings)
            logits = vl_output.mean(dim=1)

            itm_labels = torch.cat(
                [torch.ones(batch_size, dtype=torch.long), torch.zeros(2 * batch_size, dtype=torch.long)],
                dim=0,
            ).to(text_tokens.input_ids.device)
            loss_gtm = F.cross_entropy(logits, itm_labels)

        ##================= Image Captioning ========================##
        loss_lm = 0
        if self.lm:
            decoder_input_ids = text_tokens.input_ids.clone()
            decoder_input_ids[:, 0] = self.tokenizer.bos_token_id
            labels = decoder_input_ids.masked_fill(
                decoder_input_ids == self.tokenizer.pad_token_id, -100
            )
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long, device=text_tokens.input_ids.device)
            
            attention_mask = torch.cat([query_atts, prompt_tokens.attention_mask, text_tokens.attention_mask], dim=1)
            lm_output = self.Qformer(
                decoder_input_ids,
                attention_mask=attention_mask,
                past_key_values=query_output.past_key_values,
                return_dict=True,
                labels=labels,
            )

            loss_lm = lm_output.loss

        return BlipOutput(
            loss=loss_gtc + loss_gtm + loss_lm,
            loss_itc=loss_gtc,
            loss_itm=loss_gtm,
            loss_lm=loss_lm,
        )
    
    def graph_forward(self, graph):
        batch_node, batch_mask = self.graph_encoder(graph)
        batch_node = self.ln_graph(batch_node, batch_mask)
        query_tokens = self.query_tokens.expand(batch_node.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=batch_node,
            encoder_attention_mask=batch_mask, # fixme: check whether this mask is correct
            use_cache=False,
            return_dict=True,
        )
        graph_feats = self.graph_proj(query_output.last_hidden_state) # shape = [B, num_q, D]
        graph_feats = F.normalize(graph_feats, p=2, dim=-1)
        return graph_feats, batch_node, batch_mask

    def text_forward(self, text, mask):
        text_output = self.Qformer.bert(text, attention_mask=mask, return_dict=True) # shape = [B, n_max, D]
        text_feats = self.text_proj(text_output.last_hidden_state[:, 0, :] )
        text_feats = F.normalize(text_feats, dim=-1, p=2)
        return text_feats
    
    def compute_gtm(self, batch_node, batch_mask, text_ids, text_atts):
        '''
        batch_node shape = [B, N, D]
        batch_mask shape = [B, N]
        text_ids shape = [B, N]
        text_atts shape = [B, N]
        '''
        query_tokens = self.query_tokens.expand(batch_node.shape[0], -1, -1) # shape = [B, Nq, D]
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            batch_node.device
        ) # shape = [B, Nq]
        attention_mask = torch.cat([query_atts, text_atts], dim=1) # shape = [B, Nq + N]
        output_gtm = self.Qformer.bert(
            text_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=batch_node,
            encoder_attention_mask=batch_mask,
            return_dict=True,
        )
        gl_embeddings = output_gtm.last_hidden_state[:, : query_tokens.size(1), :] # shape = [B, Nq, D]
        gtm_logit = self.gtm_head(gl_embeddings).mean(dim=1) # shape = [B, Nq, 2]
        # gtm_logit = F.softmax(gtm_logit, dim=-1)[:, 1] # select the axis of the positive class
        gtm_logit = gtm_logit[:, 1] # select the axis of the positive class
        return gtm_logit

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
