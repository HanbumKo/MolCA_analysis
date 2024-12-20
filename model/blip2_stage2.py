import os
from typing import Any, Dict
import torch
from model.blip2_opt import Blip2OPT
from model.blip2_opt_only import Blip2OPTOnly
from model.blip2_llama import Blip2Llama
from model.blip2_t5 import Blip2T5
import pytorch_lightning as pl
from torch import optim
from lavis.common.optims import LinearWarmupCosineLRScheduler, LinearWarmupStepLRScheduler
import json
import torch.distributed as dist
from peft import LoraConfig, TaskType
from model.help_funcs import caption_evaluate, regression_evaluate, calculate_smiles_metrics, AttrDict
from transformers import Adafactor


def load_ignore_unexpected(model, state_dict):
    keys = set(model.state_dict().keys())
    state_dict = {k: v for k, v in state_dict.items() if k in keys}
    
    ## try to print keys that are not included
    model.load_state_dict(state_dict, strict=True)


# def load_ignore_mismatch(model, state_dict):
#     keys = set(model.state_dict().keys())
#     extra_keys = set()
#     for key in state_dict:
#         if key not in keys:
#             extra_keys.add(key)
#     missing_keys = set()
#     for key in keys:
#         if key not in state_dict:
#             missing_keys.add(key)
#     ## try to print keys that are not included
#     model.load_state_dict(state_dict, strict=False)
    

def get_module_state_dict(state_dict, module_name):
    module_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(module_name):
            key = key[len(module_name) + 1:]
            if key == '':
                return value
            module_state_dict[key] = value
    return module_state_dict
# peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
class Blip2Stage2(pl.LightningModule):
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint.pop('optimizer_states')
        # to_be_removed = []
        # for key, value in checkpoint['state_dict'].items():
        #     try:
        #         if not self.get_parameter(key).requires_grad:
        #             to_be_removed.append(key)
        #     except AttributeError:
        #         to_be_removed.append(key)
        # for key in to_be_removed:
        #     checkpoint['state_dict'].pop(key)
    
    def __init__(self, args):
        super().__init__()
        if isinstance(args, dict):
            args = AttrDict(**args)

        self.args = args
        if not hasattr(args, 'do_sample'):
            args.do_sample = False
        self.caption_eval_epoch = args.caption_eval_epoch
        self.do_sample = args.do_sample
        self.num_beams = args.num_beams
        self.max_len = args.max_len
        self.min_len = args.min_len
        self.reaction_weight = args.reaction_weight
        self.llm_tune = args.llm_tune
        self.is_regression = args.root.lower().find('property_prediction') >= 0
        if args.opt_model.find('galactica') >= 0:
            self.blip2opt = Blip2OPT(args.bert_name, args.gin_num_layers, args.gin_hidden_dim, args.drop_ratio, args.tune_gnn, args.num_query_token, args.cross_attention_freq, args.llm_tune, args.peft_dir, args.opt_model, args.prompt, args)
        elif args.opt_model.find('only') >= 0:
            self.blip2opt = Blip2OPTOnly(args.bert_name, args.gin_num_layers, args.gin_hidden_dim, args.drop_ratio, args.tune_gnn, args.num_query_token, args.cross_attention_freq, args.llm_tune, args.peft_dir, args.opt_model, args.prompt, args)
        elif args.opt_model.find('llama') >= 0 or args.opt_model.find('vicuna') >= 0:
            self.blip2opt = Blip2Llama(args.bert_name, args.gin_num_layers, args.gin_hidden_dim, args.drop_ratio, args.tune_gnn, args.num_query_token, args.cross_attention_freq, args.llm_tune, args.peft_dir, args.opt_model, args.prompt, args)
        elif args.opt_model.find('t5') >= 0:
            self.blip2opt = Blip2T5(args.bert_name, args.gin_num_layers, args.gin_hidden_dim, args.drop_ratio, args.tune_gnn, args.num_query_token, args.cross_attention_freq, args.llm_tune, args.peft_dir, args.opt_model, args.prompt, args)
        else:
            raise NotImplementedError()
        self.tokenizer = self.blip2opt.init_tokenizer()
        self.save_hyperparameters(args)

    def load_from_stage1_checkpoint(self, path):
        ckpt = torch.load(path, map_location='cpu')
        state_dict = ckpt['state_dict']
        graph_encoder_dict = get_module_state_dict(state_dict, 'blip2qformer.graph_encoder')
        qformer_dict = get_module_state_dict(state_dict, 'blip2qformer.Qformer')
        ln_graph_dict = get_module_state_dict(state_dict, 'blip2qformer.ln_graph')
        qs_weight = get_module_state_dict(state_dict, 'blip2qformer.query_tokens')
        load_ignore_unexpected(self.blip2opt.Qformer, qformer_dict)
        self.blip2opt.graph_encoder.load_state_dict(graph_encoder_dict)
        self.blip2opt.ln_graph.load_state_dict(ln_graph_dict)
        self.blip2opt.query_tokens.data.copy_(qs_weight)
        return self
    
    # def load_from_stage1_checkpoint(self, path):
    #     ckpt = torch.load(path, map_location='cpu')
    #     state_dict = ckpt['state_dict']
    #     state_dict = {k[13:]: v for k,v in state_dict.items()}
    #     load_ignore_mismatch(self.blip2opt, state_dict)
    #     return self
    
    def configure_optimizers(self):
        if self.args.optimizer == 'adafactor':
            print('Using adafactor optimizer')
            optimizer = Adafactor(
                self.parameters(),
                lr=1e-3,
                relative_step=False,
                scale_parameter=False,
                warmup_init=False
            )
            self.scheduler = None
        else:
            self.trainer.fit_loop.setup_data()
            warmup_steps = min(len(self.trainer.train_dataloader), self.args.warmup_steps)
            optimizer = optim.AdamW(self.parameters(), lr=self.args.init_lr, weight_decay=self.args.weight_decay)
            if self.args.scheduler == 'linear_warmup_cosine_lr':
                self.scheduler = LinearWarmupCosineLRScheduler(optimizer, self.args.max_epochs, self.args.min_lr, self.args.init_lr, warmup_steps, self.args.warmup_lr)
            elif self.args.scheduler == 'linear_warmup_step_lr':
                self.scheduler = LinearWarmupStepLRScheduler(optimizer, self.args.max_epochs, self.args.min_lr, self.args.init_lr, self.args.lr_decay_rate, self.args.warmup_lr, warmup_steps)
            elif self.args.scheduler == 'None':
                self.scheduler = None
            else:
                raise NotImplementedError()
        return optimizer

    def on_test_epoch_end(self, outputs):
        list_predictions, list_targets = zip(*outputs)
        predictions = [i for ii in list_predictions for i in ii]
        targets = [i for ii in list_targets for i in ii]

        all_predictions = [None for _ in range(self.trainer.world_size)]
        all_targets = [None for _ in range(self.trainer.world_size)]
        try:
            dist.all_gather_object(all_predictions, predictions)
            dist.all_gather_object(all_targets, targets)
        # except RuntimeError:
        except:
            all_predictions = [predictions]
            all_targets = [targets]
        if self.global_rank == 0:
            all_predictions = [i for ii in all_predictions for i in ii]
            all_targets = [i for ii in all_targets for i in ii]
            self.save_predictions(all_predictions, all_targets)
            if self.args.root.lower().find('forward_reaction_prediction') >= 0:
                calculate_smiles_metrics(all_predictions, all_targets, metrics=('exact match', 'fingerprint'))
            else:
                ## fixme: I am not sure if the max length is the same as previous experiments
                bleu2, bleu4, rouge_1, rouge_2, rouge_l, meteor_score = \
                    caption_evaluate(all_predictions, all_targets, self.tokenizer, self.max_len * 2) 
                self.log("bleu2", bleu2, sync_dist=True)
                self.log("bleu4", bleu4, sync_dist=True)
                self.log("rouge_1", rouge_1, sync_dist=True)
                self.log("rouge_2", rouge_2, sync_dist=True)
                self.log("rouge_l", rouge_l, sync_dist=True)
                self.log("meteor_score", meteor_score, sync_dist=True)

    def save_predictions(self, predictions, targets, filename='predictions.txt'):
        assert len(predictions) == len(targets)
        with open(os.path.join(self.logger.log_dir, filename), 'w', encoding='utf8') as f:
            for p, t in zip(predictions, targets):
                t = t.replace("SPL1T-TH1S-Pl3A5E", "").replace("[START_I_SMILES]", "").replace("[END_I_SMILES]", "")
                line = {'prediction': p, 'target': t}
                f.write(json.dumps(line, ensure_ascii=True) + '\n')

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        graphs, prompt_tokens, texts = batch
        ###============== Captioning Results ===================###
        samples = {'graphs': graphs, 'prompt_tokens': prompt_tokens}
        predictions = self.blip2opt.generate(
            samples, 
            do_sample=self.do_sample,
            num_beams=self.num_beams,
            max_length=self.max_len,
            min_length=self.min_len
        )
        return predictions, texts
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx, dataloader_idx):
        if dataloader_idx == 0:
            _, _, text_tokens = batch
            batch_size = text_tokens.input_ids.shape[0]
            loss = self.blip2opt(batch)
            # att_cos = loss['att_cos']
            # att_kl = loss['att_kl']
            # att_l2 = loss['att_l2']
            # self.log("att_cos", float(att_cos), batch_size=batch_size, sync_dist=True)
            # self.log("att_kl", float(att_kl), batch_size=batch_size, sync_dist=True)
            # self.log("att_l2", float(att_l2), batch_size=batch_size, sync_dist=True)
            ###============== Overall Loss ===================###
            self.log("val molecule loss", float(loss['loss']), batch_size=batch_size, sync_dist=True)
            if self.args.att_reg:
                # return loss['loss'] + att_cos * self.args.att_reg_lambda
                return loss['loss']
            else:
                return loss['loss']
        elif dataloader_idx == 1:
            if (self.current_epoch+1) % self.caption_eval_epoch != 0:
                return 
            graphs, prompt_tokens, texts = batch
            ###============== Captioning Results ===================###
            samples = {'graphs': graphs, 'prompt_tokens': prompt_tokens}
            predictions = self.blip2opt.generate(
                samples, 
                do_sample=self.do_sample,
                num_beams=self.num_beams,
                max_length=self.max_len,
                min_length=self.min_len
            )
            self.list_predictions.append(predictions)
            self.list_targets.append(texts)
        # elif dataloader_idx == 2:
        #     reaction_tokens, _, _ = batch
        #     batch_size = reaction_tokens.input_ids.shape[0]
        #     loss = self.blip2opt.forward_reaction(batch)
        #     ###============== Overall Loss ===================###
        #     self.log("val reaction loss", float(loss['loss']), batch_size=batch_size, sync_dist=True)
        #     return loss['loss']
        elif dataloader_idx == 2:
            if (self.current_epoch+1) % self.caption_eval_epoch != 0:
                return 
            graphs, prompt_tokens, texts = batch
            ###============== Captioning Results ===================###
            samples = {'graphs': graphs, 'prompt_tokens': prompt_tokens}
            predictions = self.blip2opt.generate(
                samples, 
                do_sample=self.do_sample,
                num_beams=self.num_beams,
                max_length=self.max_len,
                min_length=self.min_len
            )
            self.list_predictions_train.append(predictions)
            self.list_targets_train.append(texts)
        elif dataloader_idx == 3:
            if (self.current_epoch+1) % self.caption_eval_epoch != 0:
                return 
            graphs, prompt_tokens, texts = batch
            ###============== Captioning Results ===================###
            samples = {'graphs': graphs, 'prompt_tokens': prompt_tokens}
            predictions = self.blip2opt.generate(
                samples, 
                do_sample=self.do_sample,
                num_beams=self.num_beams,
                max_length=self.max_len,
                min_length=self.min_len
            )
            self.list_predictions_val.append(predictions)
            self.list_targets_val.append(texts)
        else:
            raise NotImplementedError
    
    def on_validation_epoch_start(self) -> None:
        self.list_predictions = []
        self.list_targets = []
        self.list_predictions_train = []
        self.list_targets_train = []
        self.list_predictions_val = []
        self.list_targets_val = []
    
    def on_validation_epoch_end(self) -> None:
    # def validation_epoch_end(self, outputs):
        if (self.current_epoch+1) % self.caption_eval_epoch != 0:
            return 
        # caption_outputs = outputs[1]
        # list_predictions, list_targets = zip(*caption_outputs)
        predictions = [i for ii in self.list_predictions for i in ii]
        targets = [i for ii in self.list_targets for i in ii]
        predictions_train = [i for ii in self.list_predictions_train for i in ii]
        targets_train = [i for ii in self.list_targets_train for i in ii]
        predictions_val = [i for ii in self.list_predictions_val for i in ii]
        targets_val = [i for ii in self.list_targets_val for i in ii]

        all_predictions = [None for _ in range(self.trainer.world_size)]
        all_targets = [None for _ in range(self.trainer.world_size)]
        all_predictions_train = [None for _ in range(self.trainer.world_size)]
        all_targets_train = [None for _ in range(self.trainer.world_size)]
        all_predictions_val = [None for _ in range(self.trainer.world_size)]
        all_targets_val = [None for _ in range(self.trainer.world_size)]
        try:
            dist.all_gather_object(all_predictions, predictions)
            dist.all_gather_object(all_targets, targets)
            dist.all_gather_object(all_predictions_train, predictions_train)
            dist.all_gather_object(all_targets_train, targets_train)
            dist.all_gather_object(all_predictions_val, predictions_val)
            dist.all_gather_object(all_targets_val, targets_val)
        # except RuntimeError:
        except:
            all_predictions = [predictions]
            all_targets = [targets]
            all_predictions_train = [predictions_train]
            all_targets_train = [targets_train]
            all_predictions_val = [predictions_val]
            all_targets_val = [targets_val]
        if self.global_rank == 0:
            all_predictions = [i for ii in all_predictions for i in ii]
            all_targets = [i for ii in all_targets for i in ii]
            if len(all_predictions) > 0:
                self.save_predictions(all_predictions, all_targets, filename=f'predictions_epoch{self.current_epoch}_test.txt')
                if self.is_regression:
                    mae, mse, rmse, validity = regression_evaluate(all_predictions, all_targets)
                    self.log("mae_test", mae, sync_dist=False)
                    self.log("mse_test", mse, sync_dist=False)
                    self.log("rmse_test", rmse, sync_dist=False)
                    self.log("validity_test", validity, sync_dist=False)
                elif self.args.root.lower().find('forward_reaction_prediction') >= 0 or self.args.root == "data/USPTO_forward/USPTO_50K_data/": # forward reaction prediction
                    result_dict = calculate_smiles_metrics(all_predictions, all_targets, metrics=('exact_match', 'fingerprint'))
                    for key, value in result_dict.items():
                        self.log(key+"_test", value, sync_dist=False)
                elif self.args.root.lower().find('reagent_prediction') >= 0: # reagent prediction
                    result_dict = calculate_smiles_metrics(all_predictions, all_targets, metrics=('exact_match', 'fingerprint', 'multiple_match'))
                    for key, value in result_dict.items():
                        self.log(key+"_test", value, sync_dist=False)
                elif self.args.root.lower().find('retrosynthesis') >= 0 or self.args.root == "data/USPTO_retrosynthesis/USPTO_50K_data/": # retrosynthesis
                    result_dict = calculate_smiles_metrics(all_predictions, all_targets, metrics=('exact_match', 'fingerprint', 'multiple_match'))
                    for key, value in result_dict.items():
                        self.log(key+"_test", value, sync_dist=False)
                else: # Text generation problem
                    ## fixme: I am not sure if the max length is the same as previous experiments
                    bleu2, bleu4, rouge_1, rouge_2, rouge_l, meteor_score = \
                        caption_evaluate(all_predictions, all_targets, self.tokenizer, self.max_len * 2) 
                    self.log("bleu2_test", bleu2, sync_dist=False)
                    self.log("bleu4_test", bleu4, sync_dist=False)
                    self.log("rouge_1_test", rouge_1, sync_dist=False)
                    self.log("rouge_2_test", rouge_2, sync_dist=False)
                    self.log("rouge_l_test", rouge_l, sync_dist=False)
                    self.log("meteor_score_test", meteor_score, sync_dist=False)

            all_predictions_train = [i for ii in all_predictions_train for i in ii]
            all_targets_train = [i for ii in all_targets_train for i in ii]
            if len(all_predictions_train) > 0:
                self.save_predictions(all_predictions_train, all_targets_train, filename=f'predictions_epoch{self.current_epoch}_train.txt')
                if self.is_regression:
                    mae, mse, rmse, validity = regression_evaluate(all_predictions_train, all_targets_train)
                    self.log("mae_train", mae, sync_dist=False)
                    self.log("mse_train", mse, sync_dist=False)
                    self.log("rmse_train", rmse, sync_dist=False)
                    self.log("validity_train", validity, sync_dist=False)
                elif self.args.root.lower().find('forward_reaction_prediction') >= 0 or self.args.root == "data/USPTO_forward/USPTO_50K_data/": # forward reaction prediction
                    result_dict = calculate_smiles_metrics(all_predictions_train, all_targets_train, metrics=('exact_match', 'fingerprint'))
                    for key, value in result_dict.items():
                        self.log(key+"_train", value, sync_dist=False)
                elif self.args.root.lower().find('reagent_prediction') >= 0: # reagent prediction
                    result_dict = calculate_smiles_metrics(all_predictions_train, all_targets_train, metrics=('exact_match', 'fingerprint', 'multiple_match'))
                    for key, value in result_dict.items():
                        self.log(key+"_train", value, sync_dist=False)
                elif self.args.root.lower().find('retrosynthesis') >= 0 or self.args.root == "data/USPTO_retrosynthesis/USPTO_50K_data/": # retrosynthesis
                    result_dict = calculate_smiles_metrics(all_predictions_train, all_targets_train, metrics=('exact_match', 'fingerprint', 'multiple_match'))
                    for key, value in result_dict.items():
                        self.log(key+"_train", value, sync_dist=False)
                else: # Text generation problem
                    ## fixme: I am not sure if the max length is the same as previous experiments
                    bleu2, bleu4, rouge_1, rouge_2, rouge_l, meteor_score = \
                        caption_evaluate(all_predictions_train, all_targets_train, self.tokenizer, self.max_len * 2) 
                    self.log("bleu2_train", bleu2, sync_dist=False)
                    self.log("bleu4_train", bleu4, sync_dist=False)
                    self.log("rouge_1_train", rouge_1, sync_dist=False)
                    self.log("rouge_2_train", rouge_2, sync_dist=False)
                    self.log("rouge_l_train", rouge_l, sync_dist=False)
                    self.log("meteor_score_train", meteor_score, sync_dist=False)

            all_predictions_val = [i for ii in all_predictions_val for i in ii]
            all_targets_val = [i for ii in all_targets_val for i in ii]
            if len(all_predictions_val) > 0:
                self.save_predictions(all_predictions_val, all_targets_val, filename=f'predictions_epoch{self.current_epoch}_validation.txt')
                if self.is_regression:
                    mae, mse, rmse, validity = regression_evaluate(all_predictions_val, all_targets_val)
                    self.log("mae_val", mae, sync_dist=False)
                    self.log("mse_val", mse, sync_dist=False)
                    self.log("rmse_val", rmse, sync_dist=False)
                    self.log("validity_val", validity, sync_dist=False)
                elif self.args.root.lower().find('forward_reaction_prediction') >= 0 or self.args.root == "data/USPTO_forward/USPTO_50K_data/": # forward reaction prediction
                    result_dict = calculate_smiles_metrics(all_predictions_val, all_targets_val, metrics=('exact_match', 'fingerprint'))
                    for key, value in result_dict.items():
                        self.log(key+"_val", value, sync_dist=False)
                elif self.args.root.lower().find('reagent_prediction') >= 0: # reagent prediction
                    result_dict = calculate_smiles_metrics(all_predictions_val, all_targets_val, metrics=('exact_match', 'fingerprint', 'multiple_match'))
                    for key, value in result_dict.items():
                        self.log(key+"_val", value, sync_dist=False)
                elif self.args.root.lower().find('retrosynthesis') >= 0 or self.args.root == "data/USPTO_retrosynthesis/USPTO_50K_data/": # retrosynthesis
                    result_dict = calculate_smiles_metrics(all_predictions_val, all_targets_val, metrics=('exact_match', 'fingerprint', 'multiple_match'))
                    for key, value in result_dict.items():
                        self.log(key+"_val", value, sync_dist=False)
                else: # Text generation problem
                    ## fixme: I am not sure if the max length is the same as previous experiments
                    bleu2, bleu4, rouge_1, rouge_2, rouge_l, meteor_score = \
                        caption_evaluate(all_predictions_val, all_targets_val, self.tokenizer, self.max_len * 2) 
                    self.log("bleu2_val", bleu2, sync_dist=False)
                    self.log("bleu4_val", bleu4, sync_dist=False)
                    self.log("rouge_1_val", rouge_1, sync_dist=False)
                    self.log("rouge_2_val", rouge_2, sync_dist=False)
                    self.log("rouge_l_val", rouge_l, sync_dist=False)
                    self.log("meteor_score_val", meteor_score, sync_dist=False)

    def training_step(self, batch, batch_idx):
        if self.scheduler:
            self.scheduler.step(self.trainer.current_epoch, self.trainer.global_step)
        if isinstance(batch, list) and len(batch) == 2:
            molecule_batch, reaction_batch = batch
            batch_size = molecule_batch[-1].size(0)
            ###============== molecule Loss ===================###
            molecule_loss = self.blip2opt(molecule_batch)['loss']
            self.log("molecule loss", float(molecule_loss), batch_size=batch_size, sync_dist=True)
            
            ###============== reaction Loss ===================###
            reaction_loss = self.blip2opt.forward_reaction(reaction_batch)['loss']
            self.log("reaction loss", float(reaction_loss), batch_size=batch_size, sync_dist=True)

            self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'], batch_size=batch_size, sync_dist=True)
            return molecule_loss + self.reaction_weight * reaction_loss
        else:
            batch_size = batch[-1].input_ids.size(0)
            ###============== Overall Loss ===================###
            loss = self.blip2opt(batch)
            # att_cos = loss['att_cos']
            # att_kl = loss['att_kl']
            # att_l2 = loss['att_l2']
            # self.log("att_cos", float(att_cos), batch_size=batch_size, sync_dist=True)
            # self.log("att_kl", float(att_kl), batch_size=batch_size, sync_dist=True)
            # self.log("att_l2", float(att_l2), batch_size=batch_size, sync_dist=True)
            self.log("molecule loss", float(loss['loss']), batch_size=batch_size, sync_dist=True)
            self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'], batch_size=batch_size, sync_dist=True)
            if self.args.att_reg:
                # return loss['loss'] + att_cos * self.args.att_reg_lambda
                return loss['loss']
            else:
                return loss['loss']

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("GINSimclr")
        # train mode
        # GIN
        parser.add_argument('--gin_hidden_dim', type=int, default=300)
        parser.add_argument('--gin_num_layers', type=int, default=5)
        parser.add_argument('--drop_ratio', type=float, default=0.0)
        parser.add_argument('--tune_gnn', action='store_true', default=False)
        parser.add_argument('--gnn_type', type=str, default='gin', choices=['gin', 'dnd', 'stm'])
        parser.add_argument('--projector', type=str, default='qformer', help='type of projector', choices=['qformer', 'mlp'])
        # Bert
        parser.add_argument('--bert_hidden_dim', type=int, default=768, help='')
        parser.add_argument('--bert_name', type=str, default='scibert')
        parser.add_argument('--cross_attention_freq', type=int, default=2)
        parser.add_argument('--num_query_token', type=int, default=8)
        # OPT
        parser.add_argument('--opt_model', type=str, default="facebook/galactica-1.3b")
        # parser.add_argument('--prompt', type=str, default='a molecule of ')
        parser.add_argument('--num_beams', type=int, default=5)
        parser.add_argument('--do_sample', action='store_true', default=False)
        parser.add_argument('--max_len', type=int, default=256)
        parser.add_argument('--min_len', type=int, default=8)
        parser.add_argument('--llm_tune', type=str, default='freeze')
        parser.add_argument('--peft_config', type=str, default=None)
        parser.add_argument('--peft_dir', type=str, default='')

        parser.add_argument('--save_every_n_epochs', type=int, default=1)
        ## quantization
        parser.add_argument('--load_in_8bit', action='store_true', default=False)

        ## lora config
        parser.add_argument('--lora_r', type=int, default=8)
        parser.add_argument('--lora_alpha', type=int, default=32)
        parser.add_argument('--lora_dropout', type=int, default=0.1)

        # optimization
        parser.add_argument('--reaction_weight', type=float, default=1.0)
        parser.add_argument('--weight_decay', type=float, default=0.05, help='optimizer weight decay')
        parser.add_argument('--init_lr', type=float, default=1e-4, help='optimizer init learning rate')
        parser.add_argument('--min_lr', type=float, default=1e-5, help='optimizer min learning rate')
        parser.add_argument('--warmup_lr', type=float, default=1e-6, help='optimizer warmup learning rate')
        parser.add_argument('--warmup_steps', type=int, default=1000, help='optimizer warmup steps')
        parser.add_argument('--lr_decay_rate', type=float, default=0.9, help='optimizer lr decay rate')
        parser.add_argument('--scheduler', type=str, default='linear_warmup_cosine_lr', help='type of scheduler') # or linear_warmup_step_lr
        parser.add_argument('--optimizer', type=str, default='adamw', help='type of scheduler')
        parser.add_argument('--stage1_path', type=str, default='')
        parser.add_argument('--stage2_path', type=str, default='')
        parser.add_argument('--init_checkpoint', type=str, default='')
        parser.add_argument('--caption_eval_epoch', type=int, default=10)
        
        # Attention regularization
        parser.add_argument('--att_reg', action='store_true', default=False, help='use attention regularization or not')
        parser.add_argument('--att_reg_method', type=str, default='cos', help="type of attention regularization ['cos', 'kl', 'l2']", choices=['cos', 'kl', 'l2'])
        parser.add_argument('--att_reg_lambda', type=float, default=0.1, help='weight of attention regularization')

        # Query token test
        parser.add_argument('--shuffle_query', action='store_true', default=False)
        parser.add_argument('--zero_query', action='store_true', default=False)
        parser.add_argument('--query_index', type=int, default=-1)
        return parent_parser


