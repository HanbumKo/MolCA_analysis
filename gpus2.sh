# python stage2.py --root "data/PubChem324kV2/" --devices "0,1,2,3,4,5,6,7" --filename "ft_pubchem324k_noft" --max_epochs 100 --mode ft --prompt "[START_I_SMILES]{}[END_I_SMILES]." --tune_gnn --llm_tune lora --inference_batch_size 8 --peft_config lora_config.json --caption_eval_epoch 10 --batch_size 16 --accumulate_grad_batches 2 --text_max_len 128 --max_len 512
# python stage2.py --root "data/ChEBI-20_data/" --devices "0,1,2,3,4,5,6,7" --filename "ft_chebi20_from_noft" --max_epochs 100 --mode ft --prompt "[START_I_SMILES]{}[END_I_SMILES]." --tune_gnn --llm_tune lora --inference_batch_size 8 --peft_config lora_config.json --caption_eval_epoch 10 --batch_size 16 --accumulate_grad_batches 2 --text_max_len 128 --max_len 512
# python stage2.py --root "data/PubChem324kV2/" --devices "0,1,2,3,4,5,6,7" --filename "ft_iupac_from_noft" --max_epochs 100 --mode ft --prompt "[START_I_SMILES]{}[END_I_SMILES].The molecule's IUPAC name is " --tune_gnn --llm_tune lora --inference_batch_size 8 --peft_config lora_config.json --caption_eval_epoch 10 --iupac_prediction --batch_size 16 --accumulate_grad_batches 2 --text_max_len 128 --max_len 512
# python stage2.py --root "data/property_prediction/" --devices "0,1,2,3,4,5,6,7" --filename "ft_property_from_noft" --max_epochs 40 --mode ft --prompt "[START_I_SMILES]{}[END_I_SMILES]" --tune_gnn --llm_tune lora --inference_batch_size 8 --peft_config lora_config.json --caption_eval_epoch 4 --batch_size 16 --accumulate_grad_batches 2 --text_max_len 128 --max_len 128
# python stage2.py --root "data/forward_reaction_prediction/" --devices "0,1,2,3,4,5,6,7" --filename "ft_forward_from_noft" --max_epochs 40 --mode "ft" --tune_gnn --llm_tune "lora" --inference_batch_size 4 --peft_config "lora_config.json" --max_len 600 --batch_size 4 --accumulate_grad_batches 8 --caption_eval_epoch 4
# # python stage2.py --root "data/reagent_prediction/" --devices "0,1,2,3,4,5,6,7" --filename "ft_reagent_from_noft" --max_epochs 40 --mode "ft" --tune_gnn --llm_tune "lora" --inference_batch_size 4 --peft_config "lora_config.json" --max_len 600 --batch_size 4 --accumulate_grad_batches 8 --caption_eval_epoch 4
# python stage2.py --root "data/retrosynthesis/" --devices "0,1,2,3,4,5,6,7" --filename "ft_retrosynthesis_from_noft" --max_epochs 40 --mode "ft" --tune_gnn --llm_tune "lora" --inference_batch_size 4 --peft_config "lora_config.json" --max_len 600 --batch_size 4 --accumulate_grad_batches 8 --caption_eval_epoch 4


python stage1.py --root 'data/PubChem324kV2/' --gtm --lm --devices "0,1,2,3,4,5,6,7" --mode train --filename stage1_origin_gnnfreeze --rerank_cand_num 128 --num_query_token 8
python stage2.py --root 'data/PubChem324kV2/' --devices "0,1,2,3,4,5,6,7" --filename "stage2_origin_gnnfreeze" --stage1_path "all_checkpoints/stage1_origin_gnnfreeze/last.ckpt" --opt_model 'facebook/galactica-1.3b' --max_epochs 10 --mode pretrain --prompt '[START_I_SMILES]{}[END_I_SMILES].' --llm_tune freeze --inference_batch_size 4 --batch_size 16 --accumulate_grad_batches 2 --text_max_len 128
python stage2.py --root "data/PubChem324kV2/" --devices "0,1,2,3,4,5,6,7" --filename "ft_pubchem324k_from_origin_gnnfreeze" --stage2_path "all_checkpoints/stage2_origin_gnnfreeze/last.ckpt" --max_epochs 100 --mode ft --prompt "[START_I_SMILES]{}[END_I_SMILES]." --llm_tune lora --inference_batch_size 8 --peft_config lora_config.json --caption_eval_epoch 10 --batch_size 16 --accumulate_grad_batches 2 --text_max_len 128 --max_len 512
python stage2.py --root "data/ChEBI-20_data/" --devices "0,1,2,3,4,5,6,7" --filename "ft_chebi20_from_origin_gnnfreeze" --stage2_path "all_checkpoints/stage2_origin_gnnfreeze/last.ckpt" --max_epochs 100 --mode ft --prompt "[START_I_SMILES]{}[END_I_SMILES]." --llm_tune lora --inference_batch_size 8 --peft_config lora_config.json --caption_eval_epoch 10 --batch_size 16 --accumulate_grad_batches 2 --text_max_len 128 --max_len 512
python stage2.py --root "data/PubChem324kV2/" --devices "0,1,2,3,4,5,6,7" --filename "ft_iupac_from_origingnnfreeze" --stage2_path "all_checkpoints/stage2_origin_gnnfreeze/last.ckpt" --max_epochs 100 --mode ft --prompt "[START_I_SMILES]{}[END_I_SMILES].The molecule's IUPAC name is " --llm_tune lora --inference_batch_size 8 --peft_config lora_config.json --caption_eval_epoch 10 --iupac_prediction --batch_size 16 --accumulate_grad_batches 2 --text_max_len 128 --max_len 512
python stage2.py --root "data/property_prediction/" --devices "0,1,2,3,4,5,6,7" --filename "ft_property_from_origin_gnnfreeze" --stage2_path "all_checkpoints/stage2_origin_gnnfreeze/last.ckpt" --max_epochs 40 --mode ft --prompt "[START_I_SMILES]{}[END_I_SMILES]" --llm_tune lora --inference_batch_size 8 --peft_config lora_config.json --caption_eval_epoch 4 --batch_size 16 --accumulate_grad_batches 2 --text_max_len 128 --max_len 128
python stage2.py --root "data/forward_reaction_prediction/" --devices "0,1,2,3,4,5,6,7" --filename "ft_forward_from_origin_gnnfreeze" --stage2_path "all_checkpoints/stage2_origin_gnnfreeze/last.ckpt" --max_epochs 40 --mode "ft" --llm_tune "lora" --inference_batch_size 4 --peft_config "lora_config.json" --max_len 600 --batch_size 4 --accumulate_grad_batches 8 --caption_eval_epoch 4
python stage2.py --root "data/reagent_prediction/" --devices "0,1,2,3,4,5,6,7" --filename "ft_reagent_from_origin_gnnfreeze" --stage2_path "all_checkpoints/stage2_origin_gnnfreeze/last.ckpt" --max_epochs 40 --mode "ft" --llm_tune "lora" --inference_batch_size 4 --peft_config "lora_config.json" --max_len 600 --batch_size 4 --accumulate_grad_batches 8 --caption_eval_epoch 4
python stage2.py --root "data/retrosynthesis/" --devices "0,1,2,3,4,5,6,7" --filename "ft_retrosynthesis_from_origin_gnnfreeze" --stage2_path "all_checkpoints/stage2_origin_gnnfreeze/last.ckpt" --max_epochs 40 --mode "ft" --llm_tune "lora" --inference_batch_size 4 --peft_config "lora_config.json" --max_len 600 --batch_size 4 --accumulate_grad_batches 8 --caption_eval_epoch 4
