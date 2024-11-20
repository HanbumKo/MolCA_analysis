# python stage2.py --root "data/PubChem324kV2/" --devices "0,1,2,3,4,5,6,7" --filename "ft_pubchem324k_from_nostage1" --stage2_path "all_checkpoints/stage2_nostage1/last.ckpt" --max_epochs 100 --mode ft --prompt "[START_I_SMILES]{}[END_I_SMILES]." --tune_gnn --llm_tune lora --inference_batch_size 8 --peft_config lora_config.json --caption_eval_epoch 10 --batch_size 16 --accumulate_grad_batches 2 --text_max_len 128 --max_len 512
# python stage2.py --root "data/ChEBI-20_data/" --devices "0,1,2,3,4,5,6,7" --filename "ft_chebi20_from_nostage1" --stage2_path "all_checkpoints/stage2_nostage1/last.ckpt" --max_epochs 100 --mode ft --prompt "[START_I_SMILES]{}[END_I_SMILES]." --tune_gnn --llm_tune lora --inference_batch_size 8 --peft_config lora_config.json --caption_eval_epoch 10 --batch_size 16 --accumulate_grad_batches 2 --text_max_len 128 --max_len 512
# python stage2.py --root "data/PubChem324kV2/" --devices "0,1,2,3,4,5,6,7" --filename "ft_iupac_from_nostage1" --stage2_path "all_checkpoints/stage2_nostage1/last.ckpt" --max_epochs 100 --mode ft --prompt "[START_I_SMILES]{}[END_I_SMILES].The molecule's IUPAC name is " --tune_gnn --llm_tune lora --inference_batch_size 8 --peft_config lora_config.json --caption_eval_epoch 10 --iupac_prediction --batch_size 16 --accumulate_grad_batches 2 --text_max_len 128 --max_len 512
# python stage2.py --root "data/property_prediction/" --devices "0,1,2,3,4,5,6,7" --filename "ft_property_from_nostage1" --stage2_path "all_checkpoints/stage2_nostage1/last.ckpt" --max_epochs 40 --mode ft --prompt "[START_I_SMILES]{}[END_I_SMILES]" --tune_gnn --llm_tune lora --inference_batch_size 8 --peft_config lora_config.json --caption_eval_epoch 4 --batch_size 16 --accumulate_grad_batches 2 --text_max_len 128 --max_len 128
# python stage2.py --root "data/forward_reaction_prediction/" --devices "0,1,2,3,4,5,6,7" --filename "ft_forward_from_nostage1" --stage2_path "all_checkpoints/stage2_nostage1/last.ckpt" --max_epochs 40 --mode "ft" --tune_gnn --llm_tune "lora" --inference_batch_size 4 --peft_config "lora_config.json" --max_len 600 --batch_size 4 --accumulate_grad_batches 8 --caption_eval_epoch 4
# # python stage2.py --root "data/reagent_prediction/" --devices "0,1,2,3,4,5,6,7" --filename "ft_reagent_from_nostage1" --stage2_path "all_checkpoints/stage2_nostage1/last.ckpt" --max_epochs 40 --mode "ft" --tune_gnn --llm_tune "lora" --inference_batch_size 4 --peft_config "lora_config.json" --max_len 600 --batch_size 4 --accumulate_grad_batches 8 --caption_eval_epoch 4
# python stage2.py --root "data/retrosynthesis/" --devices "0,1,2,3,4,5,6,7" --filename "ft_retrosynthesis_from_nostage1" --stage2_path "all_checkpoints/stage2_nostage1/last.ckpt" --max_epochs 40 --mode "ft" --tune_gnn --llm_tune "lora" --inference_batch_size 4 --peft_config "lora_config.json" --max_len 600 --batch_size 4 --accumulate_grad_batches 8 --caption_eval_epoch 4


# String only training
# python stage2.py --opt_model only --root "data/PubChem324kV2/" --devices "0,1,2,3,4,5,6,7" --filename "ft_pubchem324k_stringonly" --max_epochs 100 --mode ft --prompt "[START_I_SMILES]{}[END_I_SMILES]." --tune_gnn --llm_tune lora --inference_batch_size 8 --peft_config lora_config.json --caption_eval_epoch 10 --batch_size 16 --accumulate_grad_batches 2 --text_max_len 128 --max_len 512
# python stage2.py --opt_model only --root "data/ChEBI-20_data/" --devices "0,1,2,3,4,5,6,7" --filename "ft_chebi20_stringonly" --max_epochs 100 --mode ft --prompt "[START_I_SMILES]{}[END_I_SMILES]." --tune_gnn --llm_tune lora --inference_batch_size 8 --peft_config lora_config.json --caption_eval_epoch 10 --batch_size 16 --accumulate_grad_batches 2 --text_max_len 128 --max_len 512
# python stage2.py --opt_model only --root "data/PubChem324kV2/" --devices "0,1,2,3,4,5,6,7" --filename "ft_iupac_stringonly" --max_epochs 100 --mode ft --prompt "[START_I_SMILES]{}[END_I_SMILES].The molecule's IUPAC name is " --tune_gnn --llm_tune lora --inference_batch_size 8 --peft_config lora_config.json --caption_eval_epoch 10 --iupac_prediction --batch_size 16 --accumulate_grad_batches 2 --text_max_len 128 --max_len 512
# python stage2.py --opt_model only --root "data/property_prediction/" --devices "0,1,2,3,4,5,6,7" --filename "ft_property_stringonly" --max_epochs 40 --mode ft --prompt "[START_I_SMILES]{}[END_I_SMILES]" --tune_gnn --llm_tune lora --inference_batch_size 8 --peft_config lora_config.json --caption_eval_epoch 4 --batch_size 16 --accumulate_grad_batches 2 --text_max_len 128 --max_len 128
# python stage2.py --opt_model only --root "data/forward_reaction_prediction/" --devices "0,1,2,3,4,5,6,7" --filename "ft_forward_stringonly" --max_epochs 40 --mode "ft" --tune_gnn --llm_tune "lora" --inference_batch_size 4 --peft_config "lora_config.json" --max_len 600 --batch_size 4 --accumulate_grad_batches 8 --caption_eval_epoch 4
# python stage2.py --opt_model only --root "data/reagent_prediction/" --devices "0,1,2,3,4,5,6,7" --filename "ft_reagent_stringonly" --max_epochs 40 --mode "ft" --tune_gnn --llm_tune "lora" --inference_batch_size 4 --peft_config "lora_config.json" --max_len 600 --batch_size 4 --accumulate_grad_batches 8 --caption_eval_epoch 4
# python stage2.py --opt_model only --root "data/retrosynthesis/" --devices "0,1,2,3,4,5,6,7" --filename "ft_retrosynthesis_stringonly" --max_epochs 40 --mode "ft" --tune_gnn --llm_tune "lora" --inference_batch_size 4 --peft_config "lora_config.json" --max_len 600 --batch_size 4 --accumulate_grad_batches 8 --caption_eval_epoch 4


# Original training, with USPTO forward/retro
python stage2.py --root "data/USPTO_retrosynthesis/USPTO_50K_data/" --devices "0,1,2,3,4,5,6,7" --filename "ft_USPTO_retrosynthesis_from_origin" --stage2_path "all_checkpoints/stage2_origin/last.ckpt" --max_epochs 100 --mode "ft" --tune_gnn --llm_tune "lora" --inference_batch_size 4 --peft_config "lora_config.json" --max_len 600 --batch_size 4 --accumulate_grad_batches 8 --caption_eval_epoch 10
python stage2.py --root "data/USPTO_forward/USPTO_50K_data/" --devices "0,1,2,3,4,5,6,7" --filename "ft_USPTO_forward_from_origin" --stage2_path "all_checkpoints/stage2_origin/last.ckpt" --max_epochs 100 --mode "ft" --tune_gnn --llm_tune "lora" --inference_batch_size 4 --peft_config "lora_config.json" --max_len 600 --batch_size 4 --accumulate_grad_batches 8 --caption_eval_epoch 10