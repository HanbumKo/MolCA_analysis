import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from bertviz import head_view, model_view
from transformers import AutoTokenizer, OPTForCausalLM
from copy import deepcopy

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("facebook/galactica-1.3b")
model = OPTForCausalLM.from_pretrained("facebook/galactica-1.3b", device_map="cuda", output_attentions=True)

input_text = "The Transformer architecture [START_REF]"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")

with torch.no_grad():
    outputs = model.generate(input_ids, output_attentions=True, return_dict_in_generate=True)


generated_ids = outputs.sequences
attentions = outputs.attentions

n_layers = len(attentions[0])
n_headers = attentions[0][0].shape[1]
seq_len = generated_ids.shape[1]
input_seq_len = input_ids.shape[1]
output_seq_len = seq_len - input_seq_len

attention_layers = [torch.zeros(1, n_headers, seq_len, seq_len) for _ in range(n_layers)]
for seq_i in range(output_seq_len):
    for layer_i in range(n_layers):
        # print(attentions[seq_i][layer_i].shape)
        if seq_i == 0:
            attention_layers[layer_i][0, :, :input_seq_len, :input_seq_len] = deepcopy(attentions[seq_i][layer_i][0, :, :, :])
        else:
            attention_layers[layer_i][0, :, seq_i+input_seq_len-1, :seq_i+input_seq_len] = deepcopy(attentions[seq_i][layer_i][0, :, 0, :])
print()