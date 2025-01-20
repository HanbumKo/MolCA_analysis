import json


# load CoT_experiments/data/fewshot_example_all.json
with open("CoT_experiments/data/fewshot_example_all.json", "r") as f:
    data = json.load(f)


retro_reasoning_texts = [[], [], [], []]
catalyst_reasoning_texts = [[], [], [], []]



reasoning_text_all = []
# Load CoT_experiments/data/openai_batch/responses/tmp.jsonl
with open("CoT_experiments/data/openai_batch/responses/tmp.jsonl", "r") as f:
    tmp_data = f.readlines()
    tmp_data = [json.loads(d) for d in tmp_data]

for tmp in tmp_data:
    reasoning_text_all.append(tmp['response']['body']['choices'][0]['message']['content'])

retro_reasoning_texts[0] = reasoning_text_all[:10]
retro_reasoning_texts[1] = reasoning_text_all[10:20]
retro_reasoning_texts[2] = reasoning_text_all[20:30]
retro_reasoning_texts[3] = reasoning_text_all[30:40]

data['reasoning_texts']['retro'][0] = retro_reasoning_texts[0]
data['reasoning_texts']['retro'][1] = retro_reasoning_texts[1]
data['reasoning_texts']['retro'][2] = retro_reasoning_texts[2]
data['reasoning_texts']['retro'][3] = retro_reasoning_texts[3]


# Save to CoT_experiments/data/fewshot_example_all.json
with open("CoT_experiments/data/fewshot_example_all_tmp.json", "w") as f:
    json.dump(data, f, indent=4)
