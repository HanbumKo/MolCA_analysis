import json




# import /home/hko/PRESTO/logs/presto_origin/forward-0-shot/full_prompts.json
with open('/home/hko/PRESTO/logs/presto_origin/forward-0-shot/full_prompts.json') as f:
    full_prompts_presto = json.load(f)


with open('/home/hko/PRESTO/logs/presto_origin/forward-0-shot/reasonings.json') as f:
    reasoning_presto = json.load(f)


with open('/home/hko/PRESTO/logs/presto_origin/forward-0-shot/predictions.json') as f:
    predictions_presto = json.load(f)


with open('/home/hko/MolCA_analysis/CoT_experiments/results/zeroshot_test/raw_answer_reasoning.json') as f:
    raw_answer_reasoning = json.load(f)


for i in range(100):
    raw_answer_reasoning['forward'][f"instance_{i}"]["PRESTO"] = {
        "full_prompt": full_prompts_presto[i],
        "cot_answer": predictions_presto[i],
        "cot_reasoning_text": reasoning_presto[i],
    }


with open('/home/hko/MolCA_analysis/CoT_experiments/results/zeroshot_test/raw_answer_reasoning.json', 'w') as f:
    json.dump(raw_answer_reasoning, f, indent=4)