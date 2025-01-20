import json



n_shot_indices = {
    "forward": [
        [ 68007, 117631, 110806,  12507,  90314,  85034,  21948,  63507,  58936,  42481], # Seed 0
        [110380,  57767,   5210,  52537,  78208,  41492,  84434,  27686,  85363,  72254], # Seed 1
        [123018, 113459,  68692,  40973,  10737,   9162, 120447,  38970,  31720,  10615], # Seed 2
        [  3975,  67723,  98728,  39088,  33407,  91390,   3255,  86443,  40078,  53187], # Seed 3
    ],
    "retro": [
        [ 95111,  76087,  73572,  38415,  90519,  76941,  13845,  76137, 122776,  39798], # Seed 0
        [107191,  73947,  49590,  11030, 124896,  27047,  94889,  22705, 113058,  73661], # Seed 1
        [  4627,  67235, 108075,  85037, 122617, 126780,  42660,   4510,  16475,   5489], # Seed 2
        [ 18802, 125942,  43866,  30621,  36661,  28829,   1791,  75190,  80399, 109299], # Seed 3
    ],
    "reagent": [
        [ 29865,  38926,  24317,  31727,  28592,  30285,   2766,    388,  32816,  44858], # Seed 0
        [ 56968,  22898,  47068,   8615,  52197,  45646,  34943,  12647,  37485,  37402], # Seed 1
        [ 55361,  13570,  11903,  43322,  53862,  52297,  10586,  17279,  40066,   6928], # Seed 2
        [ 11200,  44758,  33218,  19854,  29028,  30069,  53427,   6425,  42490,  48422], # Seed 3
    ],
    "catalyst": [
        [  4047,   6527,   9810,   2697,   6344,    662,   3841,   1744,   6624,   9577], # Seed 0
        [  4111,  10100,  10002,   9939,   2252,   9122,   8894,   3759,   6852,   5361], # Seed 1
        [  8441,   2040,   4556,   9647,   7162,   9920,    677,   6107,   9879,   1543], # Seed 2
        [  3618,   4217,   9311,   1637,   4803,   3075,   7599,   5323,   6984,   7422], # Seed 3
    ],
    "solvent": [
        [  4721,  10575,  11841,  31973,  13721,  25730,  47724,  68704,  29500,  13373], # Seed 0
        [ 42088,    458,  14606,  29001,  29840,   6709,  53310,  10261,  23381,  57110], # Seed 1
        [ 15740,  28080,  11761,   9362,  35394,  34529,  30720,  68803,  28544,  29689], # Seed 2
        [ 34902,  66293,  53970,  67347,  35434,  59133,  13931,  60174,  31764,  22231], # Seed 3
    ],
}


# task_names = ["forward", "retro", "reagent", "catalyst", "solvent"]
task_names = ["solvent"]
n_test_samples = 100
answer_max_length = 200

# 1. Load n-shot examples
reasoning_texts = {task_name: [[], [], [], []] for task_name in task_names}
system_prompts = {task_name: [[], [], [], []] for task_name in task_names}
user_prompts = {task_name: [[], [], [], []] for task_name in task_names}
groudn_truths = {task_name: [[], [], [], []] for task_name in task_names}

for task_name in task_names:
    file_name = f"CoT_experiments/data/presto_reasoning_data/{task_name}/train.json"
    with open(file_name, 'r') as f:
        data = json.load(f)
    for seed, idx in enumerate(n_shot_indices[task_name]):
        for i in idx:
            reasoning_texts[task_name][seed].append(data[i]['reasoning'])
            system_prompts[task_name][seed].append(data[i]['system_prompt'])
            user_prompts[task_name][seed].append(data[i]['user_prompt'].replace(" .", ".").replace("[START_I_SMILES]", "").replace("[END_I_SMILES]", "").strip())
            if task_name == "forward":
                groudn_truths[task_name][seed].append(data[i]['product'])
            elif task_name == "retro":
                groudn_truths[task_name][seed].append(data[i]['reactants'])
            elif task_name == "reagent":
                groudn_truths[task_name][seed].append(data[i]['reagents'])
            elif task_name == "catalyst":
                groudn_truths[task_name][seed].append(data[i]['catalyst'])
            elif task_name == "solvent":
                groudn_truths[task_name][seed].append(data[i]['solvent'])
            # print task_name and the reasoning text
            print(f"=== {task_name} ===")
            print(f"# Question")
            print(f"{user_prompts[task_name][seed][-1]}")
            print()
            print(f"# Reasoning")
            print(f"{data[i]['reasoning']}")
            print()
            print(f"# Ground truth")
            print(f"{groudn_truths[task_name][seed][-1]}")
            print("_"*100)


