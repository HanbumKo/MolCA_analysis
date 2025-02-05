import json



forward_system_prompt = """You are a chemist. Now you are given a reaction equation. Please predict the product of the reaction. The reaction equation has the following format:\n```\nreactant1.reactant2. ... .reactantN>>product\n```\nYour task is to predict the structure representation of the product molecule. We provide the SMILES of the reactants."""
retro_system_prompt = """You are a chemist. Now you are given a product molecule. Please predict the the reactant molecules of the reaction.\nThe reaction equation has the following format:\n```\nreactant1.reactant2. ... .reactantN>>product\n```\nYour task is to predict the structure representation of the reactant molecule. We provide the SMILES of the products."""
reagent_system_prompt = """You are a chemist. Now you are given a reaction equation. Please predict the possible reagents of the reaction. The reaction equation has the following format:\n```\nreactant1.reactant2. ... .reactantN>>product\n```\nYour task is to predict the structure representation of the reagents molecule. We provide the SMILES of the reactions."""
catalyst_system_prompt = """You are a chemist. Now you are given a reaction equation. Please predict the possible catalyst of the reaction. The reaction equation has the following format:\n```\nreactant1.reactant2. ... .reactantN>>product\n```\nYour task is to predict the structure representation of the catalyst. We provide the SMILES of the reactions."""
solvent_system_prompt = """You are a chemist. Now you are given a reaction equation. Please predict the possible solvents of the reaction. The reaction equation has the following format:\n```\nreactant1.reactant2. ... .reactantN>>product\n```\nYour task is to predict the structure representation of the solvents. We provide the SMILES of the reactions."""


for task_name in ["forward", "retro", "reagent", "catalyst", "solvent"]:
    for split in ["train", "valid", "test"]:
        with open(f'/home/hko/MolCA_analysis/CoT_experiments/data/presto_reasoning_data/{task_name}/{split}.json') as f:
            data = json.load(f)
        for d in data:
            if task_name == "forward":
                d['system_prompt'] = forward_system_prompt
            elif task_name == "retro":
                d['system_prompt'] = retro_system_prompt
            elif task_name == "reagent":
                d['system_prompt'] = reagent_system_prompt
            elif task_name == "catalyst":
                d['system_prompt'] = catalyst_system_prompt
            elif task_name == "solvent":
                d['system_prompt'] = solvent_system_prompt
        # Save the data
        with open(f'/home/hko/MolCA_analysis/CoT_experiments/data/presto_reasoning_data/{task_name}/{split}.json', 'w') as f:
            json.dump(data, f, indent=4)


