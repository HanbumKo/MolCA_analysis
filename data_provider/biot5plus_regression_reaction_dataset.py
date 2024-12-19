import torch
import json
from torch_geometric.data import Dataset, InMemoryDataset, Data
import os
import selfies as sf
from rdkit import Chem
from ogb.utils import smiles2graph
from tqdm import tqdm
from glob import glob


SPLIT_MARKER = f"SPL{1}T-TH{1}S-Pl3A5E"

def smiles2data(smiles):
    graph = smiles2graph(smiles)
    x = torch.from_numpy(graph['node_feat'])
    edge_index = torch.from_numpy(graph['edge_index'], )
    edge_attr = torch.from_numpy(graph['edge_feat'])
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data


def enumerate_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol, isomericSmiles=True, kekuleSmiles=True, doRandom=True)


def split_float_with_separator(number):
    number_str = str(number)
    result = "SPL1T-TH1S-Pl3A5E".join(number_str)
    return result


class RegressionReactionDataset(InMemoryDataset):
    def __init__(self, type, shuffle_inst=True, prompt=None):
        super(RegressionReactionDataset, self).__init__()



        self.dataset = torch.load(f"data/property_prediction/{type}.pt")
        for d in self.dataset:
            d['output'] = d['y']
            d['task'] = "molinst_property"
            d['input'] = "[START_I_SMILES]" + d['smiles'] + "[END_I_SMILES]"


        """
        files = [
            # Regression tasks
            # (glob(f"data/biot5_plus_data/tasks_plus/*_property_prediction_molinst_mol_homo_{type}.json")[0], "homo_reg"),
            # (glob(f"data/biot5_plus_data/tasks_plus/*_property_prediction_molinst_mol_lumo_{type}.json")[0], "lumo_reg",) ,
            # (glob(f"data/biot5_plus_data/tasks_plus/*_property_prediction_molinst_mol_gap_{type}.json")[0] , "gap_reg"),
            (glob(f"data/biot5_plus_data/tasks_plus/*_property_prediction_molinst_mol_{type}.json")[0] , "molinst_property"),
            # Reaction tasks
            # (glob(f"data/biot5_plus_data/tasks_plus/*_reagent_prediction_molinst_mol_{type}.json")[0], "reagent"),
            # (glob(f"data/biot5_plus_data/tasks_plus/*_forward_reaction_prediction_molinst_mol_{type}.json")[0], "forward"),
            # (glob(f"data/biot5_plus_data/tasks_plus/*_retrosynthesis_molinst_mol_{type}.json")[0], "retro"),
        ]
        self.instructions = {
            "homo_reg": [
                "Please provide me with the HOMO energy value of this molecule.",
                "Please provide the highest occupied molecular orbital (HOMO) energy value for this molecule.",
                "Could you give me the HOMO energy value of this molecule?",
                "What is the HOMO energy of this molecule?",
                "Please provide the highest occupied molecular orbital (HOMO) energy of this molecule.",
                "I am interested in the HOMO energy of this molecule, could you tell me what it is?",
                "What is the highest occupied molecular orbital (HOMO) energy of this molecule?",
                "What is the HOMO level of energy for this molecule?",
                "I would like to know the highest occupied molecular orbital (HOMO) energy of this molecule, could you please provide it?",
                "I would like to know the HOMO energy of this molecule, could you please provide it?",
                "Please provide the HOMO energy value for this molecule.",
                "Can you tell me the value of the HOMO energy for this molecule?",
            ],
            "lumo_reg": [
                "Can you tell me the value of the LUMO energy for this molecule?",
                "What is the LUMO energy of this molecule?",
                "Please provide the LUMO energy value for this molecule.",
                "Please provide me with the LUMO energy value of this molecule.",
                "What is the lowest unoccupied molecular orbital (LUMO) energy of this molecule?",
                "What is the LUMO level of energy for this molecule?",
                "Could you give me the LUMO energy value of this molecule?",
                "I am interested in the LUMO energy of this molecule, could you tell me what it is?",
                "Please provide the lowest unoccupied molecular orbital (LUMO) energy of this molecule.",
                "Please provide the lowest unoccupied molecular orbital (LUMO) energy value for this molecule.",
                "I would like to know the lowest unoccupied molecular orbital (LUMO) energy of this molecule, could you please provide it?",
                "I would like to know the LUMO energy of this molecule, could you please provide it?",
            ],
            "gap_reg": [
                "Please provide the energy separation between the highest occupied and lowest unoccupied molecular orbitals (HOMO-LUMO gap) of this molecule.",
                "Can you give me the energy difference between the HOMO and LUMO orbitals of this molecule?",
                "What is the energy separation between the HOMO and LUMO of this molecule?",
                "What is the HOMO-LUMO gap of this molecule?",
                "Please give me the HOMO-LUMO gap energy for this molecule.",
                "I need to know the HOMO-LUMO gap energy of this molecule, could you please provide it?",
                "I would like to know the HOMO-LUMO gap of this molecule, can you provide it?",
                "Could you tell me the energy difference between HOMO and LUMO for this molecule?",
                "Please provide the gap between HOMO and LUMO of this molecule.",
            ],
            "reagent": [
                "Given this chemical reaction, what are some reagents that could have been used?",
                "Based on the given chemical reaction, suggest some possible reagents.",
                "Given the following chemical reaction, what are some potential reagents that could have been employed?",
                "What reagents could have been utilized in the following chemical reaction?",
                "Please propose potential reagents that might have been utilized in the provided chemical reaction.",
                "From the provided chemical reaction, propose some possible reagents that could have been used.",
                "Please provide possible reagents based on the following chemical reaction.",
                "Given the following reaction, what are some possible reagents that could have been utilized?",
                "Based on the given chemical reaction, can you propose some likely reagents that might have been utilized?",
                "Can you suggest some reagents that might have been used in the given chemical reaction?",
                "Can you provide potential reagents for the following chemical reaction?",
                "Please suggest some possible reagents that could have been used in the following chemical reaction.",
            ],
            "forward": [
                "Please provide a feasible product that could be formed using the given reactants and reagents.",
                "Given the reactants and reagents below, come up with a possible product.",
                "Using the listed reactants and reagents, offer a plausible product.",
                "Based on the given reactants and reagents, what product could potentially be produced?",
                "Using the provided reactants and reagents, can you propose a likely product?",
                "Please suggest a potential product based on the given reactants and reagents.",
                "Given the reactants and reagents provided, what is a possible product that can be formed?",
                "Given the following reactants and reagents, please provide a possible product.",
                "Based on the given reactants and reagents, suggest a possible product.",
                "Given the reactants and reagents listed, what could be a probable product of their reaction?",
                "What product could potentially form from the reaction of the given reactants and reagents?",
                "With the provided reactants and reagents, propose a potential product.",
            ],
            "retro": [
                "What reactants could lead to the production of the following product?",
                "What are the possible reactants that could have formed the following product?",
                "Which reactants could have been used to generate the given product?",
                "Given the product provided, propose some possible reactants that could have been employed in its formation.",
                "Based on the given product, provide some plausible reactants that might have been utilized to prepare it.",
                "With the given product, suggest some likely reactants that were used in its synthesis.",
                "Provided the product below, propose some possible reactants that could have been used in the reaction.",
                "Please suggest potential reactants for the given product.",
                "Given the following product, please provide possible reactants.",
                "With the provided product, recommend some probable reactants that were likely used in its production.",
                "Provide a list of potential reactants that may have produced the given product.",
                "Please suggest possible reactants for the given product.",
                "Please suggest potential reactants used in the synthesis of the provided product.",
                "Given these product, can you propose the corresponding reactants?",
                "Can you identify some reactants that might result in the given product?",
            ],
        }

        self.dataset = []
        for file, task in tqdm(files, desc="Loading data", total=len(files)):
            with open(file, 'r') as f:
                dataset_current = json.load(f)["Instances"]
                for i, d in enumerate(dataset_current):
                    # Drop "id"
                    d.pop("id")
                    d["task"] = task
                    d["instruction"] = d["instruction"]
                    d["output"] = d["output"][0]
                    self.dataset.append(d)
                    # if i == 50:
                    #     break
        """
        self.type = type
        self.smiles_max_length = 256
        # Load data pt file
        self.shuffle_inst = shuffle_inst
        self.prompt = ""
        self.perm = None

    def _selfies_to_smiles(self, selfies):
        # Convert SELFIES to SMILES
        smiles = sf.decoder(selfies)

        # Convert SMILES to RDKit molecule
        mol = Chem.MolFromSmiles(smiles)

        # Ensure the molecule is sanitized and has stereochemistry information
        Chem.SanitizeMol(mol)
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

        # Convert back to SMILES with stereochemistry (isomeric SMILES)
        isomeric_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        return isomeric_smiles

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        task = data["task"]
        input = data["input"]
        output = data["output"]
        # if self.shuffle_inst:
        #     # Randomly select an instruction
        #     instruction = self.instructions[task][torch.randint(0, len(self.instructions[task]), (1,)).item()]
        # else:
        #     instruction = self.instructions[task][index % len(self.instructions[task])]
        instruction = data["instruction"]

        if task == "homo_reg" or task == "lumo_reg" or task == "gap_reg" or task == "molinst_property":
            # iupac = input.split("<boi>")[1].split("<eoi>")[0]
            # selfies = input.split("<bom>")[1].split("<eom>")[0]
            # isomeric_smiles = self._selfies_to_smiles(selfies)
            isomeric_smiles = input.split("[START_I_SMILES]")[1].split("[END_I_SMILES]")[0]
            # if self.type == "train":
            #     isomeric_smiles = enumerate_smiles(isomeric_smiles)
            output = split_float_with_separator(output)
            label_prompt = f"{output}"
            # input_prompt = f"[START_I_SMILES]{isomeric_smiles[:self.smiles_max_length]}[END_I_SMILES]\n\nThe molecule's IUPAC name is {iupac}.\nQuestion: {instruction}\n\nAnswer: "
            input_prompt = f"[START_I_SMILES]{isomeric_smiles[:self.smiles_max_length]}[END_I_SMILES]\n\nQuestion: {instruction}\n\nAnswer: "
            graph = smiles2data(isomeric_smiles)
            graph.instruction = instruction
            graph.smiles = isomeric_smiles
            graph_list = [graph]
        """
        elif task == "reagent":
            iupac = input.split("<boi>")[1].split("<eoi>")[0]
            # selfies = input.split("<bom>")[1].split("<eom>")[0]
            # isomeric_smiles = self._selfies_to_smiles(selfies)
            isomeric_smiles = input.split("[START_I_SMILES]")[1].split("[END_I_SMILES]")[0]
            if self.type == "train":
                isomeric_smiles = enumerate_smiles(isomeric_smiles)
            label_prompt = f"{output}"
            # input_prompt = f"[START_I_SMILES]{isomeric_smiles[:self.smiles_max_length]}[END_I_SMILES] The molecule's IUPAC name is {iupac}.\n\nQuestion: {instruction}\n\nAnswer: "
            input_prompt = f"[START_I_SMILES]{isomeric_smiles[:self.smiles_max_length]}[END_I_SMILES]\n\nQuestion: {instruction}\n\nAnswer: "
            graph = smiles2data(isomeric_smiles)
            graph.instruction = instruction
            graph.smiles = isomeric_smiles
            graph_list = [graph]
        elif task == "forward":
            iupac = input.split("<boi>")[1].split("<eoi>")[0]
            # selfies = input.split("<bom>")[1].split("<eom>")[0]
            # isomeric_smiles = self._selfies_to_smiles(selfies)
            isomeric_smiles = input.split("[START_I_SMILES]")[1].split("[END_I_SMILES]")[0]
            if self.type == "train":
                isomeric_smiles = enumerate_smiles(isomeric_smiles)
            label_prompt = f"{output}"
            # input_prompt = f"[START_I_SMILES]{isomeric_smiles[:self.smiles_max_length]}[END_I_SMILES] The molecule's IUPAC name is {iupac}.\n\nQuestion: {instruction}\n\nAnswer: "
            input_prompt = f"[START_I_SMILES]{isomeric_smiles[:self.smiles_max_length]}[END_I_SMILES]\n\nQuestion: {instruction}\n\nAnswer: "
            graph = smiles2data(isomeric_smiles)
            graph.instruction = instruction
            graph.smiles = isomeric_smiles
            graph_list = [graph]
        elif task == "retro":
            iupac = input.split("<boi>")[1].split("<eoi>")[0]
            # selfies = input.split("<bom>")[1].split("<eom>")[0]
            # isomeric_smiles = self._selfies_to_smiles(selfies)
            isomeric_smiles = input.split("[START_I_SMILES]")[1].split("[END_I_SMILES]")[0]
            if self.type == "train":
                isomeric_smiles = enumerate_smiles(isomeric_smiles)
            label_prompt = f"{output}"
            # input_prompt = f"[START_I_SMILES]{isomeric_smiles[:self.smiles_max_length]}[END_I_SMILES] The molecule's IUPAC name is {iupac}.\n\nQuestion: {instruction}\n\nAnswer: "
            input_prompt = f"[START_I_SMILES]{isomeric_smiles[:self.smiles_max_length]}[END_I_SMILES]\n\nQuestion: {instruction}\n\nAnswer: "
            graph = smiles2data(isomeric_smiles)
            graph.instruction = instruction
            graph.smiles = isomeric_smiles
            graph_list = [graph]
        """
        return graph_list, f"{label_prompt}\n", input_prompt, task


if __name__ == '__main__':
    dataset = RegressionReactionDataset('train')
    for i, d in enumerate(dataset):
        print(d)
        if i == 5: break
        # print(d)
        pass