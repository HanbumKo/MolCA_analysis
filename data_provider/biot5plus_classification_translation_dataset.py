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


def split_smiles_with_separator(smiles):
    result = "SPL1T-TH1S-Pl3A5E".join(smiles)
    return result


class ClassificationTranslationDataset(InMemoryDataset):
    def __init__(self, type, shuffle_inst=True, prompt=None):
        super(ClassificationTranslationDataset, self).__init__()
        files = [
            # (glob(f"data/biot5_plus_data/tasks_plus/*_chebi20_text2mol_{type}.json")[0], "chebi20_text2mol"),
            # (glob(f"data/biot5_plus_data/tasks_plus/*_chebi20_mol2text_{type}.json")[0], "chebi20_mol2text",) ,
            (glob(f"data/biot5_plus_data/tasks_plus/*_bbbp_molnet_{type}.json")[0] , "bbbp"),
            (glob(f"data/biot5_plus_data/tasks_plus/*_hiv_molnet_{type}.json")[0], "hiv"),
            (glob(f"data/biot5_plus_data/tasks_plus/*_bace_molnet_{type}.json")[0], "bace"),
            (glob(f"data/biot5_plus_data/tasks_plus/*_clintox_fda_approved_molnet_{type}.json")[0], "clintox"),
            (glob(f"data/biot5_plus_data/tasks_plus/*_clintox_ct_tox_molnet_{type}.json")[0], "clintox"),
            # (glob(f"data/biot5_plus_data/tasks_plus/*_molecular_description_generation_molinst_mol_{type}.json")[0], "pubchem_mol2text"),
            # (glob(f"data/biot5_plus_data/tasks_plus/*_description_guided_molecule_design_molinst_mol_{type}.json")[0], "pubchem_text2mol"),
        ]
        self.instructions = {
            "chebi20_text2mol": [
                "Generate a molecule based on this description.",
                "Based on the given information, design a molecule that meets the desired specifications.",
                "Create a molecule that satisfies the conditions outlined in the description.",
                "Design a molecule that meets the criteria outlined in the description.",
                "Create a molecule with the structure as the one described.",
                "Use the given information to create a molecule that fulfills the desired purpose.",
                "Generate a molecule based on the given description.",
                "Synthesize a molecule that matches the given characteristics.",
            ],
            "chebi20_mol2text": [
                "Describe this molecule.",
                "Provide a description of this molecule.",
                "What can you tell me about this molecule?",
                "Could you provide a description of this molecule?",
                "Could you give me a brief overview of this molecule?",
                "Provide a brief overview of this molecule.",
                "Please give me some details about this molecule.",
            ],
            "bbbp": [
                "Can this molecule pass through the blood-brain barrier (BBB)?",
                "Can this molecule permeate the blood-brain barrier?",
                "Does this molecule have the blood-brain barrier permeability (BBBP)?",
                "Does this molecule have the ability to penetrate the blood-brain barrier?",
                "Is this molecule capable of crossing the blood-brain barrier (BBB)?",
                "Can this molecule penetrate the blood-brain barrier?",
                "Is it likely for this molecule to effectively traverse the blood-brain barrier?",
                "Is blood-brain barrier permeability (BBBP) present in this molecule ?",
                "Does this molecule demonstrate the potential for blood-brain barrier permeation (BBBP)?",
                "Does blood-brain barrier permeability (BBBP) apply to this molecule ?",
                "Is blood-brain barrier permeability (BBBP) a property of this molecule ?",
                "Would this molecule be able to successfully permeate the blood-brain barrier?",
            ],
            "hiv": [
                "Is this molecule known to inhibit HIV replication?",
                "Does this molecule inhibit viral replication for HIV?",
                "Could this molecule be used to prevent HIV replication?",
                "Can this molecule inhibit the replication of human immunodeficiency virus (HIV)?",
                "Could HIV replication be slowed or stopped by this molecule ?",
                "Does this molecule have an inhibitory impact on HIV?",
                "Can this molecule effectively inhibit HIV replication?",
                "Does this molecule exhibit inhibitory effects on HIV replication?",
                "Is this molecule capable of suppressing HIV replication?",
                "Would this molecule have the ability to hinder HIV replication?",
                "Can this molecule serve as an inhibitor of HIV replication?",
                "Do you suggest that this molecule can impede the replication of HIV?",
                "this molecule Predict if the molecule given above have an inhibitory impact on HIV.",
            ],
            "bace": [
                "Does this molecule inhibit the BACE enzyme?",
                "Is this molecule a BACE enzyme inhibitor?",
                "Is BACE inhibition a property of this molecule?",
                "Can this molecule effectively inhibit the BACE enzyme?",
                "Does this molecule have the ability to inhibit BACE?",
                "Could this molecule be considered as a BACE inhibitor?",
                "Is this molecule capable of inhibiting BACE activity?",
                "Is it likely that this molecule will inhibit the BACE enzyme?",
                "Does BACE inhibition apply to this molecule?",
                "Is BACE inhibition observed in this molecule?",
                "Would you say that this molecule can serve as a BACE inhibitor?",
                "Is the inhibition of BACE one of this molecule's characteristics?",
                "Tell me if this molecule is a BACE inhibitor."
            ],
            "clintox": [
                "Is this molecule toxic?",
                "Is the molecule this molecule known to exhibit toxic properties in biological systems?",
                "Is this molecule classified as a hazardous substance with documented toxicity?",
                "Does this molecule have documented toxicity in biological systems?",
                "Is this molecule considered to be a toxic substance?",
                "Does this molecule have toxicity?",
                "Do you think that this molecule is toxic?",
                "Can this molecule be considered toxic?",
                "Is this molecule associated with harmful and toxic effects?",
                "Is the substance this molecule toxic for human?",
                "this molecule Would you say that the molecule given above is toxic?",
                "this molecule Is it toxic?",
                "Tell me if this molecule is toxic.",
            ],
            "pubchem_mol2text": [
                "Describe this molecule.",
                "Provide a description of this molecule.",
                "What can you tell me about this molecule?",
                "Could you provide a description of this molecule?",
                "Could you give me a brief overview of this molecule?",
                "Provide a brief overview of this molecule.",
                "Please give me some details about this molecule.",
            ],
            "pubchem_text2mol": [
                "Generate a molecule based on this description.",
                "Based on the given information, design a molecule that meets the desired specifications.",
                "Create a molecule that satisfies the conditions outlined in the description.",
                "Design a molecule that meets the criteria outlined in the description.",
                "Create a molecule with the structure as the one described.",
                "Use the given information to create a molecule that fulfills the desired purpose.",
                "Generate a molecule based on the given description.",
                "Synthesize a molecule that matches the given characteristics.",
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
                    d["output"] = d["output"][0]
                    self.dataset.append(d)
                    # if i == 50:
                    #     break

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
        if self.shuffle_inst:
            # Randomly select an instruction
            instruction = self.instructions[task][torch.randint(0, len(self.instructions[task]), (1,)).item()]
        else:
            instruction = self.instructions[task][index % len(self.instructions[task])]

        if task == "chebi20_text2mol":
            mol_desc = input
            # selfies = output.split("<bom>")[1].split("<eom>")[0]
            # isomeric_smiles = self._selfies_to_smiles(selfies)
            isomeric_smiles = output.split("[START_I_SMILES]")[1].split("[END_I_SMILES]")[0]
            label_prompt = f"[START_I_SMILES]{isomeric_smiles}[END_I_SMILES]"
            input_prompt = f"Question: {instruction}\n{mol_desc}\n\nAnswer: \n\n"
            graph_list = []
        elif task == "chebi20_mol2text":
            iupac = input.split("<boi>")[1].split("<eoi>")[0]
            # selfies = input.split("<bom>")[1].split("<eom>")[0]
            # isomeric_smiles = self._selfies_to_smiles(selfies)
            isomeric_smiles = input.split("[START_I_SMILES]")[1].split("[END_I_SMILES]")[0]
            label_prompt = f"{output}"
            input_prompt = f"[START_I_SMILES]{isomeric_smiles[:self.smiles_max_length]}[END_I_SMILES]\n\nQuestion: {instruction}\n\nAnswer: \n\n"
            graph = smiles2data(isomeric_smiles)
            graph.instruction = instruction
            graph.smiles = isomeric_smiles
            graph_list = [graph]
        elif task == "bbbp":
            iupac = input.split("<boi>")[1].split("<eoi>")[0]
            # selfies = input.split("<bom>")[1].split("<eom>")[0]
            # isomeric_smiles = self._selfies_to_smiles(selfies)
            isomeric_smiles = input.split("[START_I_SMILES]")[1].split("[END_I_SMILES]")[0]
            label_prompt = f"{output}"
            input_prompt = f"[START_I_SMILES]{isomeric_smiles[:self.smiles_max_length]}[END_I_SMILES]\n\nQuestion: {instruction}\n\nAnswer: \n\n"
            graph = smiles2data(isomeric_smiles)
            graph.instruction = instruction
            graph.smiles = isomeric_smiles
            graph_list = [graph]
        elif task == "hiv":
            iupac = input.split("<boi>")[1].split("<eoi>")[0]
            # selfies = input.split("<bom>")[1].split("<eom>")[0]
            # isomeric_smiles = self._selfies_to_smiles(selfies)
            isomeric_smiles = input.split("[START_I_SMILES]")[1].split("[END_I_SMILES]")[0]
            label_prompt = f"{output}"
            input_prompt = f"[START_I_SMILES]{isomeric_smiles[:self.smiles_max_length]}[END_I_SMILES]\n\nQuestion: {instruction}\n\nAnswer: \n\n"
            graph = smiles2data(isomeric_smiles)
            graph.instruction = instruction
            graph.smiles = isomeric_smiles
            graph_list = [graph]
        elif task == "bace":
            iupac = input.split("<boi>")[1].split("<eoi>")[0]
            # selfies = input.split("<bom>")[1].split("<eom>")[0]
            # isomeric_smiles = self._selfies_to_smiles(selfies)
            isomeric_smiles = input.split("[START_I_SMILES]")[1].split("[END_I_SMILES]")[0]
            label_prompt = f"{output}"
            input_prompt = f"[START_I_SMILES]{isomeric_smiles[:self.smiles_max_length]}[END_I_SMILES]\n\nQuestion: {instruction}\n\nAnswer: \n\n"
            graph = smiles2data(isomeric_smiles)
            graph.instruction = instruction
            graph.smiles = isomeric_smiles
            graph_list = [graph]
        elif task == "clintox":
            iupac = input.split("<boi>")[1].split("<eoi>")[0]
            # selfies = input.split("<bom>")[1].split("<eom>")[0]
            # isomeric_smiles = self._selfies_to_smiles(selfies)
            isomeric_smiles = input.split("[START_I_SMILES]")[1].split("[END_I_SMILES]")[0]
            label_prompt = f"{output}"
            input_prompt = f"[START_I_SMILES]{isomeric_smiles[:self.smiles_max_length]}[END_I_SMILES]\n\nQuestion: {instruction}\n\nAnswer: \n\n"
            graph = smiles2data(isomeric_smiles)
            graph.instruction = instruction
            graph.smiles = isomeric_smiles
            graph_list = [graph]
        elif task == "pubchem_mol2text":
            iupac = input.split("<boi>")[1].split("<eoi>")[0]
            # selfies = input.split("<bom>")[1].split("<eom>")[0]
            # isomeric_smiles = self._selfies_to_smiles(selfies)
            isomeric_smiles = input.split("[START_I_SMILES]")[1].split("[END_I_SMILES]")[0]
            label_prompt = f"{output}"
            input_prompt = f"[START_I_SMILES]{isomeric_smiles[:self.smiles_max_length]}[END_I_SMILES]\n\nQuestion: {instruction}\n\nAnswer: \n\n"
            graph = smiles2data(isomeric_smiles)
            graph.instruction = instruction
            graph.smiles = isomeric_smiles
            graph_list = [graph]
        elif task == "pubchem_text2mol":
            mol_desc = input
            # selfies = output.split("<bom>")[1].split("<eom>")[0]
            # isomeric_smiles = self._selfies_to_smiles(selfies)
            isomeric_smiles = output.split("[START_I_SMILES]")[1].split("[END_I_SMILES]")[0]
            label_prompt = f"[START_I_SMILES]{isomeric_smiles}[END_I_SMILES]"
            input_prompt = f"Question: {instruction}\n{mol_desc}\n\nAnswer: \n\n"
            graph_list = []

        return graph_list, f"{label_prompt}\n", input_prompt, task


if __name__ == '__main__':
    dataset = ClassificationTranslationDataset('train')
    for d in dataset:
        # print(d)
        pass