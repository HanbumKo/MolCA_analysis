from rdkit import Chem
from rdkit.Chem import rdChemReactions

def read_rsmi_file(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            # Strip any leading/trailing whitespace characters
            line = line.strip()
            if line:
                # Parse the reaction SMILES
                try:
                    reaction_smiles = line.split("\t")[0]
                    reaction = rdChemReactions.ReactionFromSmarts(line)
                    print("Reactants:", [Chem.MolToSmiles(reactant) for reactant in reaction.GetReactants()])
                    print("Products:", [Chem.MolToSmiles(product) for product in reaction.GetProducts()])
                except Exception as e:
                    print(f"Error parsing line: {line}")
                    print(e)

# Example usage
file_path = 'data/USPTO/1976_Sep2016_USPTOgrants_smiles.rsmi'
read_rsmi_file(file_path)
