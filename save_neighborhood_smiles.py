from data_provider.property_prediction_dataset import PropertyPrediction
from data_provider.stage2_dm import Stage2DM
from data_provider.iupac_dm import IupacDM
from data_provider.iupac_hard_dm import IupacHardDM
from data_provider.stage2_chebi_dm import Stage2CheBIDM
from data_provider.property_prediction_dm import PropertyPredictionDM
from data_provider.forward_reaction_prediction_dm import ForwardReactionPredictionDM, USPTOForwardReactionPredictionDM
from data_provider.reagent_prediction_dm import ReagentPredictionDM
from data_provider.retrosynthesis_dm import RetrosynthesisDM, USPTORetrosynthesisDM
from transformers import AutoTokenizer


all_smiles = set()


train_dataset = PropertyPrediction('data/PubChem324kV2/train.pt', 128, "dummy{}dummy")
val_dataset = PropertyPrediction('data/PubChem324kV2/valid.pt', 128, "dummy{}dummy")
test_dataset = PropertyPrediction('data/PubChem324kV2/test.pt', 128, "dummy{}dummy")
all_smiles.update(train_dataset.data_list[0].smiles)
all_smiles.update(val_dataset.data_list[0].smiles)
all_smiles.update(test_dataset.data_list[0].smiles)

