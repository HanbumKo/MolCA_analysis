import json
import copy
import random
import pandas as pd
import numpy as np
from sklearn import metrics
import random
from sklearn.linear_model import LogisticRegression
from rxnfp.transformer_fingerprints import (
    RXNBERTFingerprintGenerator, get_default_model_and_tokenizer, generate_fingerprints
)
from rdkit import Chem
from rdkit.Chem import AllChem

def remove_atom_mapping(atommaped_reaction):
    """
    """
    if ">>" in atommaped_reaction:
        mol_list = []
        reactants, products = atommaped_reaction.split(">>")
        for mol in [reactants, products]:
            mol = Chem.MolFromSmiles(mol)
            if mol is None:
                raise ValueError("유효하지 않은 SMILES 문자열입니다.")
            for atom in mol.GetAtoms():
                atom.SetAtomMapNum(0)
            mol = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
            mol_list.append(mol)
        return f"{mol_list[0]}>>{mol_list[1]}"
    else:
        mol_list = []
        reactants, reagents, products = atommaped_reaction.split(">")
        precursors = reactants + "." + reagents
        for mol in [precursors, products]:
            mol = Chem.MolFromSmiles(mol)
            if mol is None:
                raise ValueError("유효하지 않은 SMILES 문자열입니다.")
            for atom in mol.GetAtoms():
                atom.SetAtomMapNum(0)
            mol = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
            mol_list.append(mol)
        return f"{mol_list[0]}>>{mol_list[1]}"


def evaluate_model(model, fingerprints, corresponding_classes, all_classes, all_classes_names):
    
    preds = model.predict(fingerprints)
    predicted_classes = [all_classes[x] for x in preds]
    expected_classes =[all_classes[x] for x in corresponding_classes]
    print(metrics.classification_report(expected_classes, predicted_classes))

    confusion_matrix = metrics.confusion_matrix(corresponding_classes, preds)
    colCounts = confusion_matrix.sum(axis=0)
    rowCounts = confusion_matrix.sum(axis=1)

    print(' & recall & prec & F-score  &   reaction class &  \\\\ ')
    sum_recall=0
    sum_prec=0
    for i, rxn_class in enumerate(all_classes):
        recall = 0
        if rowCounts[i] > 0:
            recall = float(confusion_matrix[i,i])/rowCounts[i]
        sum_recall += recall
        prec = 0
        if colCounts[i] > 0:
            prec = float(confusion_matrix[i,i])/colCounts[i]
        sum_prec += prec
        f_score = 0
        if (recall + prec) > 0:
            f_score = 2 * (recall * prec) / (recall + prec)   
        print('{:2d} & {:.4f} & {:.4f} &{:.4f} & {:9s} &{:s} \\\\'.format(i, recall, prec, f_score, all_classes_names[rxn_class], rxn_class))
    
    mean_recall = sum_recall/len(all_classes)
    mean_prec = sum_prec/len(all_classes)
    if (mean_recall+mean_prec) > 0:
        mean_fscore = 2*(mean_recall*mean_prec)/(mean_recall+mean_prec)
    print(" &  {:.2f} & {:.2f} & {:.2f} & Average & \\\\ ".format(mean_recall,mean_prec,mean_fscore))
    
    return confusion_matrix

def labelled_cmat(confusion_matrix, labels,
                  figsize=(20,15), label_extras=None, 
                  dpi=300,threshold=0.01, 
                  xlabel=True, ylabel=True, rotation=90):
    from matplotlib import pyplot as plt
    
    rowCounts = confusion_matrix.sum(axis=1)
    cmat_percent = confusion_matrix/rowCounts[:,None]
    #zero all elements that are less than 1% of the row contents
    ncm = cmat_percent*(cmat_percent>threshold)

    fig = plt.figure(1,figsize=figsize,dpi=dpi)
    ax = fig.add_subplot(1,1,1)
    fig.set_size_inches(figsize)
    fig.set_dpi(dpi)
    # pax=ax.pcolor(ncm,cmap=cm.ocean_r)
    # pax=ax.pcolor(ncm,cmap='gist_earth_r')

    pax=ax.pcolor(ncm,cmap='terrain_r')
    ax.set_frame_on(True)

    # put the major ticks at the middle of each cell
    ax.set_yticks(np.arange(confusion_matrix.shape[0])+0.5, minor=False)
    ax.set_xticks(np.arange(confusion_matrix.shape[1])+0.5, minor=False)

    # want a more natural, table-like display
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    if label_extras is not None:
        labels = [' {:s} {:s}'.format(x,label_extras[x].strip()) for x in labels]
    
    ax.set_xticklabels([], minor=False) 
    ax.set_yticklabels([], minor=False)

    if xlabel:
        ax.set_xticklabels(labels, minor=False, rotation=rotation, horizontalalignment='left') 
    if ylabel:
        ax.set_yticklabels(labels, minor=False)

    ax.grid(True)
    fig.colorbar(pax)
    fig.tight_layout()
    return fig


with open('CoT_experiments/data/rxnfp/rxnclass2id.json', 'r') as f:
    rxnclass2id = json.load(f)

with open('CoT_experiments/data/rxnfp/rxnclass2name.json', 'r') as f:
    rxnclass2name = json.load(f)
all_classes =sorted(rxnclass2id.keys())



import pickle
schneider_df = pd.read_csv('CoT_experiments/data/rxnfp/schneider50k.tsv', sep='\t', index_col=0)
ft_10k_fps = np.load('CoT_experiments/data/rxnfp/fps_ft_10k.npz')['fps']
ft_pretrained = np.load('CoT_experiments/data/rxnfp/fps_pretrained.npz')['fps']
schneider_df['ft_10k'] = [fp for fp in ft_10k_fps]
schneider_df['ft_pretrained'] = [fp for fp in ft_pretrained]
schneider_df['class_id'] = [rxnclass2id[c] for c in schneider_df.rxn_class]
schneider_df.head()

train_df = schneider_df[schneider_df.split=='train']
test_df = schneider_df[schneider_df.split=='test']
print(len(train_df), len(test_df))

lr_cls =  LogisticRegression(max_iter=1000)

scrambled_train_rxn_ids = [rxnclass2id[c] for c in train_df.rxn_class]
test_rxn_class_ids = [rxnclass2id[c] for c in test_df.rxn_class]


lr_cls =  LogisticRegression(max_iter=5000)
lr_classifier_ft_10k_trained = lr_cls.fit(train_df.ft_10k.values.tolist(), train_df.class_id.values.tolist())

bert_model, tokenizer = get_default_model_and_tokenizer()
rxnfp_generator = RXNBERTFingerprintGenerator(bert_model, tokenizer)
example_rxn = "Cl.[NH:2]1[CH2:5][CH2:4][CH2:3]1.[OH-].[Na+].Br[CH2:9][C:10]([O:12][C:13]([CH3:16])([CH3:15])[CH3:14])=[O:11].C(OCC)(=O)C>O1CCCC1.O>[N:2]1([CH2:9][C:10]([O:12][C:13]([CH3:16])([CH3:15])[CH3:14])=[O:11])[CH2:5][CH2:4][CH2:3]1"
removed_rxn = remove_atom_mapping(example_rxn)
fp = rxnfp_generator.convert(example_rxn)
removed_fp = rxnfp_generator.convert(removed_rxn)
preds = lr_classifier_ft_10k_trained.predict([fp, removed_fp])
predicted_classes = [rxnclass2name[all_classes[x]] for x in preds]

print(predicted_classes)


# confusion_matrix_ft_10k = evaluate_model(lr_classifier_ft_10k_trained, test_df.ft_10k.values.tolist(), test_rxn_class_ids, all_classes, rxnclass2name)
# fig = labelled_cmat(confusion_matrix_ft_10k, 
#                     all_classes,
#                     figsize=(16,12), label_extras=rxnclass2name)


# lr_cls =  LogisticRegression(max_iter=5000)
# lr_classifier_ft_pretrained = lr_cls.fit(train_df.ft_pretrained.values.tolist(), train_df.class_id.values.tolist())
# confusion_matrix_pretrained = evaluate_model(lr_classifier_ft_pretrained, test_df.ft_pretrained.values.tolist(), test_rxn_class_ids, all_classes, rxnclass2name)
# fig = labelled_cmat(confusion_matrix_pretrained, 
#                     all_classes,
#                     figsize=(16,12), label_extras=rxnclass2name)


# confusion_matrix_pretrained = evaluate_model(lr_classifier_ft_pretrained, test_df.ft_10k.values.tolist(), test_rxn_class_ids, all_classes, rxnclass2name)
# fig = labelled_cmat(confusion_matrix_pretrained, 
#                     all_classes,
#                     figsize=(16,12), label_extras=rxnclass2name)






bert_model, tokenizer = get_default_model_and_tokenizer()

rxnfp_generator = RXNBERTFingerprintGenerator(bert_model, tokenizer)

example_rxn = "Nc1cccc2cnccc12.O=C(O)c1cc([N+](=O)[O-])c(Sc2c(Cl)cncc2Cl)s1>>O=C(Nc1cccc2cnccc12)c1cc([N+](=O)[O-])c(Sc2c(Cl)cncc2Cl)s1"

fp = rxnfp_generator.convert(example_rxn)
# model.predict(fp)
# print(len(fp))
# print(fp[:5])