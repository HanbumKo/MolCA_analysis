from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rdkit import Chem, RDLogger
from rdkit.Chem import MACCSkeys
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdchiral.chiral import copy_chirality
from rdkit.Chem.AllChem import AssignStereochemistry
from rouge_score import rouge_scorer
from tqdm import tqdm
from sklearn import metrics
import numpy as np
import torch


def classification_evaluate(predictions, references):
    y_pred = predictions
    y_true = [1 if ref == 'Yes.' else 0 for ref in references]

    auroc = metrics.roc_auc_score(y_true, y_pred)
    auprc = metrics.average_precision_score(y_true, y_pred)

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
    prec, recall, _ = metrics.precision_recall_curve(y_true, y_pred)
    # different with DrugBAN (modified by qizhi)
    precision = tpr / (tpr + fpr + 0.00001)
    f1 = 2 * precision * tpr / (tpr + precision + 0.00001)
    thred_optim = thresholds[5:][np.argmax(f1[5:])]
    y_pred_s = [1 if i else 0 for i in (y_pred >= thred_optim)]
    cm1 = metrics.confusion_matrix(y_true, y_pred_s)
    accuracy = (cm1[0, 0] + cm1[1, 1]) / sum(sum(cm1))
    sensitivity = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
    specificity = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
    precision1 = metrics.precision_score(y_true, y_pred_s)

    # with open(tsv_path, "w", encoding="utf-8") as f:
    #     f.write("ground truth\toutput\n")
    #     for gt, out in zip(y_true, y_pred):
    #         f.write(str(gt) + "\t" + str(out) + "\n")

    return {
        "accuracy": accuracy,
        "auroc": auroc,
        "auprc": auprc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": np.max(f1[5:]),
        "thred_optim": thred_optim,
        "precision": precision1,
    }


def canonicalize(smiles, isomeric=False, canonical=True, kekulize=False):
    # When canonicalizing a SMILES string, we typically want to
    # run Chem.RemoveHs(mol), but this will try to kekulize the mol
    # which is not required for canonical SMILES.  Instead, we make a
    # copy of the mol retaining only the information we desire (not explicit Hs)
    # Then, we sanitize the mol without kekulization.  copy_atom and copy_edit_mol
    # Are used to create this clean copy of the mol.
    def copy_atom(atom):
        new_atom = Chem.Atom(atom.GetSymbol())
        new_atom.SetFormalCharge(atom.GetFormalCharge())
        if atom.GetIsAromatic() and atom.GetNoImplicit():
            new_atom.SetNumExplicitHs(atom.GetNumExplicitHs())
            #elif atom.GetSymbol() == 'N':
            #    print(atom.GetSymbol())
            #    print(atom.GetImplicitValence())
            #    new_atom.SetNumExplicitHs(-atom.GetImplicitValence())
            #elif atom.GetSymbol() == 'S':
            #    print(atom.GetSymbol())
            #    print(atom.GetImplicitValence())
        return new_atom

    def copy_edit_mol(mol):
        new_mol = Chem.RWMol(Chem.MolFromSmiles(''))
        for atom in mol.GetAtoms():
            new_atom = copy_atom(atom)
            new_mol.AddAtom(new_atom)
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom().GetIdx()
            a2 = bond.GetEndAtom().GetIdx()
            bt = bond.GetBondType()
            new_mol.AddBond(a1, a2, bt)
            new_bond = new_mol.GetBondBetweenAtoms(a1, a2)
            new_bond.SetBondDir(bond.GetBondDir())
            new_bond.SetStereo(bond.GetStereo())
        for new_atom in new_mol.GetAtoms():
            atom = mol.GetAtomWithIdx(new_atom.GetIdx())
            copy_chirality(atom, new_atom)
        return new_mol

    smiles = smiles.replace(" ", "")  
    tmp = Chem.MolFromSmiles(smiles, sanitize=False)
    tmp.UpdatePropertyCache()
    new_mol = copy_edit_mol(tmp)
    #Chem.SanitizeMol(new_mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL)
    if not kekulize:
        Chem.SanitizeMol(new_mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_SETAROMATICITY | Chem.SanitizeFlags.SANITIZE_PROPERTIES | Chem.SanitizeFlags.SANITIZE_ADJUSTHS, catchErrors=True)
    else:
        Chem.SanitizeMol(new_mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE | Chem.SanitizeFlags.SANITIZE_PROPERTIES | Chem.SanitizeFlags.SANITIZE_ADJUSTHS, catchErrors=True)
   
    AssignStereochemistry(new_mol, cleanIt=False, force=True, flagPossibleStereoCenters=True)
    
    new_smiles = Chem.MolToSmiles(new_mol, isomericSmiles=isomeric, canonical=canonical)
    return new_smiles


def canonicalize_molecule_smiles(smiles, return_none_for_error=True, skip_mol=False, sort_things=True, isomeric=True, kekulization=True, allow_empty_part=False):
    things = smiles.split('.')
    if skip_mol:
        new_things = things
    else:
        new_things = []
        for thing in things:
            try:
                if thing == '' and not allow_empty_part:
                    raise ValueError('SMILES contains empty part.')

                mol = Chem.MolFromSmiles(thing)
                assert mol is not None
                for atom in mol.GetAtoms():
                    atom.SetAtomMapNum(0)
                thing_smiles = Chem.MolToSmiles(mol, kekuleSmiles=False, isomericSmiles=isomeric)
                thing_smiles = Chem.MolFromSmiles(thing_smiles)
                thing_smiles = Chem.MolToSmiles(thing_smiles, kekuleSmiles=False, isomericSmiles=isomeric)
                thing_smiles = Chem.MolFromSmiles(thing_smiles)
                thing_smiles = Chem.MolToSmiles(thing_smiles, kekuleSmiles=False, isomericSmiles=isomeric)
                assert thing_smiles is not None
                can_in = thing_smiles
                can_out = canonicalize(thing_smiles, isomeric=isomeric)
                assert can_out is not None, can_in
                thing_smiles = can_out
                if kekulization:
                    thing_smiles = keku_mid = Chem.MolFromSmiles(thing_smiles)
                    assert keku_mid is not None, 'Before can: %s\nAfter can: %s' % (can_in, can_out)
                    thing_smiles = Chem.MolToSmiles(thing_smiles, kekuleSmiles=True, isomericSmiles=isomeric)
            except KeyboardInterrupt:
                raise
            except:
                if return_none_for_error:
                    return None
                else:
                    raise
            new_things.append(thing_smiles)
    if sort_things:
        new_things = sorted(new_things)
    new_things = '.'.join(new_things)
    return new_things


def get_molecule_id(smiles, remove_duplicate=True):
    if remove_duplicate:
        assert ';' not in smiles
        all_inchi = set()
        for part in smiles.split('.'):
            inchi = get_molecule_id(part, remove_duplicate=False)
            all_inchi.add(inchi)
        all_inchi = tuple(sorted(all_inchi))
        return all_inchi
    else:
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToInchi(mol)


def judge_exact_match(pred_can_smiles_list, gold_can_smiles_list):
    assert len(pred_can_smiles_list) == len(gold_can_smiles_list)
    exact_match_labels = []
    for pred_smiles, gold_smiles_list in zip(pred_can_smiles_list, gold_can_smiles_list):
        if pred_smiles is None:
            exact_match_labels.append(False)
            continue
        pred_smiles_inchi = get_molecule_id(pred_smiles)
        sample_exact_match = False
        for gold_smiles in gold_smiles_list:
            assert gold_smiles is not None
            gold_smiles_inchi = get_molecule_id(gold_smiles)
            if pred_smiles_inchi == gold_smiles_inchi:
                sample_exact_match = True
                break
        exact_match_labels.append(sample_exact_match)
    return np.array(exact_match_labels)


def calculate_fingerprint_similarity(pred_mol_list, gold_mols_list, morgan_r=2):
    assert len(pred_mol_list) == len(gold_mols_list)
    MACCS_sims = []
    morgan_sims = []
    RDK_sims = []
    for pred_mol, gold_mol_list in zip(pred_mol_list, gold_mols_list):
        if pred_mol is None or type(pred_mol) == str:
            raise ValueError(type(pred_mol))
        tmp_MACCS, tmp_RDK, tmp_morgan = 0, 0, 0
        for gold_mol in gold_mol_list:
            tmp_MACCS = max(tmp_MACCS, DataStructs.FingerprintSimilarity(MACCSkeys.GenMACCSKeys(gold_mol), MACCSkeys.GenMACCSKeys(pred_mol), metric=DataStructs.TanimotoSimilarity))
            tmp_RDK = max(tmp_RDK, DataStructs.FingerprintSimilarity(Chem.RDKFingerprint(gold_mol), Chem.RDKFingerprint(pred_mol), metric=DataStructs.TanimotoSimilarity))
            tmp_morgan = max(tmp_morgan, DataStructs.TanimotoSimilarity(AllChem.GetMorganFingerprint(gold_mol,morgan_r), AllChem.GetMorganFingerprint(pred_mol, morgan_r)))
        MACCS_sims.append(tmp_MACCS)
        RDK_sims.append(tmp_RDK)
        morgan_sims.append(tmp_morgan)
    maccs_sims_score = np.mean(MACCS_sims)
    rdk_sims_score = np.mean(RDK_sims)
    morgan_sims_score = np.mean(morgan_sims)
    return maccs_sims_score, rdk_sims_score, morgan_sims_score


def judge_multiple_match(pred_can_smiles_list, golds_can_smiles_list):
    assert len(pred_can_smiles_list) == len(golds_can_smiles_list)
    subset_labels = []
    intersection_labels = []
    for pred_smiles, gold_smiles_list in zip(pred_can_smiles_list, golds_can_smiles_list):
        if pred_smiles is None:
            subset_labels.append(False)
            intersection_labels.append(False)
            continue

        pred_ele_set = set()
        for smiles in pred_smiles.split('.'):
            pred_ele_set.add(get_molecule_id(smiles, remove_duplicate=False))

        intersection_label = False
        subset_label = False
        for gold_smiles in gold_smiles_list:
            assert gold_smiles is not None
            gold_ele_set = set()
            for smiles in gold_smiles.split('.'):
                gold_ele_set.add(get_molecule_id(smiles, remove_duplicate=False))

            if len(pred_ele_set & gold_ele_set) > 0:
                intersection_label = True
                g_p = gold_ele_set - pred_ele_set
                if len(g_p) >= 0 and len(pred_ele_set - gold_ele_set) == 0:
                    subset_label = True
                    break
        intersection_labels.append(intersection_label)
        subset_labels.append(subset_label)
    
    return intersection_labels, subset_labels


def calculate_smiles_metrics(
        preds_smiles_list, 
        golds_smiles_list,
        metrics=('exact_match', 'fingerprint')
):
    preds_smiles_list = [[pred_smiles] for pred_smiles in preds_smiles_list]
    golds_smiles_list = [[gold_smiles.replace("SPL1T-TH1S-Pl3A5E", "").replace("\n", "").replace("[START_I_SMILES]", "").replace("[END_I_SMILES]", "")] for gold_smiles in golds_smiles_list]
    num_all = len(preds_smiles_list)
    assert num_all > 0
    assert num_all == len(golds_smiles_list)
    k = len(preds_smiles_list[0])

    dk_pred_smiles_list_dict = {}
    dk_pred_no_answer_labels_dict = {}
    dk_pred_invalid_labels_dict = {}
    for dk in range(k):
        dk_pred_smiles_list_dict[dk] = []
        dk_pred_no_answer_labels_dict[dk] = []
        dk_pred_invalid_labels_dict[dk] = []
    for pred_smiles_list in tqdm(preds_smiles_list):
        if pred_smiles_list is None:
            for dk in range(k):
                dk_pred_no_answer_labels_dict[dk].append(True)
                dk_pred_invalid_labels_dict[dk].append(False)
                dk_pred_smiles_list_dict[dk].append(None)
            continue
        assert len(pred_smiles_list) == k
        for dk, item in enumerate(pred_smiles_list):
            # item = item.strip()
            if item == '' or item is None:
                item = None
                dk_pred_no_answer_labels_dict[dk].append(True)
                dk_pred_invalid_labels_dict[dk].append(False)
            else:
                dk_pred_no_answer_labels_dict[dk].append(False)
                item = canonicalize_molecule_smiles(item)
                if item is None:
                    dk_pred_invalid_labels_dict[dk].append(True)
                else:
                    dk_pred_invalid_labels_dict[dk].append(False)
            dk_pred_smiles_list_dict[dk].append(item)
    
    new_list = []
    for gold_smiles_list in tqdm(golds_smiles_list):
        sample_gold_smiles_list = []
        for gold in gold_smiles_list:
            item = gold.strip()
            new_item = canonicalize_molecule_smiles(item, return_none_for_error=False)
            # if new_item is None:
            #     new_item = item  #TODO
            # assert new_item is not None, item
            sample_gold_smiles_list.append(new_item)
        new_list.append(sample_gold_smiles_list)
    golds_smiles_list = new_list

    metric_results = {'num_all': num_all}

    tk_pred_no_answer_labels = np.array([True] * num_all)
    tk_pred_invalid_labels = np.array([True] * num_all)
    for dk in range(k):
        dk_no_answer_labels = dk_pred_no_answer_labels_dict[dk]
        dk_invalid_labels = dk_pred_invalid_labels_dict[dk]
        tk_pred_no_answer_labels = tk_pred_no_answer_labels & dk_no_answer_labels
        tk_pred_invalid_labels = tk_pred_invalid_labels & dk_invalid_labels
        metric_results['num_t%d_no_answer' % (dk + 1)] = tk_pred_no_answer_labels.sum().item()
        metric_results['num_t%d_invalid' % (dk + 1)] = tk_pred_invalid_labels.sum().item()
    
    # d1_no_answer_labels = dk_pred_no_answer_labels_dict[0]
    # # print(np.array(d1_no_answer_labels).sum().item())
    # for label, item in zip(d1_no_answer_labels, preds_smiles_list):
    #     if label:
    #         print(item)

    for metric in metrics:
        if metric == 'exact_match':
            tk_exact_match_labels = np.array([False] * num_all)
            for dk in range(k):
                dk_pred_smiles_list = dk_pred_smiles_list_dict[dk]
                dk_exact_match_labels = judge_exact_match(dk_pred_smiles_list, golds_smiles_list)
                tk_exact_match_labels = tk_exact_match_labels | dk_exact_match_labels
                metric_results['num_t%d_exact_match' % (dk + 1)] = tk_exact_match_labels.sum().item()
        elif metric == 'fingerprint':
            d1_pred_mol_list = []
            gold_mols_list = []
            for pred_smiles, gold_smiles_list, no_answer, invalid in zip(dk_pred_smiles_list_dict[0], golds_smiles_list, dk_pred_no_answer_labels_dict[0], dk_pred_invalid_labels_dict[0]):
                if pred_smiles is None or pred_smiles.strip() == '' or no_answer is True or invalid is True:
                    continue
                pred_mol = Chem.MolFromSmiles(pred_smiles)
                # if pred_mol is None:  # TODO
                #     continue
                assert pred_mol is not None, pred_smiles
                gold_mol_list = []
                for gold_smiles in gold_smiles_list:
                    gold_mol = Chem.MolFromSmiles(gold_smiles)
                    # if gold_mol is None:
                    #     continue  # TODO
                    assert gold_mol is not None, gold_smiles
                    gold_mol_list.append(gold_mol)
                # if len(gold_mol_list) == 0:
                #     continue  # TODO
                d1_pred_mol_list.append(pred_mol)
                gold_mols_list.append(gold_mol_list)
            maccs_sims_score, rdk_sims_score, morgan_sims_score = calculate_fingerprint_similarity(d1_pred_mol_list, gold_mols_list)
            metric_results['t1_maccs_fps'] = maccs_sims_score
            metric_results['t1_rdk_fps'] = rdk_sims_score
            metric_results['t1_morgan_fps'] = morgan_sims_score
        elif metric == 'multiple_match':
            tk_intersection_labels = np.array([False] * num_all)
            tk_subset_labels = np.array([False] * num_all)
            for dk in range(k):
                dk_intersection_labels, dk_subset_labels = judge_multiple_match(dk_pred_smiles_list_dict[dk], golds_smiles_list)
                tk_intersection_labels = tk_intersection_labels | dk_intersection_labels
                tk_subset_labels = tk_subset_labels | dk_subset_labels
                metric_results['num_t%d_subset' % (dk + 1)] = tk_intersection_labels.sum().item()
                metric_results['num_t%d_intersection' % (dk + 1)] = tk_intersection_labels.sum().item()
        else:
            raise ValueError(metric)
    
    return metric_results


def caption_evaluate(predictions, targets, tokenizer, text_trunc_length):
    meteor_scores = []
    references = []
    hypotheses = []
    for gt, out in tqdm(zip(targets, predictions)):
        gt_tokens = tokenizer.tokenize(gt, truncation=True, max_length=text_trunc_length,
                                            padding='max_length')
        gt_tokens = list(filter(('[PAD]').__ne__, gt_tokens))
        gt_tokens = list(filter(('[CLS]').__ne__, gt_tokens))
        gt_tokens = list(filter(('[SEP]').__ne__, gt_tokens))

        out_tokens = tokenizer.tokenize(out, truncation=True, max_length=text_trunc_length,
                                            padding='max_length')
        out_tokens = list(filter(('[PAD]').__ne__, out_tokens))
        out_tokens = list(filter(('[CLS]').__ne__, out_tokens))
        out_tokens = list(filter(('[SEP]').__ne__, out_tokens))

        references.append([gt_tokens])
        hypotheses.append(out_tokens)

        mscore = meteor_score([gt_tokens], out_tokens)
        meteor_scores.append(mscore)

    bleu2 = corpus_bleu(references, hypotheses, weights=(.5,.5))
    bleu4 = corpus_bleu(references, hypotheses, weights=(.25,.25,.25,.25))
    bleu2 *= 100
    bleu4 *= 100

    print('BLEU-2 score:', bleu2)
    print('BLEU-4 score:', bleu4)
    _meteor_score = np.mean(meteor_scores)
    _meteor_score *= 100
    print('Average Meteor score:', _meteor_score)

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])

    rouge_scores = []

    references = []
    hypotheses = []

    for gt, out in tqdm(zip(targets, predictions)):
        rs = scorer.score(out, gt)
        rouge_scores.append(rs)

    print('ROUGE score:')
    rouge_1 = np.mean([rs['rouge1'].fmeasure for rs in rouge_scores]) * 100
    rouge_2 = np.mean([rs['rouge2'].fmeasure for rs in rouge_scores]) * 100
    rouge_l = np.mean([rs['rougeL'].fmeasure for rs in rouge_scores]) * 100
    print('rouge1:', rouge_1)
    print('rouge2:', rouge_2)
    print('rougeL:', rouge_l)
    return bleu2, bleu4, rouge_1, rouge_2, rouge_l, _meteor_score


def regression_evaluate(predictions, targets):
    validity = []
    predictions_processed = []
    targets_processed = []
    for p, t in zip(predictions, targets):
        # Convert to float
        try:
            p = float(p)
        except:
            validity.append(False)
            continue
        try:
            t = float(t.replace("SPL1T-TH1S-Pl3A5E", "").replace("\n", ""))
        except:
            validity.append(False)
            continue
        predictions_processed.append(p)
        targets_processed.append(t)
        validity.append(True)
    validity = np.array(validity)
    validity = np.mean(validity)
    if validity == 0:
        return None, None, None, None
    predictions = np.array(predictions_processed)
    targets = np.array(targets_processed)
    mae = np.mean(np.abs(predictions - targets))
    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)
    # mape = np.mean(np.abs(predictions - targets) / targets)
    return mae, mse, rmse, validity


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def pad_and_concat(tensor_list, fill_value=0):
    '''
    concat the first dimension and pad the second dimension
    tensor_list: [[B (diff), N_num, *], ...]
    '''
    device = tensor_list[0].device
    dtype=tensor_list[0].dtype
    max_dim1 = max(t.shape[1] for t in tensor_list)
    sum_dim0 = sum(t.shape[0] for t in tensor_list)
    if len(tensor_list[0].shape) == 3:
        out = torch.full((sum_dim0, max_dim1, tensor_list[0].shape[-1]), fill_value=fill_value, device=device, dtype=dtype)
        i = 0
        for t in tensor_list:
            out[i:i+t.shape[0], :t.shape[1]] = t
            i += t.shape[0]
        return out
    elif len(tensor_list[0].shape) == 2:
        out = torch.full((sum_dim0, max_dim1), fill_value=fill_value, device=device, dtype=dtype)
        i = 0
        for t in tensor_list:
            out[i:i+t.shape[0], :t.shape[1]] = t
            i += t.shape[0]
        return out
    raise NotImplementedError()