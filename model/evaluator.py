import numpy as np
import nltk
import selfies as sf

from abc import ABC, abstractmethod
from typing import List
from Levenshtein import distance as lev
from nltk.translate.bleu_score import corpus_bleu
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import MACCSkeys, AllChem

RDLogger.DisableLog('rdApp.*')
nltk.download('wordnet')


def exact_match(ot_smi, gt_smi):
    m_out = Chem.MolFromSmiles(ot_smi)
    m_gt = Chem.MolFromSmiles(gt_smi)

    try:
        if Chem.MolToInchi(m_out) == Chem.MolToInchi(m_gt):
            return 1
    except:
        pass
    return 0


def maccs_similarity(ot_m, gt_m):
    return DataStructs.FingerprintSimilarity(
        MACCSkeys.GenMACCSKeys(gt_m), 
        MACCSkeys.GenMACCSKeys(ot_m), 
        metric=DataStructs.TanimotoSimilarity
    )

def morgan_similarity(ot_m, gt_m, radius=2):
    return DataStructs.TanimotoSimilarity(
        AllChem.GetMorganFingerprint(gt_m, radius), 
        AllChem.GetMorganFingerprint(ot_m, radius)
    )

def rdk_similarity(ot_m, gt_m):
    return DataStructs.FingerprintSimilarity(
        Chem.RDKFingerprint(gt_m), 
        Chem.RDKFingerprint(ot_m), 
        metric=DataStructs.TanimotoSimilarity
    )


class Evaluator(ABC):

    @abstractmethod
    def build_evaluate_tuple(self, pred, gt):
        pass

    @abstractmethod
    def evaluate(self, predictions, references, metrics: List[str] = None, verbose: bool = False, full_results: bool = False):
        pass


class MoleculeSMILESEvaluator(Evaluator):
    _metric_functions = {
        "exact_match": exact_match,
        "bleu": corpus_bleu,
        "levenshtein": lev,
        "rdk_sims": rdk_similarity,
        "maccs_sims": maccs_similarity,
        "morgan_sims": morgan_similarity,
        "validity": lambda smiles: smiles is not None,
    }

    @staticmethod
    def sf_decode(selfies):
        try:
            smiles = sf.decoder(selfies)
            return smiles
        except Exception:
            return None

    @staticmethod
    def convert_to_canonical_smiles(smiles):
        if not smiles:
            return None
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is not None:
            canonical_smiles = Chem.MolToSmiles(molecule, isomericSmiles=False, canonical=True)
            return canonical_smiles
        else:
            return None

    def build_evaluate_tuple(self, pred, gt, decode_selfies=False):
        if decode_selfies:
            pred = self.sf_decode(pred)
            gt = self.sf_decode(gt)
        return self.convert_to_canonical_smiles(pred), self.convert_to_canonical_smiles(gt)

    def evaluate(self, predictions, references, metrics: List[str] = None, verbose: bool = False, decode_selfies: bool = False, full_results: bool = False):
            
        if metrics is None:
            metrics = ["exact_match", "bleu", "levenshtein", "rdk_sims", "maccs_sims", "morgan_sims", "validity"]

        results = {metric: [] for metric in metrics}
        if "bleu" in metrics:
            results["bleu"] = [[], []]

        for pred, gt in zip(predictions, references):
            pred, gt = self.build_evaluate_tuple(pred, gt, decode_selfies=decode_selfies)

            for metric in metrics:
                if metric == "bleu" and pred and gt:
                    gt_tokens = [c for c in gt]
                    pred_tokens = [c for c in pred]
                    results[metric][0].append([gt_tokens])
                    results[metric][1].append(pred_tokens)
                elif pred is None or gt is None:
                    results[metric].append(0)
                    continue
                elif metric == "validity":
                    results[metric].append(self._metric_functions[metric](pred))
                elif metric in ["maccs_sims", "morgan_sims", "rdk_sims"]:
                    results[metric].append(self._metric_functions[metric](Chem.MolFromSmiles(pred), Chem.MolFromSmiles(gt)))
                else:
                    results[metric].append(self._metric_functions[metric](pred, gt))

        if "bleu" in metrics:
            if results["bleu"][0] and results["bleu"][1]:
                results["bleu"] = corpus_bleu(results["bleu"][0], results["bleu"][1])
            else:
                results["bleu"] = 0

        if verbose:
            print("Evaluation results:")
            for metric, values in results.items():
                print(f"{metric}: {np.mean(values)}")

        if full_results:
            return results
        else:
            return {metric: np.mean(values) for metric, values in results.items()}




