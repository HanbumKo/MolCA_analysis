import matplotlib.pyplot as plt
import matplotlib as mpl
import random
import json
import numpy as np


mpl.rcParams.update({
    'font.size': 20,        # 기본 글씨 크기
    'axes.labelsize': 25,   # x, y축 label 폰트 크기
    'axes.titlesize': 20,   # subplot title 폰트 크기
    'xtick.labelsize': 15,  # x축 tick 폰트 크기
    'ytick.labelsize': 20,  # y축 tick 폰트 크기
    'legend.fontsize': 20,  # 범례 폰트 크기
})


# Example task, reasoning, shot, seed, and metric lists
task_names = ["forward", "retro", "reagent", "catalyst", "solvent"]
# task_names = ["forward"]
# use_reasoning_texts = ["w/o reasoning instruction", "w/ reasoning instruction"]
# reasoning = ["no", "generated", "manual"]
methods = ["GPT-4o", "LlaSMol", "PRESTO", "ReactReasoner(Ours)"]
# n_shot_texts = ["0-shot", "1-shot", "2-shot", "3-shot", "4-shot", "5-shot", "6-shot", "7-shot"]
n_shot_texts = ["0-shot", "1-shot", "2-shot", "3-shot", "4-shot", "5-shot"]
# seed_texts = ["seed0", "seed1", "seed2", "seed3"]
metrics = ["exact_match", "bleu", "levenshtein", "rdk_sims", "maccs_sims", "morgan_sims", "validity"]


task_names_to_name = {
    "forward": "Forward",
    "retro": "Retrosynthesis",
    "reagent": "Reagent",
    "catalyst": "Catalyst",
    "solvent": "Solvent"
}

metric_to_name = {
    "exact_match": "EXACT↑",
    "bleu": "BLEU↑",
    "levenshtein": "LEVENSHTEIN↓",
    "rdk_sims": "RDK FTS↑",
    "maccs_sims": "MACCS FTS↑",
    "morgan_sims": "MORGAN FTS↑",
    "validity": "VALIDITY↑"
}


        # "ReactReasoner(Ours)": {
        #     "0-shot": {
        #         "exact_match": ,
        #         "bleu": ,
        #         "levenshtein": ,
        #         "rdk_sims": ,
        #         "maccs_sims": ,
        #         "morgan_sims": ,
        #         "validity": 
        #     },
        #     "1-shot": {
        #         "exact_match": ,
        #         "bleu": ,
        #         "levenshtein": ,
        #         "rdk_sims": ,
        #         "maccs_sims": ,
        #         "morgan_sims": ,
        #         "validity": 
        #     },
        #     "2-shot": {
        #         "exact_match": ,
        #         "bleu": ,
        #         "levenshtein": ,
        #         "rdk_sims": ,
        #         "maccs_sims": ,
        #         "morgan_sims": ,
        #         "validity": 
        #     },
        #     "3-shot": {
        #         "exact_match": ,
        #         "bleu": ,
        #         "levenshtein": ,
        #         "rdk_sims": ,
        #         "maccs_sims": ,
        #         "morgan_sims": ,
        #         "validity": 
        #     },
        #     "4-shot": {
        #         "exact_match": ,
        #         "bleu": ,
        #         "levenshtein": ,
        #         "rdk_sims": ,
        #         "maccs_sims": ,
        #         "morgan_sims": ,
        #         "validity": 
        #     },
        #     "5-shot": {
        #         "exact_match": ,
        #         "bleu": ,
        #         "levenshtein": ,
        #         "rdk_sims": ,
        #         "maccs_sims": ,
        #         "morgan_sims": ,
        #         "validity": 
        #     },
        # },

d = {
    "forward": {
        "GPT-4o": {
            "0-shot": {
                "exact_match": 0.09,
                "bleu": 0.679,
                "levenshtein": 9.08,
                "rdk_sims": 0.365,
                "maccs_sims": 0.542,
                "morgan_sims": 0.389,
                "validity": 0.87
            },
            "1-shot": {
                "exact_match": 0.15,
                "bleu": 0.664,
                "levenshtein": 9.85,
                "rdk_sims": 0.383,
                "maccs_sims": 0.564,
                "morgan_sims": 0.395,
                "validity": 0.89
            },
            "2-shot": {
                "exact_match": 0.13,
                "bleu": 0.735,
                "levenshtein": 7.24,
                "rdk_sims": 0.418,
                "maccs_sims": 0.551,
                "morgan_sims": 0.413,
                "validity": 0.8
            },
            "3-shot": {
                "exact_match": 0.15,
                "bleu": 0.725,
                "levenshtein": 7.53,
                "rdk_sims": 0.406,
                "maccs_sims": 0.581,
                "morgan_sims": 0.415,
                "validity": 0.87
            },
            "4-shot": {
                "exact_match": 0.15,
                "bleu": 0.74,
                "levenshtein": 7.15,
                "rdk_sims": 0.397,
                "maccs_sims": 0.557,
                "morgan_sims": 0.408,
                "validity": 0.82
            },
            "5-shot": {
                "exact_match": 0.15,
                "bleu": 0.74,
                "levenshtein": 7.2,
                "rdk_sims": 0.401,
                "maccs_sims": 0.571,
                "morgan_sims": 0.419,
                "validity": 0.83
            },
        },
        "LlaSMol": {
            "0-shot": {
                "exact_match": 0.15,
                "bleu": 0.661,
                "levenshtein": 10.89,
                "rdk_sims": 0.43,
                "maccs_sims": 0.631,
                "morgan_sims": 0.447,
                "validity": 0.98
            },
            "1-shot": {
                "exact_match": 0.08,
                "bleu": 0.383,
                "levenshtein": 27.15,
                "rdk_sims": 0.336,
                "maccs_sims": 0.544,
                "morgan_sims": 0.345,
                "validity": 0.98
            },
            "2-shot": {
                "exact_match": 0.03,
                "bleu": 0.335,
                "levenshtein": 30.8,
                "rdk_sims": 0.306,
                "maccs_sims": 0.492,
                "morgan_sims": 0.308,
                "validity": 0.97
            },
            "3-shot": {
                "exact_match": 0.03,
                "bleu": 0.334,
                "levenshtein": 21.74,
                "rdk_sims": 0.237,
                "maccs_sims": 0.357,
                "morgan_sims": 0.215,
                "validity": 0.63
            },
            "4-shot": {
                "exact_match": 0.03,
                "bleu": 0.185,
                "levenshtein": 44.14,
                "rdk_sims": 0.198,
                "maccs_sims": 0.312,
                "morgan_sims": 0.167,
                "validity": 0.61
            },
            "5-shot": {
                "exact_match": 0,
                "bleu": 0.169,
                "levenshtein": 54.31,
                "rdk_sims": 0.219,
                "maccs_sims": 0.316,
                "morgan_sims": 0.148,
                "validity": 0.69
            },
        },
        "PRESTO": {
            "0-shot": {
                "exact_match": 0.09,
                "bleu": 0.452,
                "levenshtein": 12.27,
                "rdk_sims": 0.29,
                "maccs_sims": 0.451,
                "morgan_sims": 0.337,
                "validity": 0.95
            },
            "1-shot": {
                "exact_match": 0.04,
                "bleu": 0.335,
                "levenshtein": 14.1,
                "rdk_sims": 0.214,
                "maccs_sims": 0.36,
                "morgan_sims": 0.25,
                "validity": 0.98
            },
            "2-shot": {
                "exact_match": 0.03,
                "bleu": 0.39,
                "levenshtein": 19.65,
                "rdk_sims": 0.199,
                "maccs_sims": 0.357,
                "morgan_sims": 0.215,
                "validity": 1
            },
            "3-shot": {
                "exact_match": 0.02,
                "bleu": 0.47,
                "levenshtein": 14.58,
                "rdk_sims": 0.228,
                "maccs_sims": 0.41,
                "morgan_sims": 0.276,
                "validity": 0.98
            },
            "4-shot": {
                "exact_match": 0.03,
                "bleu": 0.461,
                "levenshtein": 15.09,
                "rdk_sims": 0.229,
                "maccs_sims": 0.39,
                "morgan_sims": 0.253,
                "validity": 0.99
            },
            "5-shot": {
                "exact_match": 0.03,
                "bleu": 0.369,
                "levenshtein": 19.15,
                "rdk_sims": 0.208,
                "maccs_sims": 0.341,
                "morgan_sims": 0.209,
                "validity": 0.94
            },
        },
        "ReactReasoner(Ours)": {
            "0-shot": {
                "exact_match": 0.15,
                "bleu": 0.643,
                "levenshtein": 9.4,
                "rdk_sims": 0.373,
                "maccs_sims": 0.548,
                "morgan_sims": 0.404,
                "validity": 0.96
            },
            "1-shot": {
                "exact_match": 0.15,
                "bleu": 0.654,
                "levenshtein": 8.96,
                "rdk_sims": 0.382,
                "maccs_sims": 0.551,
                "morgan_sims": 0.408,
                "validity": 0.96
            },
            "2-shot": {
                "exact_match": 0.15,
                "bleu": 0.628,
                "levenshtein": 9.78,
                "rdk_sims": 0.377,
                "maccs_sims": 0.545,
                "morgan_sims": 0.399,
                "validity": 0.97
            },
            "3-shot": {
                "exact_match": 0.16,
                "bleu": 0.653,
                "levenshtein": 9.64,
                "rdk_sims": 0.391,
                "maccs_sims": 0.571,
                "morgan_sims": 0.423,
                "validity": 0.97
            },
            "4-shot": {
                "exact_match": 0.15,
                "bleu": 0.656,
                "levenshtein": 9.26,
                "rdk_sims": 0.385,
                "maccs_sims": 0.567,
                "morgan_sims": 0.422,
                "validity": 0.96
            },
            "5-shot": {
                "exact_match": 0.14,
                "bleu": 0.654,
                "levenshtein": 8.93,
                "rdk_sims": 0.38,
                "maccs_sims": 0.556,
                "morgan_sims": 0.419,
                "validity": 0.95
            },
        },
    },
    "retro": {
        "GPT-4o": {
            "0-shot": {
                "exact_match": 0.01,
                "bleu": 0.595,
                "levenshtein": 14.47,
                "rdk_sims": 0.292,
                "maccs_sims": 0.463,
                "morgan_sims": 0.299,
                "validity": 0.68
            },
            "1-shot": {
                "exact_match": 0,
                "bleu": 0.592,
                "levenshtein": 14.84,
                "rdk_sims": 0.285,
                "maccs_sims": 0.467,
                "morgan_sims": 0.306,
                "validity": 0.68
            },
            "2-shot": {
                "exact_match": 0.03,
                "bleu": 0.646,
                "levenshtein": 12.86,
                "rdk_sims": 0.309,
                "maccs_sims": 0.459,
                "morgan_sims": 0.332,
                "validity": 0.64
            },
            "3-shot": {
                "exact_match": 0.02,
                "bleu": 0.662,
                "levenshtein": 12.81,
                "rdk_sims": 0.301,
                "maccs_sims": 0.442,
                "morgan_sims": 0.313,
                "validity": 0.63
            },
            "4-shot": {
                "exact_match": 0.03,
                "bleu": 0.651,
                "levenshtein": 11.69,
                "rdk_sims": 0.268,
                "maccs_sims": 0.388,
                "morgan_sims": 0.271,
                "validity": 0.55
            },
            "5-shot": {
                "exact_match": 0.01,
                "bleu": 0.625,
                "levenshtein": 11.63,
                "rdk_sims": 0.229,
                "maccs_sims": 0.375,
                "morgan_sims": 0.248,
                "validity": 0.53
            },
        },
        "LlaSMol": {
            "0-shot": {
                "exact_match": 0.21,
                "bleu": 0.77,
                "levenshtein": 15.51,
                "rdk_sims": 0.696,
                "maccs_sims": 0.799,
                "morgan_sims": 0.649,
                "validity": 0.97
            },
            "1-shot": {
                "exact_match": 0.03,
                "bleu": 0.545,
                "levenshtein": 29.31,
                "rdk_sims": 0.532,
                "maccs_sims": 0.62,
                "morgan_sims": 0.452,
                "validity": 0.95
            },
            "2-shot": {
                "exact_match": 0.03,
                "bleu": 0.594,
                "levenshtein": 26.34,
                "rdk_sims": 0.578,
                "maccs_sims": 0.691,
                "morgan_sims": 0.514,
                "validity": 0.92
            },
            "3-shot": {
                "exact_match": 0,
                "bleu": 0.523,
                "levenshtein": 31.32,
                "rdk_sims": 0.551,
                "maccs_sims": 0.651,
                "morgan_sims": 0.483,
                "validity": 0.9
            },
            "4-shot": {
                "exact_match": 0,
                "bleu": 0.479,
                "levenshtein": 27.33,
                "rdk_sims": 0.401,
                "maccs_sims": 0.48,
                "morgan_sims": 0.347,
                "validity": 0.73
            },
            "5-shot": {
                "exact_match": 0,
                "bleu": 0.321,
                "levenshtein": 34.18,
                "rdk_sims": 0.27,
                "maccs_sims": 0.335,
                "morgan_sims": 0.22,
                "validity": 0.51
            },
        },
        "PRESTO": {
            "0-shot": {
                "exact_match": 0.01,
                "bleu": 0.405,
                "levenshtein": 17.55,
                "rdk_sims": 0.347,
                "maccs_sims": 0.426,
                "morgan_sims": 0.303,
                "validity": 0.76
            },
            "1-shot": {
                "exact_match": 0,
                "bleu": 0.383,
                "levenshtein": 26.22,
                "rdk_sims": 0.351,
                "maccs_sims": 0.48,
                "morgan_sims": 0.314,
                "validity": 1
            },
            "2-shot": {
                "exact_match": 0,
                "bleu": 0.465,
                "levenshtein": 23.69,
                "rdk_sims": 0.387,
                "maccs_sims": 0.527,
                "morgan_sims": 0.365,
                "validity": 0.98
            },
            "3-shot": {
                "exact_match": 0,
                "bleu": 0.315,
                "levenshtein": 27.04,
                "rdk_sims": 0.331,
                "maccs_sims": 0.479,
                "morgan_sims": 0.311,
                "validity": 0.98
            },
            "4-shot": {
                "exact_match": 0,
                "bleu": 0.481,
                "levenshtein": 24.6,
                "rdk_sims": 0.439,
                "maccs_sims": 0.557,
                "morgan_sims": 0.408,
                "validity": 0.99
            },
            "5-shot": {
                "exact_match": 0.02,
                "bleu": 0.317,
                "levenshtein": 26.12,
                "rdk_sims": 0.306,
                "maccs_sims": 0.433,
                "morgan_sims": 0.285,
                "validity": 0.97
            },
        },
        "ReactReasoner(Ours)": {
            "0-shot": {
                "exact_match": 0.27,
                "bleu": 0.776,
                "levenshtein": 15.08,
                "rdk_sims": 0.67,
                "maccs_sims": 0.798,
                "morgan_sims": 0.656,
                "validity": 0.98
            },
            "1-shot": {
                "exact_match": 0.22,
                "bleu": 0.767,
                "levenshtein": 15.25,
                "rdk_sims": 0.656,
                "maccs_sims": 0.785,
                "morgan_sims": 0.64,
                "validity": 0.98
            },
            "2-shot": {
                "exact_match": 0.25,
                "bleu": 0.776,
                "levenshtein": 14.72,
                "rdk_sims": 0.678,
                "maccs_sims": 0.796,
                "morgan_sims": 0.652,
                "validity": 0.97
            },
            "3-shot": {
                "exact_match": 0.22,
                "bleu": 0.778,
                "levenshtein": 15.48,
                "rdk_sims": 0.689,
                "maccs_sims": 0.804,
                "morgan_sims": 0.661,
                "validity": 0.99
            },
            "4-shot": {
                "exact_match": 0.24,
                "bleu": 0.764,
                "levenshtein": 16.32,
                "rdk_sims": 0.682,
                "maccs_sims": 0.805,
                "morgan_sims": 0.662,
                "validity": 1
            },
            "5-shot": {
                "exact_match": 0.26,
                "bleu": 0.77,
                "levenshtein": 15.13,
                "rdk_sims": 0.67,
                "maccs_sims": 0.802,
                "morgan_sims": 0.66,
                "validity": 0.99
            },
        },
    },
    "reagent": {
        "GPT-4o": {
            "0-shot": {
                "exact_match": 0.04,
                "bleu": 0.149,
                "levenshtein": 12.6,
                "rdk_sims": 0.215,
                "maccs_sims": 0.142,
                "morgan_sims": 0.07,
                "validity": 0.69
            },
            "1-shot": {
                "exact_match": 0,
                "bleu": 0.139,
                "levenshtein": 8.47,
                "rdk_sims": 0.21,
                "maccs_sims": 0.156,
                "morgan_sims": 0.079,
                "validity": 0.6
            },
            "2-shot": {
                "exact_match": 0.08,
                "bleu": 0.341,
                "levenshtein": 12.02,
                "rdk_sims": 0.371,
                "maccs_sims": 0.381,
                "morgan_sims": 0.247,
                "validity": 0.89
            },
            "3-shot": {
                "exact_match": 0.07,
                "bleu": 0.286,
                "levenshtein": 10.47,
                "rdk_sims": 0.338,
                "maccs_sims": 0.329,
                "morgan_sims": 0.201,
                "validity": 0.84
            },
            "4-shot": {
                "exact_match": 0.12,
                "bleu": 0.332,
                "levenshtein": 9.24,
                "rdk_sims": 0.38,
                "maccs_sims": 0.384,
                "morgan_sims": 0.253,
                "validity": 0.84
            },
            "5-shot": {
                "exact_match": 0.12,
                "bleu": 0.401,
                "levenshtein": 7.84,
                "rdk_sims": 0.356,
                "maccs_sims": 0.372,
                "morgan_sims": 0.254,
                "validity": 0.77
            },
        },
        "LlaSMol": {
            "0-shot": {
                "exact_match": 0,
                "bleu": 0.041,
                "levenshtein": 46.49,
                "rdk_sims": 0.018,
                "maccs_sims": 0.092,
                "morgan_sims": 0.023,
                "validity": 0.91
            },
            "1-shot": {
                "exact_match": 0,
                "bleu": 0.042,
                "levenshtein": 41.28,
                "rdk_sims": 0.088,
                "maccs_sims": 0.095,
                "morgan_sims": 0.028,
                "validity": 0.97
            },
            "2-shot": {
                "exact_match": 0,
                "bleu": 0.214,
                "levenshtein": 15.57,
                "rdk_sims": 0.293,
                "maccs_sims": 0.259,
                "morgan_sims": 0.102,
                "validity": 0.98
            },
            "3-shot": {
                "exact_match": 0,
                "bleu": 0.109,
                "levenshtein": 25.67,
                "rdk_sims": 0.169,
                "maccs_sims": 0.182,
                "morgan_sims": 0.076,
                "validity": 0.96
            },
            "4-shot": {
                "exact_match": 0,
                "bleu": 0.105,
                "levenshtein": 27.46,
                "rdk_sims": 0.215,
                "maccs_sims": 0.199,
                "morgan_sims": 0.08,
                "validity": 0.9
            },
            "5-shot": {
                "exact_match": 0,
                "bleu": 0.078,
                "levenshtein": 37.91,
                "rdk_sims": 0.17,
                "maccs_sims": 0.197,
                "morgan_sims": 0.086,
                "validity": 0.88
            },
        },
        "PRESTO": {
            "0-shot": {
                "exact_match": 0.07,
                "bleu": 0.27,
                "levenshtein": 13.58,
                "rdk_sims": 0.322,
                "maccs_sims": 0.315,
                "morgan_sims": 0.178,
                "validity": 1
            },
            "1-shot": {
                "exact_match": 0.05,
                "bleu": 0.268,
                "levenshtein": 13.66,
                "rdk_sims": 0.323,
                "maccs_sims": 0.288,
                "morgan_sims": 0.143,
                "validity": 0.99
            },
            "2-shot": {
                "exact_match": 0.05,
                "bleu": 0.232,
                "levenshtein": 14.78,
                "rdk_sims": 0.268,
                "maccs_sims": 0.291,
                "morgan_sims": 0.139,
                "validity": 0.99
            },
            "3-shot": {
                "exact_match": 0.04,
                "bleu": 0.268,
                "levenshtein": 14.37,
                "rdk_sims": 0.169,
                "maccs_sims": 0.264,
                "morgan_sims": 0.124,
                "validity": 1
            },
            "4-shot": {
                "exact_match": 0.05,
                "bleu": 0.217,
                "levenshtein": 15.27,
                "rdk_sims": 0.19,
                "maccs_sims": 0.245,
                "morgan_sims": 0.126,
                "validity": 0.99
            },
            "5-shot": {
                "exact_match": 0.06,
                "bleu": 0.277,
                "levenshtein": 14.77,
                "rdk_sims": 0.31,
                "maccs_sims": 0.333,
                "morgan_sims": 0.151,
                "validity": 1
            },
        },
        "ReactReasoner(Ours)": {
            "0-shot": {
                "exact_match": 0.23,
                "bleu": 0.531,
                "levenshtein": 10.04,
                "rdk_sims": 0.551,
                "maccs_sims": 0.526,
                "morgan_sims": 0.415,
                "validity": 1
            },
            "1-shot": {
                "exact_match": 0.28,
                "bleu": 0.579,
                "levenshtein": 8.68,
                "rdk_sims": 0.597,
                "maccs_sims": 0.568,
                "morgan_sims": 0.473,
                "validity": 1
            },
            "2-shot": {
                "exact_match": 0.29,
                "bleu": 0.573,
                "levenshtein": 9.29,
                "rdk_sims": 0.591,
                "maccs_sims": 0.57,
                "morgan_sims": 0.468,
                "validity": 1
            },
            "3-shot": {
                "exact_match": 0.28,
                "bleu": 0.608,
                "levenshtein": 8.59,
                "rdk_sims": 0.587,
                "maccs_sims": 0.577,
                "morgan_sims": 0.483,
                "validity": 1
            },
            "4-shot": {
                "exact_match": 0.28,
                "bleu": 0.56,
                "levenshtein": 9.47,
                "rdk_sims": 0.564,
                "maccs_sims": 0.554,
                "morgan_sims": 0.433,
                "validity": 1
            },
            "5-shot": {
                "exact_match": 0.29,
                "bleu": 0.576,
                "levenshtein": 9.48,
                "rdk_sims": 0.558,
                "maccs_sims": 0.542,
                "morgan_sims": 0.448,
                "validity": 1
            },
        },
    },
    "catalyst": {
        "GPT-4o": {
            "0-shot": {
                "exact_match": 0.32,
                "bleu": 0.248,
                "levenshtein": 6.48,
                "rdk_sims": 0.581,
                "maccs_sims": 0.583,
                "morgan_sims": 0.335,
                "validity": 0.89
            },
            "1-shot": {
                "exact_match": 0.29,
                "bleu": 0.158,
                "levenshtein": 9.64,
                "rdk_sims": 0.534,
                "maccs_sims": 0.56,
                "morgan_sims": 0.302,
                "validity": 0.87
            },
            "2-shot": {
                "exact_match": 0.32,
                "bleu": 0.243,
                "levenshtein": 7.18,
                "rdk_sims": 0.567,
                "maccs_sims": 0.579,
                "morgan_sims": 0.345,
                "validity": 0.92
            },
            "3-shot": {
                "exact_match": 0.36,
                "bleu": 0.42,
                "levenshtein": 3.55,
                "rdk_sims": 0.609,
                "maccs_sims": 0.595,
                "morgan_sims": 0.381,
                "validity": 0.83
            },
            "4-shot": {
                "exact_match": 0.38,
                "bleu": 0.384,
                "levenshtein": 3.96,
                "rdk_sims": 0.645,
                "maccs_sims": 0.653,
                "morgan_sims": 0.418,
                "validity": 0.89
            },
            "5-shot": {
                "exact_match": 0.4,
                "bleu": 0.361,
                "levenshtein": 3.79,
                "rdk_sims": 0.641,
                "maccs_sims": 0.649,
                "morgan_sims": 0.42,
                "validity": 0.88
            },
        },
        "LlaSMol": {
            "0-shot": {
                "exact_match": 0,
                "bleu": 0.026,
                "levenshtein": 42.31,
                "rdk_sims": 0.024,
                "maccs_sims": 0.059,
                "morgan_sims": 0.017,
                "validity": 0.93
            },
            "1-shot": {
                "exact_match": 0,
                "bleu": 0.032,
                "levenshtein": 39.6,
                "rdk_sims": 0.016,
                "maccs_sims": 0.083,
                "morgan_sims": 0.018,
                "validity": 0.99
            },
            "2-shot": {
                "exact_match": 0,
                "bleu": 0.031,
                "levenshtein": 20.45,
                "rdk_sims": 0.008,
                "maccs_sims": 0.035,
                "morgan_sims": 0.016,
                "validity": 0.88
            },
            "3-shot": {
                "exact_match": 0.07,
                "bleu": 0.077,
                "levenshtein": 7.94,
                "rdk_sims": 0.572,
                "maccs_sims": 0.559,
                "morgan_sims": 0.074,
                "validity": 0.96
            },
            "4-shot": {
                "exact_match": 0.03,
                "bleu": 0.035,
                "levenshtein": 18.39,
                "rdk_sims": 0.117,
                "maccs_sims": 0.131,
                "morgan_sims": 0.041,
                "validity": 0.69
            },
            "5-shot": {
                "exact_match": 0.17,
                "bleu": 0.091,
                "levenshtein": 10.52,
                "rdk_sims": 0.484,
                "maccs_sims": 0.463,
                "morgan_sims": 0.178,
                "validity": 0.95
            },
        },
        "PRESTO": {
            "0-shot": {
                "exact_match": 0.18,
                "bleu": 0.202,
                "levenshtein": 8.5,
                "rdk_sims": 0.184,
                "maccs_sims": 0.287,
                "morgan_sims": 0.186,
                "validity": 0.88
            },
            "1-shot": {
                "exact_match": 0.14,
                "bleu": 0.238,
                "levenshtein": 6.88,
                "rdk_sims": 0.153,
                "maccs_sims": 0.319,
                "morgan_sims": 0.143,
                "validity": 0.95
            },
            "2-shot": {
                "exact_match": 0.17,
                "bleu": 0.249,
                "levenshtein": 7.65,
                "rdk_sims": 0.193,
                "maccs_sims": 0.342,
                "morgan_sims": 0.176,
                "validity": 0.94
            },
            "3-shot": {
                "exact_match": 0.2,
                "bleu": 0.295,
                "levenshtein": 5.46,
                "rdk_sims": 0.309,
                "maccs_sims": 0.419,
                "morgan_sims": 0.218,
                "validity": 0.91
            },
            "4-shot": {
                "exact_match": 0.2,
                "bleu": 0.192,
                "levenshtein": 7.31,
                "rdk_sims": 0.294,
                "maccs_sims": 0.413,
                "morgan_sims": 0.205,
                "validity": 0.98
            },
            "5-shot": {
                "exact_match": 0.46,
                "bleu": 0.335,
                "levenshtein": 5.38,
                "rdk_sims": 0.685,
                "maccs_sims": 0.692,
                "morgan_sims": 0.467,
                "validity": 0.99
            },
        },
        "ReactReasoner(Ours)": {
            "0-shot": {
                "exact_match": 0.61,
                "bleu": 0.658,
                "levenshtein": 2.62,
                "rdk_sims": 0.861,
                "maccs_sims": 0.821,
                "morgan_sims": 0.613,
                "validity": 1
            },
            "1-shot": {
                "exact_match": 0.61,
                "bleu": 0.641,
                "levenshtein": 3.09,
                "rdk_sims": 0.852,
                "maccs_sims": 0.813,
                "morgan_sims": 0.615,
                "validity": 1
            },
            "2-shot": {
                "exact_match": 0.62,
                "bleu": 0.662,
                "levenshtein": 2.42,
                "rdk_sims": 0.858,
                "maccs_sims": 0.826,
                "morgan_sims": 0.631,
                "validity": 1
            },
            "3-shot": {
                "exact_match": 0.6,
                "bleu": 0.568,
                "levenshtein": 3.85,
                "rdk_sims": 0.861,
                "maccs_sims": 0.833,
                "morgan_sims": 0.605,
                "validity": 1
            },
            "4-shot": {
                "exact_match": 0.58,
                "bleu": 0.563,
                "levenshtein": 3.66,
                "rdk_sims": 0.851,
                "maccs_sims": 0.832,
                "morgan_sims": 0.585,
                "validity": 0.99
            },
            "5-shot": {
                "exact_match": 0.58,
                "bleu": 0.612,
                "levenshtein": 3.32,
                "rdk_sims": 0.825,
                "maccs_sims": 0.827,
                "morgan_sims": 0.59,
                "validity": 1
            },
        },
    },
    "solvent": {
        "GPT-4o": {
            "0-shot": {
                "exact_match": 0.12,
                "bleu": 0.166,
                "levenshtein": 3.86,
                "rdk_sims": 0.151,
                "maccs_sims": 0.218,
                "morgan_sims": 0.172,
                "validity": 0.8
            },
            "1-shot": {
                "exact_match": 0.12,
                "bleu": 0.243,
                "levenshtein": 4.61,
                "rdk_sims": 0.18,
                "maccs_sims": 0.264,
                "morgan_sims": 0.191,
                "validity": 0.99
            },
            "2-shot": {
                "exact_match": 0.14,
                "bleu": 0.244,
                "levenshtein": 4.55,
                "rdk_sims": 0.205,
                "maccs_sims": 0.283,
                "morgan_sims": 0.211,
                "validity": 1
            },
            "3-shot": {
                "exact_match": 0.2,
                "bleu": 0.296,
                "levenshtein": 4.31,
                "rdk_sims": 0.257,
                "maccs_sims": 0.333,
                "morgan_sims": 0.268,
                "validity": 0.99
            },
            "4-shot": {
                "exact_match": 0.17,
                "bleu": 0.263,
                "levenshtein": 4.5,
                "rdk_sims": 0.232,
                "maccs_sims": 0.3,
                "morgan_sims": 0.235,
                "validity": 1
            },
            "5-shot": {
                "exact_match": 0.15,
                "bleu": 0.3,
                "levenshtein": 4.34,
                "rdk_sims": 0.232,
                "maccs_sims": 0.31,
                "morgan_sims": 0.245,
                "validity": 1
            },
        },
        "LlaSMol": {
            "0-shot": {
                "exact_match": 0,
                "bleu": 0.034,
                "levenshtein": 42.99,
                "rdk_sims": 0.012,
                "maccs_sims": 0.106,
                "morgan_sims": 0.034,
                "validity": 0.98
            },
            "1-shot": {
                "exact_match": 0,
                "bleu": 0.018,
                "levenshtein": 17.49,
                "rdk_sims": 0.03,
                "maccs_sims": 0.058,
                "morgan_sims": 0.02,
                "validity": 0.52
            },
            "2-shot": {
                "exact_match": 0,
                "bleu": 0.024,
                "levenshtein": 9.67,
                "rdk_sims": 0.029,
                "maccs_sims": 0.108,
                "morgan_sims": 0.035,
                "validity": 0.86
            },
            "3-shot": {
                "exact_match": 0.01,
                "bleu": 0.024,
                "levenshtein": 20.26,
                "rdk_sims": 0.026,
                "maccs_sims": 0.091,
                "morgan_sims": 0.036,
                "validity": 0.82
            },
            "4-shot": {
                "exact_match": 0,
                "bleu": 0.044,
                "levenshtein": 7.43,
                "rdk_sims": 0.009,
                "maccs_sims": 0.031,
                "morgan_sims": 0.012,
                "validity": 0.26
            },
            "5-shot": {
                "exact_match": 0,
                "bleu": 0.042,
                "levenshtein": 37.52,
                "rdk_sims": 0.037,
                "maccs_sims": 0.13,
                "morgan_sims": 0.057,
                "validity": 0.94
            },
        },
        "PRESTO": {
            "0-shot": {
                "exact_match": 0.22,
                "bleu": 0.198,
                "levenshtein": 5.41,
                "rdk_sims": 0.278,
                "maccs_sims": 0.336,
                "morgan_sims": 0.268,
                "validity": 1
            },
            "1-shot": {
                "exact_match": 0.07,
                "bleu": 0.021,
                "levenshtein": 5.99,
                "rdk_sims": 0.083,
                "maccs_sims": 0.173,
                "morgan_sims": 0.088,
                "validity": 1
            },
            "2-shot": {
                "exact_match": 0.14,
                "bleu": 0.118,
                "levenshtein": 7.78,
                "rdk_sims": 0.163,
                "maccs_sims": 0.258,
                "morgan_sims": 0.176,
                "validity": 1
            },
            "3-shot": {
                "exact_match": 0.11,
                "bleu": 0.142,
                "levenshtein": 6.46,
                "rdk_sims": 0.207,
                "maccs_sims": 0.26,
                "morgan_sims": 0.18,
                "validity": 1
            },
            "4-shot": {
                "exact_match": 0.11,
                "bleu": 0.181,
                "levenshtein": 6.81,
                "rdk_sims": 0.193,
                "maccs_sims": 0.254,
                "morgan_sims": 0.167,
                "validity": 1
            },
            "5-shot": {
                "exact_match": 0.16,
                "bleu": 0.192,
                "levenshtein": 6.76,
                "rdk_sims": 0.237,
                "maccs_sims": 0.316,
                "morgan_sims": 0.232,
                "validity": 1
            },
        },
        "ReactReasoner(Ours)": {
            "0-shot": {
                "exact_match": 0.32,
                "bleu": 0.443,
                "levenshtein": 3.62,
                "rdk_sims": 0.459,
                "maccs_sims": 0.483,
                "morgan_sims": 0.432,
                "validity": 1
            },
            "1-shot": {
                "exact_match": 0.3,
                "bleu": 0.402,
                "levenshtein": 4.22,
                "rdk_sims": 0.409,
                "maccs_sims": 0.435,
                "morgan_sims": 0.388,
                "validity": 1
            },
            "2-shot": {
                "exact_match": 0.26,
                "bleu": 0.429,
                "levenshtein": 3.86,
                "rdk_sims": 0.42,
                "maccs_sims": 0.434,
                "morgan_sims": 0.381,
                "validity": 1
            },
            "3-shot": {
                "exact_match": 0.32,
                "bleu": 0.438,
                "levenshtein": 3.82,
                "rdk_sims": 0.458,
                "maccs_sims": 0.473,
                "morgan_sims": 0.423,
                "validity": 1
            },
            "4-shot": {
                "exact_match": 0.29,
                "bleu": 0.427,
                "levenshtein": 3.89,
                "rdk_sims": 0.424,
                "maccs_sims": 0.445,
                "morgan_sims": 0.4,
                "validity": 1
            },
            "5-shot": {
                "exact_match": 0.29,
                "bleu": 0.453,
                "levenshtein": 3.84,
                "rdk_sims": 0.449,
                "maccs_sims": 0.479,
                "morgan_sims": 0.426,
                "validity": 1
            },
        },
    },
}

# method 별로 라인을 구분하기 위해 색상, 마커 등을 지정
method_styles = {
    "GPT-4o": {
        "color": "#1f77b4", 
        "marker": "o", 
        "label": "GPT-4o"
    },
    "LlaSMol": {
        "color": "#ff7f0e", 
        "marker": "o", 
        "label": "LlaSMol"
    },
    "PRESTO": {
        "color": "#2ca02c",
        "marker": "o",
        "label": "PRESTO"
    },
    "ReactReasoner(Ours)": {
        "color": "#d62728",
        "marker": "o",
        "label": "ReactReasoner(Ours)"
    },
}



fig, axes = plt.subplots(nrows=5, ncols=7, figsize=(35/1.3, 25/1.3), sharey=False)
for method in methods:
    for i, task in enumerate(task_names):
        for j, metric in enumerate(metrics):
            ax = axes[i, j]
            
            # x축을 n_shot_list의 인덱스 값으로 사용하되, 레이블은 n_shot_list 자체
            x_vals = np.arange(len(n_shot_texts))
            
            # 각 method에 대해 라인으로 그림
            ax.plot(
                x_vals,
                [d[task][method][n_shot][metric] for n_shot in n_shot_texts],
                color=method_styles[method]["color"],
                marker=method_styles[method]["marker"],
                label=method_styles[method]["label"]
            )
            
            # x축 눈금과 레이블 설정
            ax.set_xticks(x_vals)
            ax.set_xticklabels(n_shot_texts, rotation=45, ha='right')
            
            # (변경 1) subplot title 제거
            # ax.set_title(f"{task} - {metric}")  # 제거
            
            # (변경 2) 맨 왼쪽 컬럼이면 row label(= task)
            if j == 0:
                ax.set_ylabel(task_names_to_name[task])
            
            # (변경 3) 맨 윗줄이면 column label(= metric)
            if i == 0:
                ax.set_title(metric_to_name[metric])

# Legend 정보를 수집
all_handles = []
all_labels = []
for row in axes:
    for ax in row:
        handles, labels = ax.get_legend_handles_labels()
        all_handles.extend(handles)
        all_labels.extend(labels)

# 중복 라벨 제거
by_label = {label: handle for label, handle in zip(all_labels, all_handles)}

# 범례 추가: 플롯 상단에 한 번만 표시
fig.legend(by_label.values(),
           by_label.keys(),
           loc='upper center',  # 플롯 상단 중앙에 배치
           ncol=4,  # 범례 항목을 한 줄에 네 개씩 정렬
           bbox_to_anchor=(0.5, 0.99),  # 플롯 위로 약간 띄워서 배치
        #    fontsize='medium',
           frameon=True)

plt.tight_layout(rect=[0, 0, 1, 0.97])  # 전체 플롯 레이아웃 조정
# plt.tight_layout()
plt.savefig(f"CoT_experiments/results/cot_prompt_test/plots/CoT_test_line.png", dpi=200)
plt.savefig(f"CoT_experiments/results/cot_prompt_test/plots/CoT_test_line.pdf", dpi=200)
plt.clf()
