"""
Module: structure_learning.py
Implements structure learning methods: PC algorithm, score-based search.
""" 
import pandas as pd
from pgmpy.estimators import PC, HillClimbSearch, BicScore, BDeuScore


def learn_structure_pc(data: pd.DataFrame):
    pc = PC(data)
    return pc.estimate()

def learn_structure_score_based(data: pd.DataFrame, score_type="bic"):
    if score_type == "bic":
        score = BicScore(data)
    elif score_type == "bde":
        score = BDeuScore(data)
    else:
        raise ValueError("Invalid score type")
    hc = HillClimbSearch(data)
    return hc.estimate(scoring_method=score)
