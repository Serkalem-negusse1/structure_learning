"""
Module: evaluate_model.py
Evaluation functions: compare structure to ground truth, score models.
"""
from pgmpy.metrics import hamming_distance

def compare_structures(learned_model, true_model):
    return hamming_distance(learned_model, true_model)
