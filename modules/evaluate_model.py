"""
Module: evaluate_model.py
Evaluation functions: compare structure to ground truth, score models.
"""

def hamming_distance(model1, model2):
    """
    Computes the Hamming distance between two Bayesian networks
    by comparing their edge sets.
    """
    edges1 = set(model1.edges())
    edges2 = set(model2.edges())
    return len(edges1.symmetric_difference(edges2))

def compare_structures(learned_model, true_model):
    """
    Compares a learned Bayesian model to the ground truth using Hamming distance.
    """
    return hamming_distance(learned_model, true_model)
