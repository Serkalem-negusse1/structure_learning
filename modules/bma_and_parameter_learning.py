"""
Module: bma_and_parameter_learning.py
Includes Bayesian Model Averaging approximation and parameter learning.
"""

import pandas as pd
import random
from pgmpy.estimators import HillClimbSearch, BicScore, MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from collections import defaultdict
import networkx as nx

def sample_structures(data, num_samples=20, random_state=42):
    """
    Perform greedy hill climbing multiple times with different seeds
    to simulate Bayesian Model Averaging over network structures.
    """
    random.seed(random_state)
    edge_counts = defaultdict(int)
    all_models = []

    for i in range(num_samples):
        seed = random.randint(0, 10000)
        #hc = HillClimbSearch(data, scoring_method=BicScore(data))
        hc = HillClimbSearch(data)
        model = hc.estimate(scoring_method=BicScore(data))

        model = hc.estimate()
        all_models.append(model)
        for edge in model.edges():
            edge_counts[tuple(sorted(edge))] += 1

    # Convert to probabilities
    edge_probs = {edge: count / num_samples for edge, count in edge_counts.items()}
    return edge_probs, all_models

def learn_parameters(model, data):
    """
    Use Maximum Likelihood Estimation to learn CPDs for a given structure.
    """
    model.fit(data, estimator=MaximumLikelihoodEstimator)
    return model

def perform_inference(model, query, evidence):
    """
    Use Variable Elimination for exact inference.
    """
    infer = VariableElimination(model)
    return infer.query(variables=query, evidence=evidence)
