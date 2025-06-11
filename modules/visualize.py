"""
Module: visualize.py
Functions to visualize Bayesian networks and learning progress.
"""
import matplotlib.pyplot as plt
import networkx as nx

def draw_model(model, title="Bayesian Network"):
    plt.figure(figsize=(10, 6))
    nx.draw(model.to_digraph(), with_labels=True, node_size=2000, node_color='lightblue')
    plt.title(title)
    plt.show()
