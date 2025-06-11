"""
Streamlit Dashboard: Structure Learning in Bayesian Networks
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from pgmpy.estimators import PC, HillClimbSearch, BicScore
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from modules.bma_and_parameter_learning import sample_structures

st.set_page_config(layout="wide")
st.title("📊 Bayesian Network Structure Learning Dashboard")
st.markdown("Based on Chapter 18 from *Probabilistic Graphical Models* by Koller & Friedman")

# Upload data
st.sidebar.header("1. Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("### Preview of Uploaded Data", data.head())

    # Choose learning method
    st.sidebar.header("2. Choose Structure Learning Method")
    method = st.sidebar.radio("Learning Method", ["PC Algorithm", "Score-Based (BIC)"])

    if method == "PC Algorithm":
        st.subheader("🔍 Learning Structure with PC Algorithm")
        pc = PC(data)
        model = pc.estimate()
    else:
        st.subheader("📊 Learning Structure with Score-Based Hill Climbing (BIC)")
        hc = HillClimbSearch(data)
        model = hc.estimate(scoring_method=BicScore(data))

    # Visualize DAG
    st.subheader("🖼️ Learned Structure (DAG)")
    fig, ax = plt.subplots(figsize=(10, 6))
    nx.draw(model.to_digraph(), with_labels=True, node_color="lightblue", node_size=2000, ax=ax)
    st.pyplot(fig)

    # BMA Section
    st.subheader("🔁 Bayesian Model Averaging (20 samples)")
    edge_probs, _ = sample_structures(data, num_samples=20)
    G = nx.Graph()
    for edge, prob in edge_probs.items():
        G.add_edge(*edge, weight=prob)
    pos = nx.spring_layout(G)
    weights = [d["weight"] * 5 for (_, _, d) in G.edges(data=True)]
    fig, ax = plt.subplots(figsize=(10, 6))
    nx.draw(G, pos, with_labels=True, width=weights, node_color="lightyellow", ax=ax)
    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    st.pyplot(fig)

    # Parameter Learning & Inference
    st.subheader("📐 Learn Parameters & Run Inference")
    try:
        model.fit(data, estimator=MaximumLikelihoodEstimator)
        infer = VariableElimination(model)

        all_vars = list(data.columns)
        query_var = st.selectbox("Select Query Variable", all_vars)
        evidence_var = st.multiselect("Select Evidence Variables", [v for v in all_vars if v != query_var])
        evidence = {}
        for var in evidence_var:
            val = st.selectbox(f"Value for {var}", data[var].unique(), key=var)
            evidence[var] = val

        if st.button("🔎 Run Inference"):
            result = infer.query(variables=[query_var], evidence=evidence)
            st.write("### Inference Result")
            st.write(result)
            st.bar_chart(result.values)
    except Exception as e:
        st.warning(f"Inference failed: {e}")
else:
    st.info("👈 Upload a dataset to get started.")
