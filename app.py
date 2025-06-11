import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "watchdog"

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import io

from pgmpy.estimators import PC, HillClimbSearch, BicScore
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from modules.bma_and_parameter_learning import sample_structures

st.set_page_config(layout="wide")
st.title("📊 Bayesian Network Structure Learning Dashboard")
st.markdown("Based on Chapter 18 from *Probabilistic Graphical Models* by Koller & Friedman")

# Sidebar steps
st.sidebar.header("Workflow")

# Step 1: Upload Data
uploaded_file = st.sidebar.file_uploader("Step 1: Upload CSV Dataset", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview", data.head())

    # Step 2: Select learning method
    method = st.sidebar.radio("Step 2: Choose Structure Learning Method",
                              ["PC Algorithm", "Score-Based (BIC)"])

    # Step 3: Learn structure
    learn_button = st.sidebar.button("Step 3: Learn Structure")

    if learn_button:
        with st.spinner("Learning the Bayesian Network structure..."):
            try:
                if method == "PC Algorithm":
                    pc = PC(data)
                    model = pc.estimate()
                else:
                    hc = HillClimbSearch(data)
                    model = hc.estimate(scoring_method=BicScore(data))
                st.success("Structure learning completed!")
                st.session_state['model'] = model  # Store model in session state
            except Exception as e:
                st.error(f"Failed to learn structure: {e}")
    elif 'model' in st.session_state:
        model = st.session_state['model']
    else:
        model = None

    if model:
        st.subheader("🖼️ Learned Structure (DAG)")
        fig, ax = plt.subplots(figsize=(10, 6))
        #nx.draw(model.to_networkx_graph(), with_labels=True, node_color="lightblue", node_size=2000, ax=ax)
        G = nx.DiGraph()
        G.add_nodes_from(model.nodes())
        G.add_edges_from(model.edges())
        nx.draw(G, with_labels=True, node_color="lightblue", node_size=2000, ax=ax)

        st.pyplot(fig)

        # Download learned structure as PNG
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        st.download_button(label="Download DAG Image", data=buf, file_name="learned_structure.png", mime="image/png")

        # Step 4: Bayesian Model Averaging (optional)
        st.subheader("🔁 Bayesian Model Averaging (20 samples)")
        with st.spinner("Sampling structures for BMA..."):
            edge_probs, _ = sample_structures(data, num_samples=20)
        G = nx.Graph()
        for edge, prob in edge_probs.items():
            G.add_edge(*edge, weight=prob)
        pos = nx.spring_layout(G)
        weights = [d["weight"] * 5 for (_, _, d) in G.edges(data=True)]
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        nx.draw(G, pos, with_labels=True, width=weights, node_color="lightyellow", ax=ax2)
        edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        st.pyplot(fig2)

        # Step 5: Parameter Learning and Inference
        st.subheader("📐 Learn Parameters & Run Inference")

        with st.spinner("Fitting parameters..."):
            try:
                bayes_model = BayesianNetwork(model.edges())
                bayes_model.fit(data, estimator=MaximumLikelihoodEstimator)
                infer = VariableElimination(bayes_model)
            except Exception as e:
                st.error(f"Parameter learning failed: {e}")
                infer = None

        if infer:
            all_vars = list(data.columns)

            query_var = st.selectbox("Select Query Variable", all_vars)

            evidence_vars = [v for v in all_vars if v != query_var]
            evidence_var = st.multiselect("Select Evidence Variables (optional)", evidence_vars)

            evidence = {}
            for i, var in enumerate(evidence_var):
                val = st.selectbox(f"Value for {var}", data[var].unique(), key=f"evidence_{i}")
                evidence[var] = val

            run_inf = st.button("🔎 Run Inference")

            if run_inf:
                with st.spinner("Running inference..."):
                    try:
                        result = infer.query(variables=[query_var], evidence=evidence)
                        st.success("Inference completed!")

                        # Display nicely
                        st.write(f"### Posterior Distribution of {query_var}")
                        prob_df = pd.DataFrame({
                            f"{query_var}": result.state_names[query_var],
                            "Probability": result.values
                        })
                        st.dataframe(prob_df)

                        st.bar_chart(prob_df.set_index(query_var))

                    except Exception as e:
                        st.error(f"Inference failed: {e}")

else:
    st.info("👈 Please upload a CSV dataset to get started.")
st.markdown("---")
st.markdown("🔗 **Sample Datasets:** [Bayesian Network Repository](https://www.bnlearn.com/bnrepository/)")
