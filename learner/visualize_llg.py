# visualize_llg.py
import matplotlib
matplotlib.use("TkAgg")
import torch
import networkx as nx
import matplotlib.pyplot as plt
from representation.llg import LiftedLearningGraph, LLG_FEATURES, LLG_EDGE_LABELS

# === CONFIGURATION ===
DOMAIN_FILE = "../dataset/goose/blocks/domain.pddl"
PROBLEM_FILE = "../dataset/goose/blocks/test_small/blocks3-task00-seed9000.pddl"
MODEL_FILE = "blocks_llg_mean_8.dt"

# === 1. Build the graph ===
llg = LiftedLearningGraph(domain_pddl=DOMAIN_FILE, problem_pddl=PROBLEM_FILE)
llg._compute_graph_representation()
G = llg.G
print("Nodes",len(llg.G.nodes),"Edges", len(llg.G.edges))

# === 2. Prepare node colors ===
node_colors = []
for node in G.nodes:
    # Default color
    color = "skyblue"
    
    # Goal nodes
    if hasattr(llg, "_pos_goal_nodes") and node in llg._pos_goal_nodes:
        color = "green"
    elif hasattr(llg, "_neg_goal_nodes") and node in llg._neg_goal_nodes:
        color = "red"
    
    # Activated propositions can be colored differently if you have a state
    # Here we just highlight goal nodes
    
    node_colors.append(color)

# === 3. Prepare edge colors based on LLG edge types ===
edge_colors = []
for u, v, data in G.edges(data=True):
    label = data.get("edge_label", LLG_EDGE_LABELS["neutral"])
    # Map labels to colors
    if label in [LLG_EDGE_LABELS["pre_pos"], LLG_EDGE_LABELS["eff_pos"]]:
        edge_colors.append("green")
    elif label in [LLG_EDGE_LABELS["pre_neg"], LLG_EDGE_LABELS["eff_neg"]]:
        edge_colors.append("red")
    elif label == LLG_EDGE_LABELS["ground"]:
        edge_colors.append("blue")
    else:
        edge_colors.append("gray")

# === 4. Draw the graph using Network X===
plt.figure(figsize=(15, 12))
pos = nx.spring_layout(G, seed=42)  # positions for all nodes

nx.draw(
    G, pos,
    with_labels=True,
    node_size=500,
    node_color=node_colors,
    edge_color=edge_colors,
    font_size=8,
    width=1
)
plt.title("Lifted Learning Graph (LLG) - Blocks World")
plt.savefig("llg_graph.png", dpi=300)
print("Graph saved as llg_graph.png")
plt.show()


