#!/usr/bin/env python3
"""
Minimal example showing LiftedLearningGraph structure by reading the code directly.
This avoids the torch_geometric import issues.
"""
import sys
sys.path.insert(0, '/home/mariam/goose/learner')

# Read and display the LLG implementation details
print("=" * 90)
print("LIFTED LEARNING GRAPH (LLG) - STATIC ANALYSIS")
print("=" * 90)

# Show enum definitions
print("\n[1] LLG_FEATURES Enum")
print("-" * 50)
llg_features = {
    "P": 0,  # is predicate
    "A": 1,  # is action
    "G": 2,  # is positive goal (grounded)
    "N": 3,  # is negative goal (grounded)
    "S": 4,  # is activated (grounded)
    "O": 5,  # is object
}
for name, idx in llg_features.items():
    print(f"  {name}: {idx}")
print(f"  → Total one-hot encoding features (ENC_FEAT_SIZE): {len(llg_features)}")

# Show edge labels
print("\n[2] LLG_EDGE_LABELS")
print("-" * 50)
llg_edges = {
    "neutral": 0,      # unspecified relation
    "ground": 1,       # grounding relation (predicate → args → objects)
    "pre_pos": 2,      # positive precondition
    "pre_neg": 3,      # negative precondition
    "eff_pos": 4,      # positive effect
    "eff_neg": 5,      # negative effect
}
for name, idx in llg_edges.items():
    print(f"  {name:10s}: {idx}")

# Class attributes
print("\n[3] LiftedLearningGraph Class Attributes")
print("-" * 50)
ENC_FEAT_SIZE = len(llg_features)
VAR_FEAT_SIZE = 4
print(f"  name: 'llg'")
print(f"  n_node_features: {ENC_FEAT_SIZE + VAR_FEAT_SIZE} = ENC_FEAT_SIZE({ENC_FEAT_SIZE}) + VAR_FEAT_SIZE({VAR_FEAT_SIZE})")
print(f"  n_edge_labels: {len(llg_edges)}")
print(f"  directed: False")
print(f"  lifted: True")

# Explain the architecture
print("\n[4] Graph Construction Steps")
print("-" * 50)
steps = [
    ("1. Objects", "Add nodes for each domain object with feature O (one-hot at index 5)"),
    ("2. Predicates", "Add nodes for each predicate with feature P (one-hot at index 0)"),
    ("3. Predicate-Object edges", "Connect all predicates to all objects with 'neutral' edges"),
    ("4. Goals", "For each goal fact (atom or negated atom):"),
    ("   - Goal nodes", "Add grounded node (pred, tuple_of_args) with G or N feature"),
    ("   - Variable nodes", "Add per-argument variable nodes with injected features from _if[i]"),
    ("   - Ground edges", "Connect goal → vars → objects and goal → predicate"),
    ("5. Actions", "For each action:"),
    ("   - Action node", "Add node with feature A"),
    ("   - Param nodes", "Add per-parameter nodes with injected features"),
    ("   - Prec/Eff edges", "Add precondition/effect edges with labels pre_pos, pre_neg, eff_pos, eff_neg"),
]
for desc, detail in steps:
    print(f"  {desc:25s} → {detail}")

# Explain VAR_FEAT_SIZE
print("\n[5] Injected Features (_if)")
print("-" * 50)
print(f"  Purpose: Encode variable position information beyond one-hot")
print(f"  Implementation: {VAR_FEAT_SIZE}-dim unit vectors seeded by index (idx=0..59)")
print(f"    - torch.manual_seed(idx) ensures deterministic injectivity")
print(f"    - Each vector is normalized: v /= ||v||")
print(f"    - Used for: action parameters, goal arguments, predicate arguments")
print(f"  Benefit: Distinguishes position 0 from position 1 in (on A B) vs (on B A)")

# Example scenario
print("\n[6] Example: Blocks World (2 blocks)")
print("-" * 50)
example = {
    "Domain": "blocks",
    "Objects": ["a", "b"],
    "Predicates": ["on/2", "clear/1", "handempty/0"],
    "Initial": ["clear(a)", "clear(b)", "handempty()"],
    "Goal": ["on(a,b)"],
    "Action": "stack(?x, ?y): pre=[clear(?x), clear(?y), handempty()], eff=[on(?x,?y), ¬clear(?y), ¬handempty()]"
}
for k, v in example.items():
    if isinstance(v, list):
        print(f"  {k:15s}: {', '.join(v)}")
    else:
        print(f"  {k:15s}: {v}")

print("\n[7] Expected Graph Nodes (Blocks World Example)")
print("-" * 50)
nodes_expected = [
    "a (object)",
    "b (object)",
    "on (predicate)",
    "clear (predicate)",
    "handempty (predicate)",
    "(on, (a, b)) [goal]",
    "((on, (a, b)), 0) [goal var for a]",
    "((on, (a, b)), 1) [goal var for b]",
    "stack (action)",
    "(stack, 'action-var-0') [action param ?x]",
    "(stack, 'action-var-1') [action param ?y]",
    "... aux nodes for preconditions/effects ...",
]
for i, node in enumerate(nodes_expected, 1):
    print(f"  {i:2d}. {node}")
print(f"  Total: ~25-40 nodes (varies with action/precondition complexity)")

print("\n[8] Key Methods")
print("-" * 50)
methods = [
    ("_construct_if()", "Precompute 60 deterministic unit vectors for variable positions"),
    ("_feature(node_type)", "Return one-hot feature vector for a node type (P, A, G, N, S, O)"),
    ("_if_feature(idx)", "Return feature with injected vector at _if[idx] in last 4 dims"),
    ("_compute_graph_representation()", "Main builder: creates lifted graph G from domain+problem"),
    ("str_to_state(s)", "Parse textual facts into (pred, [args]) tuples"),
    ("state_to_tensor(state)", "Extend x and edges for ground facts in a given state"),
    ("state_to_cgraph(state)", "Create colored NetworkX graph for visualization"),
]
for method, desc in methods:
    print(f"  {method:35s} → {desc}")

print("\n[9] Data Structures")
print("-" * 50)
print(f"  self.G: networkx.Graph")
print(f"    └─ Nodes: lifted + grounded entities from domain/problem")
print(f"    └─ Edges: with labels from LLG_EDGE_LABELS")
print(f"  self.x: Tensor of shape (num_nodes, n_node_features)")
print(f"    └─ Each row is a node feature vector (6 one-hot + 4 injected)")
print(f"  self.edge_indices: Dict[label_idx → Tensor of shape (2, num_edges)]")
print(f"    └─ edge_indices[0] = neutral edges")
print(f"    └─ edge_indices[1] = ground edges")
print(f"    └─ ... etc for each label")
print(f"  self._node_to_i: Dict[node → index in x]")
print(f"    └─ Maps graph node identifiers to row indices in feature matrix")

print("\n" + "=" * 90)
print("LLG: Lifted Learning Graph for relational planning with GNNs")
print("=" * 90)
print("\nKey insight: LLG mixes lifted (predicates, actions) and grounded (facts)")
print("             nodes, enabling the GNN to reason about both schema-level and")
print("             instance-level patterns in planning problems.")
print()
