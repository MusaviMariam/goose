# Lifted Learning Graph (LLG) - Examples & Analysis

This directory contains examples and analysis of the **Lifted Learning Graph (LLG)** representation used in the GOOSE planning project.

## Files

1. **`llg_analysis.py`** — Static analysis script
   - Displays LLG constants, features, and edge labels
   - Shows graph construction steps
   - Explains the injected variable features (4-dim unit vectors)
   - Runs without heavy dependencies (no torch_geometric issues)
   - **To run:** `python3 llg_analysis.py`

2. **`llg_example.ipynb`** — Interactive Jupyter notebook
   - Instantiates `LiftedLearningGraph` on a toy Blocks domain
   - Shows node feature vectors (`x`) and their shapes
   - Displays edge indices for each label type
   - Attempts NetworkX visualization
   - Includes unit tests and assertions
   - Falls back to manual construction if import fails
   - **To run:** Open in VS Code or Jupyter

## Quick Start

### Option 1: Run Analysis (No Dependencies)
```bash
cd /home/mariam/goose
python3 llg_analysis.py
```

### Option 2: Open Notebook
```bash
# In VS Code, open:
/home/mariam/goose/llg_example.ipynb
```

## LLG Overview

### What is LLG?
The **Lifted Learning Graph** is a graph representation for relational planning problems that mixes:
- **Lifted nodes:** predicates, actions (domain schema)
- **Grounded nodes:** objects, ground facts, goals (problem instance)

This hybrid approach allows GNNs to reason at both schema and instance levels.

### Key Components

#### Node Types (6 one-hot features)
| Type | Code | Meaning |
|------|------|---------|
| P | 0 | Predicate (lifted) |
| A | 1 | Action (lifted) |
| G | 2 | Positive goal (grounded) |
| N | 3 | Negative goal (grounded) |
| S | 4 | Activated fact (in current state) |
| O | 5 | Object |

#### Edge Labels (6 types)
| Label | Code | Purpose |
|-------|------|---------|
| neutral | 0 | Unspecified relations |
| ground | 1 | Grounding links (fact ↔ predicate, fact ↔ args) |
| pre_pos | 2 | Positive precondition |
| pre_neg | 3 | Negative precondition |
| eff_pos | 4 | Positive effect |
| eff_neg | 5 | Negative effect |

#### Node Features (10-dimensional)
- **Dims 0-5:** One-hot encoding of node type (6 values)
- **Dims 6-9:** Injected position features (4-dim unit vectors)
  - Deterministically seeded by index (0..59)
  - Distinguishes argument positions: `on(A,B)` vs `on(B,A)`

### Graph Construction (5 Steps)

1. **Add objects:** One node per domain object (feature O)
2. **Add predicates:** One node per predicate (feature P)
3. **Connect predicates ↔ objects:** Fully connected with "neutral" edges
4. **Add goals:** Grounded goal facts with variable nodes, connected via "ground" edges
5. **Add actions:** Action nodes with parameter nodes, precondition/effect edges

### Data Structures

After construction, `LiftedLearningGraph` provides:

```python
llg = LiftedLearningGraph(domain_pddl, problem_pddl)

# Nodes & features
llg.x              # Tensor of shape (num_nodes, 10)
llg.num_nodes      # Total nodes
llg._node_to_i     # Dict: node identifier → index in x

# Edges
llg.edge_indices   # Dict[label_idx → Tensor of shape (2, num_edges)]
llg.num_edges      # Total edges across all labels
llg.G              # NetworkX graph (for inspection)

# State handling
x_new, edges_new = llg.state_to_tensor(state)  # Extend for ground facts
```

## Example: Blocks World (2 blocks)

```
Domain: blocks
Objects: a, b
Predicates: on/2, clear/1, handempty/0
Action: stack(?x, ?y) 
  Pre: clear(?x), clear(?y), handempty()
  Eff: on(?x, ?y), ¬clear(?y), ¬handempty()
Goal: on(a, b)

Graph Nodes (~20-30):
  - Objects: a, b
  - Predicates: on, clear, handempty
  - Goal fact: (on, (a,b)) + variable nodes
  - Action: stack + parameter nodes
  - Aux nodes for preconditions/effects

Graph Edges (~40-60):
  - neutral: predicate ↔ object connections
  - ground: goal structure, argument grounding
  - pre_pos, pre_neg: precondition links
  - eff_pos, eff_neg: effect links
```

## File Locations in Codebase

```
/home/mariam/goose/learner/
├── representation/
│   ├── __init__.py          # Exports LiftedLearningGraph as "llg"
│   ├── base_class.py        # Representation base class
│   ├── llg.py               # LiftedLearningGraph implementation ← MAIN FILE
│   ├── slg.py               # Symbolic Learning Graph (comparison)
│   ├── flg.py               # Fact Learning Graph (comparison)
│   └── dlg.py               # Domain Learning Graph (comparison)
└── train_gnn.py             # Uses --rep llg for GNN training
```

## Running LLG in Experiments

From `goose/learner/train_gnn.py`:
```bash
python3 train_gnn.py <domain> -r llg --aggr mean -L 8 --epochs 1000
```

Example (from README):
```bash
python3 train_gnn.py blocks -L 8 --aggr mean --rep llg --save-file blocks_llg_mean_8.dt
```

## Implementation Details

### Injected Features (`_if`)
Ensures that variable positions can be distinguished:
```python
for idx in range(60):
    torch.manual_seed(idx)
    rep = 2 * torch.rand(4) - 1  # U[-1,1]
    rep /= torch.linalg.norm(rep)  # Normalize
    self._if.append(rep)
```

### State to Tensor Transformation
When given a state (list of ground facts):
1. For each ground fact:
   - If it's already a goal node → set activation bit (S feature)
   - Otherwise → add new node(s) and edges to `x` and `edge_indices`
2. Return extended feature matrix and updated edge indices

### Visualization Methods
- `state_to_cgraph()` — Create colored NetworkX graph for visualization
- Can be converted to PNG/SVG using NetworkX layout algorithms

## Troubleshooting

### Import Issues
If you get `ModuleNotFoundError: No module named 'torch_geometric'`:
- The notebook has fallback to manual construction
- Or run `python3 llg_analysis.py` which avoids that import

### Version Conflicts
Current environment has:
- numpy < 2.0 (for scipy/sklearn compatibility)
- torch_geometric has TorchScript compilation issues (known limitation)

**Workaround:** Use the analysis script or manual notebook mode.

## References

- **Location:** `goose/learner/representation/llg.py`
- **Base class:** `goose/learner/representation/base_class.py`
- **Training script:** `goose/learner/train_gnn.py`
- **Tests:** `goose/goose/tests/test_*.py`
