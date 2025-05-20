## 🧬 MARL-PTM: A Modular Multi-Agent Reinforcement Learning System for Predicting Post-Translational Modifications

**MARL-PTM** is a state-of-the-art multi-agent deep learning framework for predicting residue-specific post-translational modifications (PTMs) using six biological data modalities. Designed around a **Centralized Training, Decentralized Execution (CTDE)** paradigm, MARL-PTM simulates specialized biological reasoning through reinforcement-learned PTM agents.

---

## 🔧 I. Full Architecture Overview

### 🧩 Agent Layers

| **Level**       | **Agent Name**    | **Function**                                                   |
| --------------- | ----------------- | -------------------------------------------------------------- |
| Data Agents (6) | `SequenceAgent`   | Encodes amino acid motifs and sequential dependencies          |
|                 | `StructureAgent`  | Learns from DSSP and 3D features (e.g., helix, sheet, coil)    |
|                 | `GraphAgent`      | Processes residue-residue contact graphs                       |
|                 | `NetworkAgent`    | Embeds pathway and interaction network context                 |
|                 | `ExpressionAgent` | Captures context-specific expression patterns                  |
|                 | `ProteoformAgent` | Encodes historical PTM sites and isoform variants              |
| Integration     | `IntegratorAgent` | Learns to fuse embeddings across modalities via attention      |
| PTM Agents (13) | `PTMAgent_k`      | Learns PTM-specific classifiers (e.g., phosphorylation, etc.)  |
| Aggregator      | `AggregatorAgent` | Resolves competition between PTM agents to finalize prediction |

---

```text
[ Protein_ID ]
     ↓
[ Data Loaders ] ──┬────────────┬────────────┬────────────┬────────────┬────────────┬────────────┐
                   ↓            ↓            ↓            ↓            ↓            ↓            ↓
            [Seq Agent]   [Str Agent]  [Graph Agent]  [Expr Agent]  [Prot Agent]  [Net Agent]   ...
                   ↓            ↓            ↓            ↓            ↓            ↓
                   └──────→ [Integrator Agent: Fuse 6 × [L, D] → H] ────────→
                                                   ↓
                                    ┌──────────────┴──────────────┐
                                    ↓             ↓               ↓
                            [PTM Agent 1]   ...   [PTM Agent 13]
                                    ↓             ↓
                                  PTM Scores [L] × 13
                                    ↓
                           [Aggregator Agent: Argmax across PTMs]
                                    ↓
                      {residue: (PTM type, confidence score)}
```

## 🔁 II. Training Methodology

### 🏗 Centralized Training, Decentralized Execution (CTDE)

* **Training**: All agents access shared embeddings and optimize jointly.
* **Execution**: Each PTM agent independently predicts residue-level scores.
* **Communication**: Through vector embeddings; no direct messaging between agents.

### 🎯 Reward Structure

| **Condition**                    | **Reward** |
| -------------------------------- | ---------- |
| ✅ Correct PTM and correct site   | +1.0       |
| ❌ Incorrect PTM (false positive) | -1.0       |
| 🔄 Correct site, wrong PTM       | +0.5       |
| 🧬 Motif/Proteoform matched      | +0.25      |

Bonus rewards reflect biological plausibility even in ambiguous sites.

### 🔍 Optimization

* Loss: `BCEWithLogitsLoss` + optional reward-shaping
* Optimizer: `Adam` with optional warmup + decay
* Strategy: Multi-agent shared gradients through integrator → data agents

---

## 📁 III. Codebase Overview

```
MARL-PTM/
│
├── data_loaders/          # Modality-specific tensor loaders
│   ├── sequence_loader.py
│   ├── structure_loader.py
│   ├── graph_loader.py
│   ├── expression_loader.py
│   ├── proteoform_loader.py
│   └── network_loader.py
│
├── agents/                # Agent architectures and logic
│   ├── base_agent.py
│   ├── sequence_agent.py
│   ├── structure_agent.py
│   ├── graph_agent.py
│   ├── expression_agent.py
│   ├── proteoform_agent.py
│   ├── network_agent.py
│   ├── integrator_agent.py
│   ├── ptm_agent.py
│   └── aggregator_agent.py
│
├── training/              # Training orchestration
│   ├── train_loop.py
│   ├── replay_buffer.py
│   ├── rewards.py
│   └── scheduler.py
│
├── configs/               # YAML configuration files
│   └── config.yaml
│
├── utils/                 # Logging, visualization, metrics
│   ├── metrics.py
│   ├── logger.py
│   └── visualization.py
│
├── inference/             # Inference & PTM site prediction
│   └── predict_ptms.py
│
└── main.py                # Entry point for training
```

---

## 📦 Installation

```bash
git clone https://github.com/yourusername/MARL-PTM.git
cd MARL-PTM
pip install -r requirements.txt
```

Optional: install `torch-geometric` with correct CUDA version via:
[https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)

---

## 🚀 Getting Started

To train your model on synthetic or real-world datasets:

```bash
python main.py
```

To predict PTMs for trained models:

```bash
python inference/predict_ptms.py
```

Modify `configs/config.yaml` to set paths, agents, and hyperparameters.

---

## 📊 Evaluation

Performance is measured per PTM using:

* Precision / Recall / F1-score
* ROC-AUC (optional)
* Top-k residue prediction accuracy

---

## 📚 Citation

If you use MARL-PTM in your research, please cite:

> "MARL-PTM: A Multi-Agent Reinforcement Learning Framework for Predicting Post-Translational Modifications", \[Author names], 2025. bioRxiv (in preparation)

---

## 🧠 Contributions Welcome!

We’re happy to collaborate on:

* Additional PTM types (e.g., SUMOylation, ADP-ribosylation)
* Real-world PTM data integration
* New reward strategies or graph encoders

Feel free to fork the repo, open issues, or submit a pull request!

---

Would you like this converted to a `README.md` file or want badges (license, build, citation) added next?
