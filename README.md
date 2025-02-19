# Multi-Agent Reinforcement Learning for Post-Translational Modification (PTM) Prediction

## Project Overview
Post-translational modifications (PTMs) regulate protein function and play a crucial role in cellular signaling and disease progression. This project develops a **Multi-Agent Reinforcement Learning (MARL) model** to predict PTM sites by integrating **protein sequence, structural data, pathway interactions, and gene expression networks**.

## Key Features
- **Multi-Agent AI System**: Uses specialized agents for different biological aspects.
- **Reinforcement Learning (RL)**: Trained using a biologically-aware reward function.
- **Graph Neural Networks (GNNs)**: Models protein-pathway interactions.
- **Attention Mechanisms**: Enhances interpretability by highlighting key features.

---

## **Methodology**
### **Data Processing**
- **Protein Sequences**: Extracted from **UniProt**, represented using **ESM-2 embeddings**.
- **PTM Annotations**: Sourced from **PhosphoSitePlus, UniProt** for supervised learning.
- **Protein Structures**: Derived from **AlphaFold**, capturing secondary structure features.
- **Pathway Graph Data**: Constructed from **KEGG, Reactome, STRING**, processed using **GraphSAGE**.
- **Gene Expression Data**: Extracted from **GTEx, TCGA**, preprocessed using PCA & WGCNA.

### **Multi-Agent Reinforcement Learning (MARL)**
- **Sequence Agent**: Learns PTM patterns based on amino acid motifs.
- **Structure Agent**: Evaluates PTM probability based on structural constraints.
- **Graph Agent**: Captures pathway-specific PTM regulations.
- **Gene Expression Agent**: Identifies PTM relevance based on transcriptomic signals.
- **PTM Agent**: Integrates outputs from all agents to make the final PTM site prediction.
- **Reward Agent**: Provides feedback using accuracy, confidence, and pathway impact.

### **Training & Evaluation**
- Uses **Deep Q-Networks (DQN)** for agent training.
- Reward function **penalizes incorrect PTM predictions** and **rewards high-confidence biological insights**.
- Model is evaluated using **Precision, Recall, F1-score, and AUPRC**.

---

## Installation & Setup
```bash
# Clone the repository
git clone https://github.com/your-username/marl-ptm-prediction.git
cd marl-ptm-prediction

# Create a virtual environment
python -m venv env
source env/bin/activate  # For MacOS/Linux
# On Windows, use: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

