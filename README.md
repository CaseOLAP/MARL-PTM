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


## Project Structure
```
  marl-ptm/
  │── data/                     # Data directory (protein sequences, pathways, gene expression)
  │── models/                   # Trained models
  │── utils/                    # Utility functions
  │── main.py                   # Entry point for training and evaluation
  │── data_processing.py         # Data preprocessing and feature extraction
  │── marl_agents.py             # Defines multi-agent RL architecture
  │── train_marl.py              # Training pipeline
  │── evaluate_marl.py           # Evaluation and benchmarking
  │── reward_function.py         # Reward mechanism for reinforcement learning
  │── config.py                  # Configuration file (hyperparameters)
  │── requirements.txt           # Dependencies
  │── README.md                  # Project documentation

```

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
```

## Usage
### Preprocess Data
```python
python data_processing.py --input data/proteins.fasta --output processed_data/
```

### Train the Mulit-Agent Model
```python
python train_marl.py --epochs 50 --batch_size 64
```

### Evaluate the Model
```python
python evaluate_marl.py --test_data data/test_set.csv
```

## Expected Results
- Improved PTM site prediction accuracy by integrating multi-omic data.
- Higher interpretability using attention-based visualization.
- Biological relevance via pathway-informed reinforcement learning.

## Citation
If you use this work, please cite

```
@article{marl-ptm,
  author = {Your Name, Collaborators},
  title = {Multi-Agent Reinforcement Learning for PTM Prediction},
  year = {2025},
  journal = {ArXiv Preprint},
  url = {https://github.com/your-username/marl-ptm-prediction}
}
```

### Contact & Contributions
For questions, reach out via ```dibakarsigdel@ucla.edu```. Contributions are welcome via pull requests! 🚀


