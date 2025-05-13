
# MARL-PTM: Multi-Agent Reinforcement Learning for Post-Translational Modification Prediction

## Overview

**MARL-PTM** is a modular, interpretable, and biologically grounded framework for predicting post-translational modification (PTM) sites in proteins. It employs a multi-agent reinforcement learning (MARL) architecture, where each agent is responsible for a specific biological modality, such as sequence, structure, expression, and interaction networks.

The system is designed to:
- Integrate multiple biological data sources (UniProt, AlphaFold, GTEx, STRING, etc.)
- Predict PTM sites with high confidence and biological context
- Provide interpretable decision pathways per prediction
- Support ablation, visualization, and biological validation

## Workflow


```

+------------------------+      +---------------------+      +---------------------+      
|  Protein Sequences     | ---> |   Sequence Agent    |---> |                       |      
+------------------------+      +---------------------+     |                       |      
+------------------------+      +---------------------+     |                       |      
|  Protein Structures    | ---> |   Structure Agent   |---> |                       |     
+------------------------+      +---------------------+     |                       |       +-----------------------+
+------------------------+      +---------------------+     |                       |       |                       |
|  Pathway Graph Data    | ---> |   Graph Agent        |--->| PTM Integration Agent  |  --->| Predicted PTM Sites   |
+------------------------+      +---------------------+     |  (Meta-policy layer)  |       |                       |
+------------------------+      +---------------------+     |                       |       +-----------------------+
|  Expression Profiles   | ---> |   Expression Agent   | --->|                      |    
+------------------------+      +---------------------+     |                       |      
+------------------------+      +---------------------+     |                       |     
|  Proteoform Domains    | ---> |   Proteoform Agent   | --->|                      |      
+------------------------+      +---------------------+      +----------------------+     
                                                                 |
                                                                 |
                                                                 v
                                                      +------------------------+
                                                      |     Reward Agent       |
                                                      |  - Correctness Score   |
                                                      |  - Confidence Score    |
                                                      |  - Contextual Match    |
                                                      |  - Agent Agreement     |
                                                      +------------------------+
                                                                 |
                                                                 v
                                                      +------------------------+
                                                      |    Policy Updates      |
                                                      | (RL learning per agent)|
                                                      +------------------------+
```


## Folder Structure

```
marl_ptm/
├── agents/                  # All core and integration agents
├── reward/                  # Reward matrix and controller
├── data/                    # Biological data loaders and preprocessors
├── env/                     # Environment logic and replay buffers
├── training/                # Training loop, scheduler, logger
├── evaluation/              # Metrics, ablation, case studies, visualization
├── config/                  # YAML configuration files
├── main.py                  # Main entry point to run training
└── README.md                # Project documentation
```

## Installation

```bash
git clone https://github.com/your-username/marl-ptm.git
cd marl-ptm

# Create a virtual environment
python -m venv env
source env/bin/activate  # For Windows: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Dependencies include**: PyTorch, torch-geometric, transformers, BioPython, scikit-learn, seaborn, matplotlib

## Usage

**Run training:**

```bash
python main.py --config config/default_config.yaml --agent_config config/agent_config.yaml
```

**Evaluate model:**

Update `evaluation/evaluator.py` to pass test set and use the `Evaluator` class to compute metrics.

## Configuration

* `default_config.yaml`: Global training parameters
* `agent_config.yaml`: Agent-specific network dimensions and learning rates

## Citation

If you use MARL-PTM in your work, please cite:

```
@article{marlptm2025,
  title={Multi-Agent Reinforcement Learning for Post-Translational Modification Prediction},
  author={Your Name and Collaborators},
  year={2025},
  journal={Bioinformatics / Preprint},
  url={https://github.com/your-username/marl-ptm}
}
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

----
