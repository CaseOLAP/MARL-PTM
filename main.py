# main.py

import os
import yaml
import torch
import pandas as pd
from training.train_loop import Trainer
from agents.sequence_agent import SequenceAgent
from agents.structure_agent import StructureAgent
from agents.graph_agent import GraphAgent
from agents.expression_agent import ExpressionAgent
from agents.proteoform_agent import ProteoformAgent
from agents.network_agent import NetworkAgent
from agents.integrator_agent import IntegratorAgent
from agents.aggregator_agent import AggregatorAgent
from agents.ptm_agent import PTMAgent

from data_loaders.sequence_loader import SequenceLoader
from data_loaders.structure_loader import StructureLoader
from data_loaders.graph_loader import GraphLoader
from data_loaders.expression_loader import ExpressionLoader
from data_loaders.proteoform_loader import ProteoformLoader
from data_loaders.network_loader import NetworkLoader

#from utils.logger import Logger

# ========== Load Configuration ==========
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

device = "cuda" if torch.cuda.is_available() else "cpu"
logger = Logger(log_dir=config["paths"]["log_dir"])
ptm_types = config["ptm_types"]

# ========== Initialize Data Loaders ==========
base_path = config["paths"]["base_data_dir"]
file_cfg = config["files"]

data_loaders = {
    "sequence": SequenceLoader(os.path.join(base_path, file_cfg["sequence"])),
    "structure": StructureLoader(os.path.join(base_path, file_cfg["structure"])),
    "graph": GraphLoader(os.path.join(base_path, file_cfg["graph"])),
    "expression": ExpressionLoader(os.path.join(base_path, file_cfg["expression"])),
    "proteoform": ProteoformLoader(os.path.join(base_path, file_cfg["proteoform"])),
    "network": NetworkLoader(os.path.join(base_path, file_cfg["network"]))
}

# Standard wrapper for interface uniformity
class LoaderWrapper:
    def __init__(self, loader, tensor_method):
        self.loader = loader
        self.tensor_method = tensor_method

    def get_tensor(self, pid):
        return getattr(self.loader, self.tensor_method)(pid)

modality_loaders = {
    "sequence": LoaderWrapper(data_loaders["sequence"], "get_sequence_tensor"),
    "structure": LoaderWrapper(data_loaders["structure"], "get_structure_tensor"),
    "graph": LoaderWrapper(data_loaders["graph"], "get_graph_edge_index"),
    "expression": LoaderWrapper(data_loaders["expression"], "get_expression_tensor"),
    "proteoform": LoaderWrapper(data_loaders["proteoform"], "get_proteoform_sites"),
    "network": LoaderWrapper(data_loaders["network"], "get_pathway_nodes")
}

# ========== Initialize All Agents ==========
data_agents = {
    "sequence": SequenceAgent(device=device),
    "structure": StructureAgent(device=device),
    "graph": GraphAgent(device=device),
    "expression": ExpressionAgent(device=device),
    "proteoform": ProteoformAgent(device=device),
    "network": NetworkAgent(device=device)
}

integrator = IntegratorAgent(device=device)

# 13 PTM-specific agents using same architecture
ptm_agents = {
    ptm: PTMAgent(device=device) for ptm in ptm_types
}

aggregator = AggregatorAgent(ptm_types=ptm_types, device=device)

# ========== Load Ground-Truth PTM Labels ==========
label_df = pd.read_csv(os.path.join(base_path, file_cfg["labels"]))
label_dict = {}
for _, row in label_df.iterrows():
    pid = row["protein_id"]
    residue = int(row["residue"])
    ptm = row["ptm_type"]
    if pid not in label_dict:
        label_dict[pid] = {}
    label_dict[pid][residue] = ptm

# ========== Train ==========
trainer = Trainer(
    data_agents=data_agents,
    integrator=integrator,
    ptm_agents=ptm_agents,
    aggregator=aggregator,
    data_loader_dict=modality_loaders,
    criterion=torch.nn.BCELoss(),
    device=device
)

protein_ids = list(label_dict.keys())
trainer.train(protein_ids=protein_ids, label_dict=label_dict, num_epochs=config["training"]["num_epochs"])

# ========== Save All Models ==========
checkpoint_dir = config["paths"]["checkpoint_dir"]
os.makedirs(checkpoint_dir, exist_ok=True)

# Save PTM agents
for ptm, agent in ptm_agents.items():
    agent.save(os.path.join(checkpoint_dir, f"{ptm}_agent.pt"))

# Save integrator
integrator.save(os.path.join(checkpoint_dir, "integrator.pt"))

# Save data agents
for name, agent in data_agents.items():
    agent.save(os.path.join(checkpoint_dir, f"{name}_agent.pt"))

print(f"✅ All models saved to: {checkpoint_dir}")
