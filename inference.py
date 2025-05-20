import os
import torch
import yaml
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

# ========== Load Configuration ==========
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

device = "cuda" if torch.cuda.is_available() else "cpu"
base_path = config["paths"]["base_data_dir"]
checkpoint_dir = config["paths"]["checkpoint_dir"]
ptm_types = config["ptm_types"]

# ========== Load Trained Agents ==========
data_agents = {
    "sequence": SequenceAgent(device=device),
    "structure": StructureAgent(device=device),
    "graph": GraphAgent(device=device),
    "expression": ExpressionAgent(device=device),
    "proteoform": ProteoformAgent(device=device),
    "network": NetworkAgent(device=device)
}
integrator = IntegratorAgent(device=device)
ptm_agents = {ptm: PTMAgent(device=device) for ptm in ptm_types}
aggregator = AggregatorAgent(ptm_types=ptm_types, device=device)

# Load all weights from checkpoint
for ptm in ptm_types:
    ptm_agents[ptm].load(os.path.join(checkpoint_dir, f"{ptm}_agent.pt"))
integrator.load(os.path.join(checkpoint_dir, "integrator.pt"))
for name in data_agents:
    data_agents[name].load(os.path.join(checkpoint_dir, f"{name}_agent.pt"))

# ========== Load Data Loaders ==========
data_loaders = {
    "sequence": SequenceLoader(os.path.join(base_path, config["files"]["sequence"])),
    "structure": StructureLoader(os.path.join(base_path, config["files"]["structure"])),
    "graph": GraphLoader(os.path.join(base_path, config["files"]["graph"])),
    "expression": ExpressionLoader(os.path.join(base_path, config["files"]["expression"])),
    "proteoform": ProteoformLoader(os.path.join(base_path, config["files"]["proteoform"])),
    "network": NetworkLoader(os.path.join(base_path, config["files"]["network"]))
}

# ========== Run Inference on 10 Proteins ==========
for i in range(10):
    test_pid = f"protein_{i}"

    with torch.no_grad():
        inputs = {
            "sequence": data_loaders["sequence"].get_sequence_tensor(test_pid).to(device),
            "structure": data_loaders["structure"].get_structure_tensor(test_pid).to(device),
            "graph": data_loaders["graph"].get_graph_edge_index(test_pid).to(device),
            "expression": data_loaders["expression"].get_expression_tensor(test_pid).to(device),
            "proteoform": torch.zeros(config["model"]["max_sequence_length"]).long().to(device),
            "network": [
                int(pw.split("_")[1]) for pw in data_loaders["network"].get_pathway_nodes(test_pid)
            ]
        }

        context = {}
        for name in data_agents:
            if name == "graph":
                L = inputs["sequence"].shape[0]
                node_feats = torch.ones(L, 1).to(device)
                context["graph"] = data_agents["graph"](node_feats, inputs["graph"])
            elif name == "network":
                context["network"] = data_agents["network"](inputs["network"], residue_len=inputs["sequence"].shape[0])
            else:
                context[name] = data_agents[name](inputs[name])

        # Integrate modalities
        integrated = integrator(context)  # [L, D]

        # Predict each PTM
        ptm_outputs = {
            ptm: ptm_agents[ptm](integrated) for ptm in ptm_types
        }

        # Aggregate final results
        predictions = aggregator(ptm_outputs)  # {residue_idx: (ptm_type, score)}

    # ========== Print Result ==========
    print(f"\n📌 Final PTM predictions for {test_pid}:")
    for res_idx, (ptm, score) in predictions.items():
        print(f" - Residue {res_idx:3d}: {ptm:<15s} (score: {score:.4f})")
