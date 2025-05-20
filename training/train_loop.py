# training/train_loop.py

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class Trainer:
    def __init__(self, data_agents, integrator, ptm_agents, aggregator, data_loader_dict, criterion, device='cpu'):
        self.data_agents = data_agents
        self.integrator = integrator
        self.ptm_agents = ptm_agents
        self.aggregator = aggregator
        self.data_loader_dict = data_loader_dict
        self.criterion = criterion
        self.device = device

        self.input_dim = integrator.input_dim  # expected embedding dim

        # Optimizers for all agents
        self.optimizers = {
            name: optim.Adam(agent.parameters(), lr=1e-3)
            for name, agent in {**data_agents, **ptm_agents, "integrator": integrator}.items()
        }

    def train(self, protein_ids, label_dict, num_epochs=10):
        for epoch in range(num_epochs):
            epoch_loss = 0.0

            for pid in tqdm(protein_ids, desc=f"Epoch {epoch+1}/{num_epochs}"):

                # ========== Step 1: Load raw inputs ==========
                raw_inputs = {}
                for mod, loader in self.data_loader_dict.items():
                    value = loader.get_tensor(pid)
                    if isinstance(value, torch.Tensor):
                        value = value.to(self.device)
                    raw_inputs[mod] = value

                # ========== Step 2: Generate modality embeddings ==========
                context_embeddings = {}

                for mod, agent in self.data_agents.items():
                    if mod == "graph":
                        num_residues = raw_inputs["sequence"].shape[0]
                        node_feats = torch.ones(num_residues, 1).to(self.device)
                        context_embeddings["graph"] = agent(node_feats, raw_inputs["graph"])
                    elif isinstance(raw_inputs[mod], torch.Tensor):
                        context_embeddings[mod] = agent(raw_inputs[mod])
                    else:
                        # Inject placeholder zero tensor for non-tensor modalities
                        L = raw_inputs["sequence"].shape[0]
                        D = self.input_dim
                        context_embeddings[mod] = torch.zeros((L, D), device=self.device)

                # ========== Step 3: Fuse modalities ==========
                integrated = self.integrator(context_embeddings)  # [L, D]

                # ========== Step 4: Run all PTM agents ==========
                ptm_outputs = {
                    ptm: self.ptm_agents[ptm](integrated)
                    for ptm in self.ptm_agents
                }

                # ========== Step 5: Compute loss ==========
                total_loss = 0.0
                if pid in label_dict:
                    for ptm_type, agent in self.ptm_agents.items():
                        preds = ptm_outputs[ptm_type]  # [L]
                        target_mask = torch.zeros_like(preds)

                        for res_idx, true_ptm in label_dict[pid].items():
                            if true_ptm == ptm_type and res_idx < len(target_mask):
                                target_mask[res_idx] = 1.0

                        loss = self.criterion(preds, target_mask.to(self.device))
                        total_loss += loss

                # ========== Step 6: Backpropagation ==========
                for opt in self.optimizers.values():
                    opt.zero_grad()
                total_loss.backward()
                for opt in self.optimizers.values():
                    opt.step()

                epoch_loss += total_loss.item()

            print(f"🧠 Epoch {epoch+1}/{num_epochs} - Total Loss: {epoch_loss:.4f}")
