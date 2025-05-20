import os
import random
import pandas as pd
import numpy as np
import networkx as nx

# ========== CONFIGURATION ========== #
base_dir = "./data"
num_proteins = 100
seq_length = 100
num_pathways = 20
ptm_types = [
    "phosphorylation", "acetylation", "ubiquitination", "methylation", "glycosylation",
    "sumoylation", "nitration", "s-nitrosylation", "palmitoylation", "myristoylation",
    "hydroxylation", "sulfation", "citrullination"
]

motif_dict = {
    "phosphorylation": ["SP", "TP", "RxxS"],
    "acetylation": ["K"],
    "ubiquitination": ["K"],
    "methylation": ["R", "K"],
    "glycosylation": ["N"],
    "sumoylation": ["ΨKxE"],
    "nitration": ["Y"],
    "s-nitrosylation": ["C"],
    "palmitoylation": ["C"],
    "myristoylation": ["G"],
    "hydroxylation": ["P"],
    "sulfation": ["Y"],
    "citrullination": ["R"]
}
# =================================== #

folders = ["sequence", "structure", "graph", "expression", "proteoform", "labels", "network"]
for folder in folders:
    os.makedirs(os.path.join(base_dir, folder), exist_ok=True)

def generate_motif_sequence(ptm_type, length):
    motif = random.choice(motif_dict[ptm_type])
    if "x" in motif:
        motif = motif.replace("x", random.choice("ACDEFGHIKLMNPQRSTVWY"))
    if "Ψ" in motif:
        motif = motif.replace("Ψ", random.choice("IVLFM"))
    seq = ''.join(random.choices("ACDEFGHIKLMNPQRSTVWY", k=length - len(motif)))
    insert_pos = random.randint(0, length - len(motif))
    return seq[:insert_pos] + motif + seq[insert_pos:]

def generate_sequences():
    data = []
    for i in range(num_proteins):
        ptm_type = random.choice(ptm_types)
        seq = generate_motif_sequence(ptm_type, seq_length)
        data.append({"protein_id": f"protein_{i}", "sequence": seq, "dominant_ptm": ptm_type})
    pd.DataFrame(data).to_csv(f"{base_dir}/sequence/sequences.csv", index=False)

def generate_structures():
    data = []
    for i in range(num_proteins):
        dssp = ''.join(random.choices("HEC", k=seq_length))
        data.append({"protein_id": f"protein_{i}", "dssp": dssp})
    pd.DataFrame(data).to_csv(f"{base_dir}/structure/structures.csv", index=False)

def generate_graphs():
    graph_data = []
    for i in range(num_proteins):
        G = nx.watts_strogatz_graph(seq_length, 4, 0.3)
        for u, v in G.edges():
            graph_data.append({
                "protein_id": f"protein_{i}",
                "residue_1": u,
                "residue_2": v
            })
    pd.DataFrame(graph_data).to_csv(f"{base_dir}/graph/graphs.csv", index=False)

def generate_expression():
    data = []
    for i in range(num_proteins):
        trend = np.linspace(0, 1, seq_length) + np.random.normal(0, 0.1, seq_length)
        for j, expr in enumerate(trend):
            data.append({
                "protein_id": f"protein_{i}",
                "residue": j,
                "expression": round(min(max(expr, 0), 1), 3)
            })
    pd.DataFrame(data).to_csv(f"{base_dir}/expression/expression.csv", index=False)

def generate_proteoforms():
    records = []
    for i in range(num_proteins):
        ptm_type = random.choice(ptm_types)
        positions = random.sample(range(seq_length), 3)
        for pos in positions:
            records.append({
                "protein_id": f"protein_{i}",
                "residue": pos,
                "prior_ptm": ptm_type
            })
    pd.DataFrame(records).to_csv(f"{base_dir}/proteoform/proteoforms.csv", index=False)

def generate_labels():
    labels = []
    seq_df = pd.read_csv(f"{base_dir}/sequence/sequences.csv")
    for _, row in seq_df.iterrows():
        prot_id = row['protein_id']
        sequence = row['sequence']
        ptm_type = row['dominant_ptm']
        motif = random.choice(motif_dict[ptm_type])
        motif_pos = sequence.find(motif.replace("x", ""))
        if motif_pos != -1:
            labels.append({
                "protein_id": prot_id,
                "residue": motif_pos,
                "ptm_type": ptm_type
            })
    pd.DataFrame(labels).to_csv(f"{base_dir}/labels/ptm_sites.csv", index=False)

def generate_protein_pathway_network():
    pathways = [f"pathway_{i}" for i in range(num_pathways)]
    edges = []
    for i in range(num_proteins):
        protein_id = f"protein_{i}"
        assigned_pathways = random.sample(pathways, random.randint(1, 3))
        for pw in assigned_pathways:
            edges.append({
                "source_node": protein_id,
                "edge_type": "PART_OF",
                "target_node": pw
            })
    pd.DataFrame(edges).to_csv(f"{base_dir}/network/protein_pathway_edges.csv", index=False)

def generate_all():
    generate_sequences()
    generate_structures()
    generate_graphs()
    generate_expression()
    generate_proteoforms()
    generate_labels()
    generate_protein_pathway_network()
    print("✅ All biologically realistic synthetic data generated for MARL-PTM-v2.")

generate_all()
