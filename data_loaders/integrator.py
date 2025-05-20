import numpy as np
from dataprep.sequence_loader import load_sequence_features
from dataprep.structure_loader import load_structure_features
from dataprep.expression_loader import load_expression_features
from dataprep.graph_loader import load_graph_edges
from dataprep.proteoform_loader import load_proteoform_features

def integrate_features(
    sequence_path="data/sequence/sequences.csv",
    structure_path="data/structure/structures.csv",
    expression_path="data/expression/expression.csv",
    graph_path="data/graph/graphs.csv",
    proteoform_path="data/proteoform/proteoforms.csv",
    seq_len=100
):
    # Load all modalities
    seq_feats = load_sequence_features(sequence_path)
    struct_feats = load_structure_features(structure_path)
    expr_feats = load_expression_features(expression_path)
    prot_feats = load_proteoform_features(proteoform_path, seq_len=seq_len)
    graph_edges = load_graph_edges(graph_path)

    integrated = {}

    for protein_id in seq_feats:
        # Validate presence in all modalities
        if protein_id not in struct_feats or protein_id not in expr_feats or protein_id not in prot_feats:
            continue

        # Feature matrices
        f_seq = seq_feats[protein_id]           # (L, 20)
        f_struct = struct_feats[protein_id]     # (L, 3)
        f_expr = expr_feats[protein_id]         # (L, 1)
        f_prot = prot_feats[protein_id]         # (L, 13) ← 13 PTM types assumed

        # Align and concatenate along feature dimension
        try:
            residue_tensor = np.concatenate([f_seq, f_struct, f_expr, f_prot], axis=1)  # (L, D)
        except ValueError as e:
            print(f"⛔ Error integrating {protein_id}: {e}")
            continue

        integrated[protein_id] = {
            "features": residue_tensor,           # (L, D)
            "graph_edges": graph_edges.get(protein_id, [])  # optional graph structure
        }

    return integrated  # Dict[protein_id → {features, graph_edges}]
