import argparse
import yaml
import torch

from training.train_loop import Trainer
from training.trainer import initialize_agents, initialize_ptm_agent
from env.marl_env import MultiAgentEnvironment
from data.sequence_loader import SequenceLoader
from data.structure_loader import StructureLoader
from data.graph_loader import GraphLoader
from data.expression_loader import ExpressionLoader
from data.proteoform_loader import ProteoformLoader
from reward.reward_agent import RewardAgent
from evaluation.evaluator import Evaluator

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def build_environment(state_data, max_seq_len):
    return MultiAgentEnvironment(state_sources=state_data, max_seq_len=max_seq_len)

def simulate_data_loader(state_data, labels, context_info):
    """
    Simulated batch iterator for demonstration.
    Returns one item at a time: (protein_id, ground_truth, context)
    """
    for protein_id in labels:
        yield protein_id, labels[protein_id], context_info.get(protein_id, {})

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/default_config.yaml')
    parser.add_argument('--agent_config', default='config/agent_config.yaml')
    args = parser.parse_args()

    # Load configs
    config = load_yaml(args.config)
    agent_config = load_yaml(args.agent_config)
    device = torch.device(config.get('device', 'cpu'))

    # Initialize agents
    agents = initialize_agents(agent_config, device=device)
    ptm_agent = initialize_ptm_agent(num_agents=config['integration']['num_agents'],
                                     action_dim=config['integration']['action_dim'],
                                     device=device)

    # Load data
    print("Loading biological data...")
    sequence_loader = SequenceLoader()
    structure_loader = StructureLoader("data/structures/")
    graph_loader = GraphLoader("data/pathway_graph.edgelist")
    expression_loader = ExpressionLoader("data/expression_matrix.csv")
    proteoform_loader = ProteoformLoader("data/canonical.fasta", "data/isoforms.fasta")

    state_data = {
        "sequence": sequence_loader.batch_embed_sequences(sequence_loader.load_fasta_sequences("data/sequences.fasta")),
        "structure": structure_loader.load_batch_structures(),
        "graph": graph_loader.build_embeddings(),
        "expression": expression_loader.load_embeddings(),
        "proteoform": proteoform_loader.build_features()
    }

    # Load dummy labels and context (replace with actual loader)
    dummy_labels = {pid: torch.randint(0, 2, (1024,)) for pid in state_data['sequence'].keys()}
    dummy_context = {pid: {'is_critical': [[False]*1024],
                           'context_supported': {k: [[False]*1024] for k in state_data}} for pid in dummy_labels}

    # Build environment
    env = build_environment(state_data, config['training']['max_seq_len'])

    # Setup reward logic and training
    reward_cfg = {"agreement_threshold": config['training']['reward_agreement_threshold']}
    data_loader = simulate_data_loader(state_data, dummy_labels, dummy_context)

    trainer = Trainer(
        agents=agents,
        ptm_agent=ptm_agent,
        env=env,
        data_loader=data_loader,
        reward_config=reward_cfg,
        device=device
    )

    trainer.train(
        num_epochs=config['training']['num_epochs'],
        epsilon=config['training']['epsilon_start'],
        update_target_every=config['training']['update_target_every']
    )

if __name__ == "__main__":
    main()
