import torch
import torch.optim as optim
from marl_agents import SequenceAgent, StructureAgent, GraphAgent, GeneExpressionAgent
from reward_function import reward_function
from config import config

def train():
    # Initialize Agents
    sequence_agent = SequenceAgent(config["input_dim"], config["hidden_dim"], config["output_dim"])
    structure_agent = StructureAgent(config["input_dim"], config["hidden_dim"], config["output_dim"])
    graph_agent = GraphAgent(config["input_dim"], config["hidden_dim"], config["output_dim"])
    gene_expression_agent = GeneExpressionAgent(config["input_dim"], config["hidden_dim"], config["output_dim"])

    agents = [sequence_agent, structure_agent, graph_agent, gene_expression_agent]
    optimizers = [optim.Adam(agent.parameters(), lr=config["lr"]) for agent in agents]

    for epoch in range(config["epochs"]):
        for agent, optimizer in zip(agents, optimizers):
            inputs = torch.randn(config["batch_size"], config["input_dim"])  # Simulated data
            labels = torch.randint(0, 2, (config["batch_size"],))  # Simulated PTM labels

            optimizer.zero_grad()
            outputs = agent(inputs)
            loss = torch.nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}: Loss {loss.item()}")

if __name__ == "__main__":
    train()
