import torch
from marl_agents import SequenceAgent
from config import config

def evaluate():
    model = SequenceAgent(config["input_dim"], config["hidden_dim"], config["output_dim"])
    model.load_state_dict(torch.load("models/sequence_agent.pth"))

    test_input = torch.randn(10, config["input_dim"])
    predictions = model(test_input).argmax(dim=1)
    print("Predictions:", predictions.tolist())

if __name__ == "__main__":
    evaluate()
