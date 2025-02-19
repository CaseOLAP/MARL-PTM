config = {
    "input_dim": 1280,   # Protein sequence embedding dimension
    "hidden_dim": 128,   # Hidden layer size
    "output_dim": 2,     # PTM prediction: (0 = No PTM, 1 = PTM)
    "lr": 0.001,         # Learning rate
    "batch_size": 64,
    "epochs": 50,
    "gamma": 0.95,       # Discount factor for RL
    "epsilon": 1.0,      # Exploration-exploitation balance
    "epsilon_decay": 0.995,
    "epsilon_min": 0.01,
}
