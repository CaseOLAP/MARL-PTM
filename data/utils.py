import numpy as np
import torch

def pad_sequence_to_length(seq_tensor, target_length, padding_value=0.0):
    """
    Pads or truncates a tensor along the sequence axis to a fixed length.
    
    Parameters:
        seq_tensor (torch.Tensor): Tensor of shape (L, D)
        target_length (int): Desired sequence length
        padding_value (float): Value to pad with

    Returns:
        torch.Tensor of shape (target_length, D)
    """
    current_len = seq_tensor.size(0)
    feature_dim = seq_tensor.size(1)

    if current_len == target_length:
        return seq_tensor
    elif current_len > target_length:
        return seq_tensor[:target_length, :]
    else:
        pad = torch.full((target_length - current_len, feature_dim), padding_value)
        return torch.cat([seq_tensor, pad], dim=0)


def normalize_tensor(tensor, axis=0):
    """
    Applies z-score normalization to a tensor along a given axis.

    Parameters:
        tensor (torch.Tensor): Input tensor
        axis (int): Axis along which to normalize

    Returns:
        torch.Tensor: Normalized tensor
    """
    mean = tensor.mean(dim=axis, keepdim=True)
    std = tensor.std(dim=axis, keepdim=True)
    return (tensor - mean) / (std + 1e-8)


def tokenize_sequence(seq, vocab):
    """
    Converts an amino acid string into integer indices based on a vocabulary.

    Parameters:
        seq (str): Amino acid sequence (e.g., "MKT...")
        vocab (dict): Mapping from amino acid to index

    Returns:
        list[int]: List of token indices
    """
    return [vocab.get(aa, vocab.get('X', 0)) for aa in seq]


def build_vocab():
    """
    Builds a standard amino acid vocabulary mapping.

    Returns:
        dict: Mapping of amino acids to integer tokens
    """
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    vocab = {aa: idx + 1 for idx, aa in enumerate(amino_acids)}
    vocab['X'] = 0  # unknown/masked
    return vocab
