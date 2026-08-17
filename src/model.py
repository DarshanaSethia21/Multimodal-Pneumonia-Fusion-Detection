import numpy as np
import torch
import torch.nn as nn


class FusionClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dims=(64, 16), dropout: float = 0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def build_fusion_input(image_embeddings, clinical_features):
    if isinstance(image_embeddings, np.ndarray):
        return np.concatenate([image_embeddings, clinical_features], axis=1)
    return torch.cat([image_embeddings, clinical_features], dim=1)