import torch
import torch.nn as nn


class NetworkWithCategoryEmbedding(nn.Module):
    def __init__(self, input_size, num_categories, embedding_dim):
        super().__init__()
        self.category_embedding = nn.Embedding(num_categories, embedding_dim)
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256 + embedding_dim, 256)
        self.batchnorm_emb = nn.BatchNorm1d(embedding_dim)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.1)
        self.batchnorm = nn.BatchNorm1d(256)
        self.activation_fn = nn.ELU()

    def forward(self, x, categories):
        emb = self.batchnorm_emb(self.category_embedding(categories))
        x = self.activation_fn(self.fc1(x))
        x = torch.cat((x, emb), dim=1)
        x = self.activation_fn(self.fc2(x))
        x = self.batchnorm(x)
        x = self.activation_fn(self.fc3(x))
        x = self.dropout(x)
        x = self.activation_fn(self.fc4(x))
        x = self.activation_fn(self.fc5(x))
        x = self.fc6(x)
        return x
