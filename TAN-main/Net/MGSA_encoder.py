import torch
import torch.nn as nn
import torch.nn as nn
from torch_geometric.nn import GCNConv

class MGSA(nn.Module):
    def __init__(self, emb_dim, hidden_size, entity_type_dim, vocab_size, num_entity_types):
        super(MGSA, self).__init__()
        self.word_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim)
        self.entity_type_embedding = nn.Embedding(num_embeddings=num_entity_types, embedding_dim=entity_type_dim)
        self.W = nn.Linear(emb_dim, emb_dim * 3)
        self.cnn = nn.Conv1d(emb_dim * 3, hidden_size, kernel_size=3, padding=1)
        self.fc = nn.Linear(hidden_size * 3, hidden_size)
        self.init_weight()
    def init_weight(self):
        nn.init.xavier_uniform_(self.cnn.weight)
        nn.init.zeros_(self.cnn.bias)

    def forward(self, X, head_entity_idx, tail_entity_idx, head_type_idx, tail_type_idx):
        word_embeds = self.word_embedding(X)
        head_type_embeds = self.entity_type_embedding(head_type_idx)
        tail_type_embeds = self.entity_type_embedding(tail_type_idx)
        head_embed = word_embeds[torch.arange(X.size(0)), head_entity_idx]
        tail_embed = word_embeds[torch.arange(X.size(0)), tail_entity_idx]
        enriched_head = torch.cat([head_embed, head_type_embeds], dim=-1)
        enriched_tail = torch.cat([tail_embed, tail_type_embeds], dim=-1)
        D = enriched_tail - enriched_head
        attention_weights = torch.softmax(torch.matmul(D.unsqueeze(1), word_embeds.transpose(1, 2)), dim=-1)
        s1 = torch.matmul(attention_weights, word_embeds).squeeze(1)
        X = word_embeds.transpose(1, 2)
        X = self.W(X)
        conv_out = self.cnn(X)
        conv_out = torch.relu(conv_out)
        q1 = torch.max(conv_out[:, :, :head_entity_idx], dim=2)[0]
        q2 = torch.max(conv_out[:, :, head_entity_idx:tail_entity_idx], dim=2)[0]
        q3 = torch.max(conv_out[:, :, tail_entity_idx:], dim=2)[0]
        s2 = torch.cat([q1, q2, q3], dim=1)
        s2 = self.fc(s2)
        beta = nn.Parameter(torch.tensor(0.5))
        s = beta * s1 + (1 - beta) * s2
        return s


class GNN(nn.Module):
    def __init__(self, edge_index, type_num, rel_num, emb_dim, emb_mid, emb_out, num_layers=2):
        super(GNN, self).__init__()
        self.type_num = type_num
        self.edge_index = edge_index
        self.node_embedding = nn.Embedding(type_num + rel_num, emb_dim)
        
        self.encoder = nn.ModuleList(
            [
                GCNConv(emb_dim if i == 0 else emb_mid, emb_mid if i < num_layers - 1 else emb_out)
                for i in range(num_layers)
            ]
        )
        
        self.act = nn.GELU()
        self.init_weight()

    def init_weight(self):
        nn.init.xavier_uniform_(self.node_embedding.weight)

    def forward(self):
        X = self.node_embedding.weight
        for i, layer in enumerate(self.encoder):
            X = self.act(layer(X, self.edge_index))

        Type, Rel = X[:self.type_num], X[self.type_num:]
        return Type, Rel