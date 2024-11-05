import torch
import torch.nn as nn
import torch.nn.functional as F
from Net import Entity_Aware_Embedding, MGSA, GNN
import numpy as np
from relationship_utils import compute_embedding_difference, calculate_affinity_matrix, calculate_similarity_weights


class TAN(nn.Module):
    def __init__(self, word_vec_dir, edge, constraint, type_num, rel_num, opt):
        super(TAN, self).__init__()

        word_vec = torch.from_numpy(np.load(word_vec_dir))
        word_dim = word_vec.shape[-1]
        self.rel_num = rel_num
        self.type_num = type_num
        self.constraint = constraint
        self.entity_type = {}

        self.Embedding = Entity_Aware_Embedding(word_vec, opt['pos_dim'], opt['max_pos_length'], opt['lambda'])
        self.MGSA_encoder = MGSA(word_dim * 3, opt['hidden_size'])
        sent_out = 3 * opt['hidden_size']

        graph_out = opt['graph_out']
        class_dim = opt['class_dim']
        self.fc1 = nn.Linear(sent_out + graph_out * 2, class_dim)
        self.fc2 = nn.Linear(graph_out * 3, class_dim)
        self.classifier = nn.Linear(class_dim, rel_num)
        self.MGSA_encoder = GNN(edge, type_num, rel_num, opt['graph_emb'], opt['graph_hid'], graph_out,
                                 encoder=opt['self.MGSA_encoder'], num_layers=opt['num_layers'], num_heads=opt['num_heads'])
        self.drop = nn.Dropout(opt['dropout'])
        self.init_weight()

        self.affinity_matrix = None
        self.centroids = {}
        self.type_constraint_matrix = np.zeros((self.type_num, self.type_num, self.rel_num))

    def init_weight(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, data):
        X_Rel = None

        if len(data) == 10:
            X, X_Pos1, X_Pos2, X_Len, X_Ent1, X_Ent2, X_Mask, X_Type, X_Scope, X_Rel = data
        else:
            X, X_Pos1, X_Pos2, X_Len, X_Ent1, X_Ent2, X_Mask, X_Type, X_Scope = data
        X = self.Embedding(X, X_Pos1, X_Pos2, X_Ent1, X_Ent2)
        X = self.MGSA_encoder(X, X_Mask)

        Type, Rel = self.self.MGSA_encoder()

        Type_s = F.embedding(X_Type, Type).reshape(X_Type.shape[0], -1)
        X = torch.cat([X, Type_s], -1)

        Type_r = []
        for i in range(self.rel_num):
            type_rep = F.embedding(self.constraint[i], Type)
            avg_type_rep = type_rep.mean(1)
            Type_r.append(avg_type_rep.reshape(-1))
        Type_r = torch.stack(Type_r)
        Constraints = torch.cat([Rel, Type_r], -1)

        X = torch.relu(self.fc1(X))
        Constraints = torch.relu(self.fc2(Constraints))

        X = self.sentence_attention(X, X_Scope, Constraints, X_Rel)
        return X

    def compute_affinity(self, embeddings):
        embedding_diff = compute_embedding_difference(embeddings)
        self.affinity_matrix = calculate_affinity_matrix(embeddings.keys(), embedding_diff)

    def update_type_constraints(self, embedding_diff):
        type_constraint_matrix = np.zeros((self.type_num, self.type_num, self.rel_num))

        for relation, centroid in self.centroids.items():
            similarity_weights = calculate_similarity_weights(embedding_diff, centroid)

            for (ent1, ent2), weight in similarity_weights.items():
                type1 = self.entity_type[ent1]
                type2 = self.entity_type[ent2]
                type_constraint_matrix[type1, type2, relation] += weight

        for type1 in range(self.type_num):
            for type2 in range(self.type_num):
                if np.sum(type_constraint_matrix[type1, type2]) > 0:
                    type_constraint_matrix[type1, type2] /= np.sum(type_constraint_matrix[type1, type2])

        self.type_constraint_matrix = type_constraint_matrix

    def sentence_attention(self, X, X_Scope, Constraints, X_Rel=None):
        bag_output = []
        if X_Rel is not None:
            Con = F.embedding(X_Rel, Constraints)
            for i in range(X_Scope.shape[0]):
                bag_rep = X[X_Scope[i][0]: X_Scope[i][1]]
                att_score = F.softmax(bag_rep.matmul(Con[i]), 0).view(1, -1)
                att_output = att_score.matmul(bag_rep)
                bag_output.append(att_output.squeeze())
            bag_output = torch.stack(bag_output)
            bag_output = self.drop(bag_output)
            bag_output = self.classifier(bag_output)
        else:
            att_score = X.matmul(Constraints.t())
            for s in X_Scope:
                bag_rep = X[s[0]:s[1]]
                bag_score = F.softmax(att_score[s[0]:s[1]], 0).t()
                att_output = bag_score.matmul(bag_rep)
                bag_output.append(torch.diagonal(F.softmax(self.classifier(att_output), -1)))
            bag_output = torch.stack(bag_output)
        return bag_output

    def collect_embeddings(self, data):
        embeddings = {}
        X_Rel = None

        if len(data) == 10:
            X, X_Pos1, X_Pos2, X_Len, X_Ent1, X_Ent2, X_Mask, X_Type, X_Scope, X_Rel = data
        else:
            X, X_Pos1, X_Pos2, X_Len, X_Ent1, X_Ent2, X_Mask, X_Type, X_Scope = data
        X = self.Embedding(X, X_Pos1, X_Pos2, X_Ent1, X_Ent2)
        X = self.MGSA_encoder(X, X_Mask)

        for i in range(X_Type.shape[0]):
            for j in range(X_Type.shape[1]):
                ent_pair = (X_Type[i][0].item(), X_Type[i][1].item())
                if ent_pair not in embeddings:
                    embeddings[ent_pair] = []
                embeddings[ent_pair].append(X[i].detach().cpu().numpy())

        for ent_pair in embeddings:
            embeddings[ent_pair] = np.mean(embeddings[ent_pair], axis=0)

        return embeddings
