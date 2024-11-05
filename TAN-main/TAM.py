import numpy as np

def compute_centroid(embedding_diff):
    diffs = list(embedding_diff.values())
    centroid = np.mean(diffs, axis=0)
    return centroid
def compute_embedding_difference(embeddings):
    embedding_diff = {}
    for (ent1, ent2), emb in embeddings.items():
        embedding_diff[(ent1, ent2)] = np.abs(emb[ent1] - emb[ent2])
    return embedding_diff

def calculate_affinity_matrix(entity_pairs, embedding_diff):
    affinity_matrix = {}
    for (type1, type2) in entity_pairs:
        if (type1, type2) not in affinity_matrix:
            affinity_matrix[(type1, type2)] = []
        affinity_matrix[(type1, type2)].append(embedding_diff[(type1, type2)])
    for types, diffs in affinity_matrix.items():
        affinity_matrix[types] = np.mean(diffs)
    return affinity_matrix

def calculate_similarity_weights(embedding_diff, centroid):
    similarity_weights = {}
    for entity_pair, diff in embedding_diff.items():
        similarity_weights[entity_pair] = np.dot(diff, centroid) / (np.linalg.norm(diff) * np.linalg.norm(centroid))
    return similarity_weights