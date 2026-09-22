import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import numpy as np


def get_concept_diversity_ours(concept_list):

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-mpnet-base-v2")

    embeddings_concept = model.encode(concept_list)
    similarity_matrix = model.similarity(embeddings_concept, embeddings_concept)
    # lower_similarity_matrix = np.tril(similarity_matrix)
    identity_matrix = np.eye(similarity_matrix.shape[0])
    similarity_matrix = similarity_matrix - identity_matrix
    average_similarity = (similarity_matrix.sum())/(similarity_matrix.shape[0] * (similarity_matrix.shape[0]-1))
    return(1-average_similarity)
    