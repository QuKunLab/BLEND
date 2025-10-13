import numpy as np
import torch
import scanpy as sc
from .model import *
from .utils import *


def BLEND_fit(spatial_data = None, sc_data = None, spa_k=20, exp_k=15, setrank=0, device='cuda:0' ):

    S1 = S_calculate(spatial_data, spatial_k=spa_k, expression_k=exp_k, algorithm='auto', metric='euclidean')
    spatial_data.obsm['X_norm'] = median_normalize(spatial_data.X)
    
    rank = calculate_rank(spatial_data, setrank=setrank, multi=5, method_calcrank='wedge')
    W_initial, H_initial = initialize( spatial_data.obsm['X_norm'], init='wdgsvd', rank=rank, random_state=123, eps=1e-6, n_pca=100, bound=0.085)
    if sc_data is not None:
        sc_data.obsm['X_norm'] = median_normalize(sc_data.X)
        sc_H = sc_embedding_calculate(sc_data, n_components=rank, max_iter=1000)
        H_s, sc_H_s = reorder_rows(H_initial, sc_H)
    else:
        H_s = H_initial
        sc_H_s = None

    data1 = spatial_data.obsm['X_norm']
    rows, cols = H_s.shape
    V = np.zeros((rows, cols))
    W2 = W_fit(data1, W_initial, H_s, V, S1, lamb=10, rank=rank, tol=1e-3, weight=0.7, max_iter=100, block_size=5000, distance='frobenius', device=device)
    return data1, W2, H_s, sc_H_s, V, S1, rank