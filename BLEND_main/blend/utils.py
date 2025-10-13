import scanpy as sc
import scipy.sparse as sp
import numpy as np
from math import sqrt
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA, NMF
from scipy.sparse import csr_matrix, diags
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr
from scipy.sparse.linalg import svds
from sklearn.utils.extmath import randomized_svd, squared_norm


def S_calculate(adata, spatial_k=30, expression_k=15, algorithm='auto', metric='euclidean'):
    """
    Construct a similarity matrix (S) for an AnnData object based on spatial and PCA features.

    Steps:
    1. Compute PCA if not already available.
    2. Use spatial coordinates to find the top-k neighbors.
    3. Within these neighbors, refine neighbor selection using PCA-based distances.
    4. Compute similarity between each cell and its refined neighbors using cosine similarity.
    5. Build a symmetric sparse similarity matrix.

    Parameters
    ----------
    adata : AnnData
        AnnData object with `obsm['spatial']` and gene expression matrix.
    n_neighbors : int, optional (default: 30)
        Number of neighbors to consider in the spatial space (k).
    neighbors1 : int, optional (default: 15)
        Number of neighbors to refine using PCA (k1).
    algorithm : str, optional (default: 'auto')
        NearestNeighbors search algorithm (passed to sklearn).
    metric : str, optional (default: 'euclidean')
        Distance metric for NearestNeighbors.

    Returns
    -------
    S1 : scipy.sparse.csr_matrix
        Symmetric sparse similarity matrix of shape (n_cells, n_cells).
    """

    # Ensure PCA exists
    if "X_pca" not in adata.obsm:
        sc.tl.pca(adata, svd_solver='arpack')

    # Step 1: spatial neighbors
    nn_spatial = NearestNeighbors(n_neighbors=spatial_k + 1, algorithm=algorithm, metric=metric)
    nn_spatial.fit(adata.obsm['spatial'])
    _, nearest_neighbors_spot = nn_spatial.kneighbors(adata.obsm['spatial'])

    # Step 2: refine neighbors using PCA
    nearest_neighbors = np.zeros((adata.shape[0], expression_k + 1), dtype=int)
    nn_model_pca = NearestNeighbors(n_neighbors=expression_k + 1, algorithm=algorithm, metric=metric)

    for n in range(adata.shape[0]):
        nn_model_pca.fit(adata.obsm['X_pca'][nearest_neighbors_spot[n]])
        _, nearest_neighbors_pca = nn_model_pca.kneighbors(adata.obsm['X_pca'][nearest_neighbors_spot[n]])
        nearest_neighbors[n, :] = nearest_neighbors_spot[n][nearest_neighbors_pca[0]]

    # Step 3: build sparse similarity matrix
    data_list, indices_list, indptr_list = [], [], [0]

    for i in range(adata.shape[0]):
        cell_near = nearest_neighbors[i, :]
        for j in cell_near:
            similarity = np.exp(1 - cosine(adata.obsm['X_pca'][i], adata.obsm['X_pca'][j]))
            data_list.append(similarity)
            indices_list.append(j)
        indptr_list.append(len(data_list))

    S1 = csr_matrix((data_list, indices_list, indptr_list), shape=(adata.shape[0], adata.shape[0]))
    S1 = (S1 + S1.T) / 2 

    return S1

def median_normalize(X):
    """
    Median normalization for count matrices (dense or sparse).
    
    Steps:
    1. Compute row sums (library size per cell/sample).
    2. Scale each row so that its total equals the median library size.
    3. Apply log1p transformation if values are large (>= 15).
    
    Parameters
    ----------
    X : array-like or scipy.sparse matrix
        Input count matrix (cells x genes).
    
    Returns
    -------
    X_norm : same type as input
        Median-normalized matrix, log-transformed if needed.
    """
    # Step 1: Compute row sums (library size)
    if sp.issparse(X):
        row_sums = X.sum(axis=1).A1  # efficient 1D extraction
    else:
        row_sums = X.sum(axis=1)

    # Step 2: Handle all-zero rows
    row_sums_safe = row_sums.copy()
    row_sums_safe[row_sums_safe == 0] = np.inf
    
    median_val = np.median(row_sums_safe[row_sums_safe != np.inf])
    scaling_factors = median_val / row_sums_safe
    scaling_factors[row_sums_safe == np.inf] = 0

    # Step 3: Scale the matrix
    if sp.issparse(X):
        scaler = diags(scaling_factors)
        X_norm = scaler.dot(X)
    else:
        X_norm = scaling_factors[:, None] * X

    # Step 4: Optional log1p transform for stability
    if X_norm.max() >= 15:
        X_norm = np.log1p(X_norm)

    return X_norm

def calc_rank(X, method='wedge', n_pca=100, wedgebound=0.085, alracut=78, alratrshv=6):
    """

    calculate rank of a matrix using WEDGE and ALRA methods.
    Args:
        X: input data. Rows are cells/spots and columns are genes.
        method: the way to calculate rank.
        n_pca: number of PCA.
        wedgebound: bound of fluctuation in WEDGE. Defualt: 0.085.
        alracut: default False. If True, median_normalize input data.
        alratrshv:
    Returns:
        rank of WEDGE or rank of ALRA
    """
    if method not in ['wedge', 'alra']:
        raise ValueError('can only calculate rank using "wedge" or "alra"')

    if method == 'wedge':
        n_rank = calc_wedge_rank(X, n_pca, wedgebound)
    elif method == 'alra':
        n_rank = calc_alra_rank(X, n_pca, alracut, alratrshv)

    return n_rank

def calc_wedge_rank(X, n_pca, wedgebound):
    """
    Estimate the effective rank of a matrix using the WEDGE method.

    Parameters
    ----------
    X : ndarray or sparse matrix, shape (n_samples, n_features)
        Input data matrix.
    n_pca : int
        Number of principal components (maximum rank to consider).
    wedgebound : float
        Threshold for the eigenvalue ratio test used in WEDGE.

    Returns
    -------
    n_rank : int
        Estimated rank of the input matrix.
    """

    n_pca = min(n_pca, X.shape[1] - 1)
    W0, single_value, _ = svds(X, k=n_pca, which='LM')
    sv = single_value[range(n_pca - 1, -1, -1)]
    sv = sv[sv > 0]
    n_svd = min(n_pca, len(sv) - 1)
    sv = sv / sv[0]
    latent_new_diff = sv[0:n_svd] / sv[1:n_svd + 1] - 1
    n_rank = 2
    for i in range(len(latent_new_diff) - 10):
        n_rank = i + 1
        if (latent_new_diff[i] >= wedgebound) and all(latent_new_diff[i + 1:11] < wedgebound):
            break
    n_rank = max(n_rank, 3)

    return n_rank


def calc_alra_rank(X, n_pca, alracut, alratrshv):
    """
    Estimate the effective rank of a matrix using the ALRA method.

    Parameters
    ----------
    X : ndarray or sparse matrix, shape (n_samples, n_features)
        Input data matrix.
    n_pca : int
        Number of principal components (maximum rank to consider).
    alracut : int
        Index cutoff for computing mean and standard deviation of singular value gaps.
    alratrshv : float
        Threshold value for determining the rank based on standardized gaps.

    Returns
    -------
    n_rank : int
        Estimated rank of the input matrix.
    """

    n_pca = min(n_pca, X.shape[1] - 1)
    W0, single_value, _ = svds(X, k=n_pca, which='LM')
    sv = single_value[range(n_pca - 1, -1, -1)]
    s1 = np.delete(sv, -1)
    s2 = np.delete(sv, 0)
    s_diff = s1 - s2
    mu = np.mean(s_diff[alracut:])
    sigma = np.std(s_diff[alracut:])
    thresh = (s_diff - mu) / sigma
    n_rank = max(np.where(thresh > alratrshv)[0]) + 1

    return n_rank

def calculate_rank(adata, setrank=30, multi=5, method_calcrank='wedge'):
    if setrank == 0:
        rank = multi * calc_rank(X=adata.obsm['X_norm'], method=method_calcrank)
        if rank==0:
            print('rank=0, please set a rank!')
    else:
        rank = setrank
        
    return rank

def sc_embedding_calculate(sc_adata, n_components, max_iter=1000, random_state=100):
    """
    Compute single-cell embeddings using Non-negative Matrix Factorization (NMF).

    This function takes a normalized single-cell dataset stored in an AnnData object 
    and applies NMF to extract low-rank embeddings.

    Parameters
    ----------
    sc_adata : AnnData
        Single-cell AnnData object. 
        Must contain the normalized matrix in `sc_adata.obsm['X_norm']` (shape: cells × genes).
    rank1 : int
        Target rank (number of components) for the NMF decomposition.
    max_iter : int, optional (default=1000)
        Maximum number of iterations for the NMF solver.
    random_state : int, optional (default=100)
        Random seed for reproducibility.

    Returns
    -------
    sc_H : np.ndarray
        Embedding matrix of shape (rank1, n_cells).
        Each column corresponds to a low-dimensional embedding of a single cell.
    """

    nmf = NMF(n_components=n_components, max_iter=max_iter, random_state=random_state)
    sc_H = nmf.fit_transform(sc_adata.obsm['X_norm'].T)
    sc_H = sc_H.T 

    return sc_H

def initialize(X, init='wdgsvd', rank=10, random_state=123, eps=1e-6, n_pca=100, bound=0.085):
    """
    Initialize factor matrices W and H for Nonnegative Matrix Factorization (NMF).

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Input data matrix to be factorized.

    init : str, default='wdgsvd'
        Initialization method. Options:
        - 'nndsvd': Nonnegative Double Singular Value Decomposition (NNDSVD).
        - 'wdgsvd': Weighted/Modified SVD-based initialization.

    rank : int, default=10
        Target rank (number of latent factors).

    random_state : int, default=123
        Random seed for reproducibility in randomized SVD.

    eps : float, default=1e-6
        Small constant for numerical stability. Values smaller than `eps` are set to zero.

    n_pca : int, default=100
        Number of PCA components to compute during SVD-based initialization (used in 'wdgsvd').

    bound : float, default=0.085
        Threshold parameter (currently unused in this implementation).

    Returns
    -------
    W : ndarray, shape (n_samples, rank)
        Initialized nonnegative basis matrix.

    H : ndarray, shape (rank, n_features)
        Initialized nonnegative coefficient matrix.

    Raises
    ------
    ValueError
        If `rank > min(n_samples, n_features)` or unsupported initialization method is provided.

    Notes
    -----
    - The 'nndsvd' initialization is designed to provide a good starting point for NMF,
      reducing convergence time compared to random initialization.
    - The 'wdgsvd' initialization ensures nonnegativity by adjusting SVD components
      based on the dominance of positive vs. negative parts.
    """

    ncell, ngene = X.shape
    if rank > min(ncell, ngene):
        raise ValueError("Initialization only works when rank <= min(n_samples, n_features)")

    if init == 'nndsvd':
        # --- NNDSVD Initialization ---
        # Perform randomized SVD
        U, S, V = randomized_svd(X, rank, random_state=random_state)
        W, H = np.zeros(U.shape), np.zeros(V.shape)

        # Use the first singular triplet (guaranteed nonnegative)
        W[:, 0] = np.sqrt(S[0]) * np.abs(U[:, 0])
        H[0, :] = np.sqrt(S[0]) * np.abs(V[0, :])

        # Process remaining singular vectors
        for j in range(1, rank):
            x, y = U[:, j], V[j, :]
            # Split into positive and negative parts
            x_p, y_p = np.maximum(x, 0), np.maximum(y, 0)
            x_n, y_n = np.abs(np.minimum(x, 0)), np.abs(np.minimum(y, 0))

            # Compute norms
            x_p_nrm, y_p_nrm = sqrt(squared_norm(x_p)), sqrt(squared_norm(y_p))
            x_n_nrm, y_n_nrm = sqrt(squared_norm(x_n)), sqrt(squared_norm(y_n))

            # Choose the part with larger magnitude
            m_p, m_n = x_p_nrm * y_p_nrm, x_n_nrm * y_n_nrm
            if m_p > m_n:
                u, v, sigma = x_p / x_p_nrm, y_p / y_p_nrm, m_p
            else:
                u, v, sigma = x_n / x_n_nrm, y_n / y_n_nrm, m_n

            lbd = np.sqrt(S[j] * sigma)
            W[:, j] = lbd * u
            H[j, :] = lbd * v

        # Ensure nonnegativity
        W[W < eps] = 0
        H[H < eps] = 0

    elif init == 'wdgsvd':
        # --- Weighted/Modified SVD Initialization ---
        n_pca = min(n_pca, ngene - 1)
        W0, single_value, H0 = svds(X, k=n_pca, which='LM', random_state=random_state)

        # Sort singular values in descending order
        single_value = single_value[::-1]
        W0 = W0[:, ::-1]
        H0 = H0[::-1, :]

        # Adjust W0 to be nonnegative
        W0_P, W0_N = np.maximum(W0, 0), np.maximum(-W0, 0)
        Wn_P_big = np.sum(W0_P ** 2, axis=0)
        Wn_N_big = np.sum(W0_N ** 2, axis=0)

        if np.sum(Wn_P_big >= Wn_N_big) > 0:
            W0[:, Wn_P_big >= Wn_N_big] = W0_P[:, Wn_P_big >= Wn_N_big]
        if np.sum(Wn_P_big < Wn_N_big) > 0:
            W0[:, Wn_P_big < Wn_N_big] = W0_N[:, Wn_P_big < Wn_N_big]

        W = W0[:, :rank]

        # Adjust H0 to be nonnegative
        H0_P, H0_N = np.maximum(H0, 0), np.maximum(-H0, 0)
        Hn_P_big = np.sum(H0_P ** 2, axis=1)
        Hn_N_big = np.sum(H0_N ** 2, axis=1)

        if np.sum(Hn_P_big >= Hn_N_big) > 0:
            H0[Hn_P_big >= Hn_N_big, :] = H0_P[Hn_P_big >= Hn_N_big, :]
        if np.sum(Hn_P_big < Hn_N_big) > 0:
            H0[Hn_P_big < Hn_N_big, :] = H0_N[Hn_P_big < Hn_N_big, :]

        H = H0[:rank, :]

    else:
        raise ValueError('Initialization method must be one of ("wdgsvd", "nndsvd")')

    return np.abs(W), np.abs(H)


def reorder_rows(H, sc_H):
    """
    Reorder the rows of matrices H and sc_H based on maximum pairwise correlation.

    Parameters
    ----------
    H : np.ndarray
        Original matrix of shape (n_rows, n_columns).
    sc_H : np.ndarray
        Reference matrix of shape (n_rows, n_columns).

    Returns
    -------
    H_sorted : np.ndarray
        Reordered version of H based on maximum correlation.
    sc_H_sorted : np.ndarray
        Reordered version of sc_H aligned with H_sorted.
    """
    n = H.shape[0]
    correlation_matrix = np.zeros((n, n), dtype=float)

    # Compute pairwise row correlations
    for i in range(n):
        for j in range(n):
            correlation_matrix[i, j] = pearsonr(H[i, :], sc_H[j, :])[0]

    H_sorted, sc_H_sorted = [], []
    used_H, used_sc_H = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)

    for _ in range(n):
        correlation_matrix[used_H, :] = -np.inf
        correlation_matrix[:, used_sc_H] = -np.inf

        max_idx = np.argmax(correlation_matrix)
        i, j = np.unravel_index(max_idx, correlation_matrix.shape)

        H_sorted.append(H[i, :])
        sc_H_sorted.append(sc_H[j, :])

        used_H[i] = True
        used_sc_H[j] = True

    return np.array(H_sorted), np.array(sc_H_sorted)

'''
from sklearn.preprocessing import StandardScaler

def reorder_rows_based_on_max_correlation(H, sc_H):
    """
    Reorder rows of H and sc_H based on maximum pairwise Pearson correlation.

    Parameters
    ----------
    H : np.ndarray
        Original matrix, shape (n_rows, n_columns).
    W : np.ndarray
        Not used in sorting here, kept for compatibility.
    sc_H : np.ndarray
        Matrix to match with H, shape (n_rows, n_columns).

    Returns
    -------
    H_sorted : np.ndarray
        Reordered H matrix.
    sc_H_sorted : np.ndarray
        Reordered sc_H matrix.
    """

    n = H.shape[0]

    # Z-score normalization along each row (zero mean, unit variance)
    H_z = StandardScaler(with_mean=True, with_std=True).fit_transform(H.T).T
    sc_H_z = StandardScaler(with_mean=True, with_std=True).fit_transform(sc_H.T).T

    # Compute row-to-row Pearson correlation via dot product
    correlation_matrix = H_z @ sc_H_z.T / (H.shape[1] - 1)

    H_sorted, sc_H_sorted = [], []
    used_rows_H = np.zeros(n, dtype=bool)
    used_rows_sc_H = np.zeros(n, dtype=bool)

    for _ in range(n):
        # Mask used rows
        correlation_matrix[used_rows_H, :] = -np.inf
        correlation_matrix[:, used_rows_sc_H] = -np.inf

        # Find max correlation pair
        max_idx = np.argmax(correlation_matrix)
        max_i, max_j = np.unravel_index(max_idx, correlation_matrix.shape)

        # Append corresponding rows
        H_sorted.append(H[max_i, :])
        sc_H_sorted.append(sc_H[max_j, :])

        # Mark as used
        used_rows_H[max_i] = True
        used_rows_sc_H[max_j] = True

    return np.array(H_sorted), np.array(sc_H_sorted)
'''