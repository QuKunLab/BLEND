import time
import numpy as np
import pandas as pd
import torch
from scipy.sparse import coo_matrix, diags
from .utils import *


def W_fit(X, W, H, V, S, lamb, rank, tol=1e-3, weight=0.5, max_iter=100, block_size=5000, distance='frobenius', device=None):
    """
    Optimize basis matrix W for a single dataset using block-wise updates.

    Parameters
    ----------
    X : array-like or sparse matrix, shape (n_samples, n_features)
        Input data matrix.

    W : ndarray, shape (n_samples, rank)
        Initial basis matrix.

    H : ndarray, shape (rank, n_features)
        Coefficient matrix (fixed during optimization).

    V : ndarray, shape (rank, n_features)
        Auxiliary matrix (optimized).

    S : sparse matrix or None
        Similarity/adjacency matrix used to construct Laplacian regularizer.

    lamb : float
        Regularization strength for Laplacian penalty.

    rank : int
        Latent rank.

    tol : float, default=1e-3
        Convergence tolerance on the objective decrease.

    weight : float, default=0.5
        Weight applied to zero entries in X.

    max_iter : int, default=100
        Maximum number of optimization iterations.

    block_size : int, default=5000
        Block size for memory-efficient computation.

    distance : {'frobenius', 'KL'}, default='frobenius'
        Distance metric used for reconstruction loss.

    device : torch.device or None
        Device for computation. Default: "cuda:0" if available, else "cpu".

    Returns
    -------
    W : ndarray, shape (n_samples, rank)
        Optimized nonnegative basis matrix.
    """

    # Select device
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Dimension check
    if W.shape[1] != rank or H.shape[0] != rank:
        raise ValueError("Dimension mismatch between W/H and rank.")

    # Convert inputs to torch tensors
    W = torch.tensor(W, dtype=torch.float32, requires_grad=True, device=device)
    H = torch.tensor(H, dtype=torch.float32, device=device)
    V = torch.tensor(V, dtype=torch.float32, requires_grad=True, device=device)

    # Sparse representation of X
    X = coo_matrix(X)
    X = torch.sparse_coo_tensor(
        indices=torch.tensor(np.vstack((X.row, X.col)), dtype=torch.long, device=device),
        values=torch.tensor(X.data, dtype=torch.float32, device=device),
        size=X.shape
    ).coalesce()

    # Construct Laplacian matrix L
    if S is not None:
        if np.max(abs(S)) > 0:
            S = S / np.max(abs(S))
        row_sum = np.array(abs(S).sum(axis=1)).ravel()
        L = diags(row_sum, format='csr') - S
    else:
        raise ValueError("The similarity matrix has not been created")

    L = L.tocoo()
    L = torch.sparse_coo_tensor(
        indices=torch.tensor(np.vstack((L.row, L.col)), dtype=torch.long, device=device),
        values=torch.tensor(L.data, dtype=torch.float32, device=device),
        size=L.shape
    ).coalesce()

    # Optimizer
    optimizer = torch.optim.Adam([W, V], lr=0.005)
    eps = torch.tensor(1e-4, dtype=torch.float32, device=device)

    def train_step():
        """Perform one training step and return total cost."""
        num_rows, num_cols = X.shape
        total_cost = 0.0

        # Block-wise computation
        for i in range(0, num_rows, block_size):
            end_i = min(i + block_size, num_rows)
            W_block = W[i:end_i]

            for j in range(0, num_cols, block_size):
                end_j = min(j + block_size, num_cols)
                H_block = H[:, j:end_j]
                V_block = V[:, j:end_j]

                # Extract block of X (dense)
                A_block = X.to_dense()[i:end_i, j:end_j]
                WH_block = torch.matmul(W_block, H_block + V_block)

                # Reweight zero entries
                mask_block = torch.ones_like(A_block, dtype=torch.float32, device=device)
                mask_block[A_block == 0] = weight
                if weight < 1:
                    WH_block = mask_block * WH_block

                # Compute reconstruction loss
                if distance == 'frobenius':
                    cost_block = torch.sum((A_block - WH_block) ** 2)
                elif distance == 'KL':
                    cost_block = torch.sum(
                        A_block * torch.log(A_block / (WH_block + eps) + eps) - A_block + WH_block
                    )
                else:
                    raise ValueError("Distance must be 'frobenius' or 'KL'.")

                total_cost += cost_block

        # Laplacian regularization
        cos_reg = torch.trace(W.t() @ torch.sparse.mm(L, W))
        total_cost = total_cost + lamb * cos_reg

        # Backpropagation
        optimizer.zero_grad()
        total_cost.backward()
        optimizer.step()

        # Enforce nonnegativity
        W.data = torch.clamp(W.data, min=0)
        V.data = torch.clamp(V.data, min=0)

        return total_cost.item()

    # Training loop
    start = time.perf_counter()
    prev_cost = np.inf
    for i in range(max_iter):
        current_cost = train_step()
        improvement = prev_cost - current_cost
        prev_cost = current_cost

        if i % 50 == 0:
            print(f"Iter {i}: cost={current_cost:.6f}, improvement={improvement:.6f}")
        if improvement < tol or current_cost < 0:
            break

    end = time.perf_counter()
    print(f"Training finished in {end - start:.2f} seconds.")

    # Return learned W only
    return W.detach().cpu().numpy()
        
def fit(X, W, H, V, S=None, P=None, lamb=10, alpha=13, rank=10, tol=1e-3, weight=0.5, max_iter=100, block_size=5000, distance='frobenius', device=None):
    """
    Optimize nonnegative basis matrix W for a single dataset using block-wise updates.

    Parameters
    ----------
    X : array-like or sparse matrix (n_samples, n_features)
        Input data matrix.
    W : ndarray (n_samples, rank)
        Initial basis matrix.
    H : ndarray (rank, n_features)
        Coefficient matrix.
    V : ndarray (rank, n_features)
        Auxiliary matrix.
    S : sparse matrix or None
        Similarity matrix for Laplacian regularization.
    P : ndarray or None
        Constraint matrix for KL divergence.
    lamb : float
        Weight for Laplacian regularization.
    alpha : float
        Weight for KL divergence penalty.
    rank : int
        Rank of factorization.
    tol : float
        Convergence tolerance.
    weight : float
        Weight applied to zero entries in X.
    max_iter : int
        Maximum number of iterations.
    block_size : int
        Block size for memory-efficient computation.
    distance : {'frobenius', 'KL'}
        Distance metric for reconstruction loss.
    device : torch.device or None
        GPU/CPU device.

    Returns
    -------
    dict : learned matrices and objective
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Convert to torch tensors
    W = torch.tensor(W, dtype=torch.float32, requires_grad=True, device=device)
    V = torch.tensor(V, dtype=torch.float32, requires_grad=True, device=device)
    H = torch.tensor(H, dtype=torch.float32, requires_grad=True, device=device)
    eps = 1e-4

    if P is not None:
        P = torch.tensor(P, dtype=torch.float32, device=device)

    # Sparse X
    X = coo_matrix(X)
    X = torch.sparse_coo_tensor(
        indices=torch.tensor(np.vstack((X.row, X.col)), dtype=torch.long, device=device),
        values=torch.tensor(X.data, dtype=torch.float32, device=device),
        size=X.shape
    ).coalesce()

    # Laplacian
    if S is not None:
        if np.max(abs(S)) > 0:
            S = S / np.max(abs(S))
        row_sum = np.array(abs(S).sum(axis=1)).ravel()
        L = diags(row_sum, format='csr') - S
    else:
        raise ValueError("The similarity matrix has not been created")
    
    L = L.tocoo()
    L = torch.sparse_coo_tensor(
        indices=torch.tensor(np.vstack((L.row, L.col)), dtype=torch.long, device=device),
        values=torch.tensor(L.data, dtype=torch.float32, device=device),
        size=L.shape
    ).coalesce()

    # Optimizer
    optimizer = torch.optim.Adam([W, V, H], lr=0.005)

    def train_step():
        total_cost = 0.0
        num_rows, num_cols = X.shape

        for i in range(0, num_rows, block_size):
            end_i = min(i + block_size, num_rows)
            W_block = W[i:end_i]
            
            for j in range(0, num_cols, block_size):
                end_j = min(j + block_size, num_cols)
                H_block = H[:, j:end_j]
                V_block = V[:, j:end_j]

                # Extract block of X (dense)
                A_block = X.to_dense()[i:end_i, j:end_j]
                WH_block = torch.matmul(W_block, (H_block + V_block))

                # Reweight zero entries
                mask_block = torch.ones_like(A_block, dtype=torch.float32, device=device)
                mask_block[A_block == 0] = weight
                
                if weight < 1:
                    WH_block = mask_block * WH_block

                # Compute reconstruction loss
                if distance == 'frobenius':
                    cost_block = torch.sum((A_block - WH_block) ** 2)
                elif distance == 'KL':
                    cost_block = torch.sum(
                        A_block * torch.log(A_block / (WH_block + eps) + eps) - A_block + WH_block
                    )
                else:
                    raise ValueError("Distance must be 'frobenius' or 'KL'.")

                total_cost += cost_block

        # Laplacian regularization
        cos_reg = torch.trace(W.t() @ torch.sparse.mm(L, W))
        total_cost += lamb * cos_reg
        #print('cos_reg:',cos_reg)

        # KL divergence
        if P is not None:
            kl_div = torch.sum(P * torch.log((P + eps) / (H + eps)) - P + H)
            total_cost += alpha * kl_div

        optimizer.zero_grad()
        total_cost.backward()
        optimizer.step()

        # Non-negativity
        W.data = torch.clamp_(W.data, min=0)
        V.data = torch.clamp_(V.data, min=0)
        H.data = torch.clamp_(H.data, min=0)

        return total_cost.item()

    # Training loop
    start = time.perf_counter()
    prev_cost = np.inf
    for i in range(max_iter):
        curr_cost = train_step()
        improvement = prev_cost - curr_cost
        prev_cost = curr_cost
        if i % 50 ==0:
            print(f"Iter {i}: cost={curr_cost:.6f}, improvement={improvement:.6f}")
        if improvement < tol or curr_cost < 0:
            break

    end = time.perf_counter()

    return {
        'W': W.detach().cpu().numpy(),
        'H': H.detach().cpu().numpy(),
        'V': V.detach().cpu().numpy(),
        'obj': prev_cost,
        'time': end - start
    }

def loss(X, W, H, V, S=None, P=None, lamb=10, alpha=13, rank=10, tol=1e-3, weight=1, max_iter=1, block_size=5000, distance='frobenius', device=None):
    """
    Optimize nonnegative basis matrix W for a single dataset using block-wise updates.

    Parameters
    ----------
    X : array-like or sparse matrix (n_samples, n_features)
        Input data matrix.
    W : ndarray (n_samples, rank)
        Initial basis matrix.
    H : ndarray (rank, n_features)
        Coefficient matrix.
    V : ndarray (rank, n_features)
        Auxiliary matrix.
    S : sparse matrix or None
        Similarity matrix for Laplacian regularization.
    P : ndarray or None
        Constraint matrix for KL divergence.
    lamb : float
        Weight for Laplacian regularization.
    alpha : float
        Weight for KL divergence penalty.
    rank : int
        Rank of factorization.
    tol : float
        Convergence tolerance.
    weight : float
        Weight applied to zero entries in X.
    max_iter : int
        Maximum number of iterations.
    block_size : int
        Block size for memory-efficient computation.
    distance : {'frobenius', 'KL'}
        Distance metric for reconstruction loss.
    device : torch.device or None
        GPU/CPU device.

    Returns
    -------
    dict : learned matrices and objective
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Convert to torch tensors
    W = torch.tensor(W, dtype=torch.float32, requires_grad=True, device=device)
    V = torch.tensor(V, dtype=torch.float32, requires_grad=True, device=device)
    H = torch.tensor(H, dtype=torch.float32, requires_grad=True, device=device)
    eps = 1e-4

    if P is not None:
        P = torch.tensor(P, dtype=torch.float32, device=device)

    # Sparse X
    X = coo_matrix(X)
    X = torch.sparse_coo_tensor(
        indices=torch.tensor(np.vstack((X.row, X.col)), dtype=torch.long, device=device),
        values=torch.tensor(X.data, dtype=torch.float32, device=device),
        size=X.shape
    ).coalesce()

    # Laplacian
    if S is not None:
        if np.max(abs(S)) > 0:
            S = S / np.max(abs(S))
        row_sum = np.array(abs(S).sum(axis=1)).ravel()
        L = diags(row_sum, format='csr') - S
    else:
        raise ValueError("The similarity matrix has not been created")
    
    L = L.tocoo()
    L = torch.sparse_coo_tensor(
        indices=torch.tensor(np.vstack((L.row, L.col)), dtype=torch.long, device=device),
        values=torch.tensor(L.data, dtype=torch.float32, device=device),
        size=L.shape
    ).coalesce()

    # Optimizer
    optimizer = torch.optim.Adam([W, V, H], lr=0.005)

    def train_step():
        total_cost = 0.0
        num_rows, num_cols = X.shape

        for i in range(0, num_rows, block_size):
            end_i = min(i + block_size, num_rows)
            W_block = W[i:end_i]
            
            for j in range(0, num_cols, block_size):
                end_j = min(j + block_size, num_cols)
                H_block = H[:, j:end_j]
                V_block = V[:, j:end_j]

                # Extract block of X (dense)
                A_block = X.to_dense()[i:end_i, j:end_j]
                WH_block = torch.matmul(W_block, (H_block + V_block))

                # Reweight zero entries
                mask_block = torch.ones_like(A_block, dtype=torch.float32, device=device)
                mask_block[A_block == 0] = weight
                
                if weight < 1:
                    WH_block = mask_block * WH_block

                # Compute reconstruction loss
                if distance == 'frobenius':
                    cost_block = torch.sum((A_block - WH_block) ** 2)
                elif distance == 'KL':
                    cost_block = torch.sum(
                        A_block * torch.log(A_block / (WH_block + eps) + eps) - A_block + WH_block
                    )
                else:
                    raise ValueError("Distance must be 'frobenius' or 'KL'.")

                total_cost += cost_block
        print('reconstraction loss :',total_cost)
        # Laplacian regularization
        cos_reg = torch.trace(W.t() @ torch.sparse.mm(L, W))
        print('Laplacian loss :',cos_reg)

        # KL divergence
        if P is not None:
            kl_div = torch.sum(P * torch.log((P + eps) / (H + eps)) - P + H)
            print('kl loss :',kl_div)

        optimizer.zero_grad()
        total_cost.backward()
        optimizer.step()

        # Non-negativity
        W.data = torch.clamp_(W.data, min=0)
        V.data = torch.clamp_(V.data, min=0)
        H.data = torch.clamp_(H.data, min=0)

        return total_cost.item(), cos_reg.item(), kl_div.item()
    
    

    for i in range(max_iter):
        reconstraction_loss,Laplacian_loss, kl_loss  = train_step()
        percent = (Laplacian_loss/reconstraction_loss)*100

    return percent, kl_loss    



def predict_parameter(model, kl_loss, percent):

    """
        Determine the parameter category based on kl_loss and percent.

    Returns:
        predicted_parameter: Predicted parameter combination.
        probability_class1: Probability of belonging to the lambda=10, gamma=0.6 class.
    """

    if kl_loss <= 0:
        raise ValueError("kl_loss must be greater than 0.")

    new_data = pd.DataFrame({
        "log10_kl_loss": [np.log10(kl_loss)],
        "percent": [percent]
    })

    probability_class1 = model.predict_proba(new_data)[0, 1]
    predicted_label = int(probability_class1 >= 0.5)

    label_mapping = {
        0: "lambda=5, gamma=0.3",
        1: "lambda=10, gamma=0.6"
    }

    return {
        "predicted_parameter": label_mapping[predicted_label],
        "probability_lambda10_gamma06": probability_class1,
        "probability_lambda5_gamma03": 1.0 - probability_class1
    }
