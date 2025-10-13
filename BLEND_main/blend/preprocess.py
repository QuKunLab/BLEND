import numpy as np                   
import scipy.sparse as sp           
from scipy.sparse import csr_matrix  
import scanpy as sc                  
import anndata                      

def select_common_genes(adata, sc_adata):
    """
    Filter two AnnData objects to retain only the common genes.

    Parameters
    ----------
    adata : AnnData
        Spatial data (AnnData object).
    sc_adata : AnnData
        Single-cell data (AnnData object).

    Returns
    -------
    adata : AnnData
        Filtered AnnData object with only common genes.
    sc_adata : AnnData
        Filtered AnnData object with only common genes.
    """

    # Ensure unique gene names
    adata.var_names_make_unique()
    sc_adata.var_names_make_unique()

    # Find common genes
    common_genes = adata.var_names.intersection(sc_adata.var_names)
    print(f"Number of common genes: {len(common_genes)}")

    # Subset both datasets by common genes
    adata = adata[:, common_genes]
    sc_adata = sc_adata[:, common_genes]

    # Print shapes for confirmation
    print(f"adata shape after filtering: {adata.shape}")
    print(f"sc_adata shape after filtering: {sc_adata.shape}")

    return adata, sc_adata

def minmax_scale_cells(adata):
    """
    Apply Min-Max scaling to an AnnData object so that 
    each cell (row) is scaled to the [0, 1] range.
    
    Assumes adata.X is a sparse matrix.
    
    Parameters
    ----------
    adata : AnnData
        Input AnnData object with gene expression matrix in adata.X.
    
    Returns
    -------
    adata : AnnData
        AnnData object with scaled expression values in adata.X (CSR sparse matrix).
    """
    # Ensure X is in CSR format
    X = adata.X
    if not isinstance(X, csr_matrix):
        X = X.tocsr()

    # Compute row-wise min and max
    min_values = X.min(axis=1).toarray().ravel()
    max_values = X.max(axis=1).toarray().ravel()

    # Avoid division by zero
    range_values = max_values - min_values
    range_values[range_values == 0] = 1

    # Apply Min-Max scaling
    scaled_X = (X - min_values[:, None]) / range_values[:, None]

    # Save back to AnnData in CSR format
    adata.X = csr_matrix(scaled_X)

    return adata

def is_counts_matrix(adata):
    """
    Check whether the gene expression matrix in an AnnData object 
    is a raw counts matrix (i.e., all values are integers).

    This is useful for avoiding redundant normalization or log-transformation.

    Parameters
    ----------
    adata : AnnData
        Input AnnData object with expression matrix in adata.X.

    Returns
    -------
    bool
        True if adata.X appears to be a counts matrix (all values are integers),
        False otherwise.
    """
    X = adata.X

    # Sparse matrix case
    if sp.issparse(X):
        return np.all(np.mod(X.data, 1) == 0)

    # Dense integer matrix
    if np.issubdtype(X.dtype, np.integer):
        return True

    # Dense float matrix, check if all values are (close to) integers
    if np.issubdtype(X.dtype, np.floating):
        return np.allclose(X, np.round(X))

    return False


def data_process(adata, min_genes=1, min_cells=50, tar_sum = None):
    """
    Basic preprocessing for two AnnData objects:
    1. Filter out low-quality cells and genes.
    2. Normalize and log-transform if the data is raw counts.

    Parameters
    ----------
    adata : AnnData
        Spatial or Single-cell AnnData object.
    min_genes : int, optional (default: 1)
        Minimum number of genes required for a cell to be kept.
    min_cells : int, optional (default: 50)
        Minimum number of cells required for a gene to be kept.

    Returns
    -------
    adata : AnnData
        Processed AnnData object.
    sc_adata : AnnData
        Processed single-cell AnnData object.
    """

    # Filter low-quality cells and genes
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)

    # Normalize and log-transform if data is counts
    if is_counts_matrix(adata):
        print("adata: Normalize and log1p")
        sc.pp.normalize_total(adata, target_sum = tar_sum)
        sc.pp.log1p(adata)

    return adata