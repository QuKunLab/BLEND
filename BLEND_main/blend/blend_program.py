import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# =============== 1) Convert H matrix into DataFrame =================
def make_factor_df_from_matrix(H: np.ndarray, gene_names, factor_prefix="data1_", pad=2) -> pd.DataFrame:
    """
    Convert a factor-gene matrix (H) into a DataFrame with labeled rows.

    Parameters
    ----------
    H : np.ndarray, shape (n_factors, n_genes)
        Factor-gene matrix where rows correspond to factors and columns to genes.
    gene_names : list or array-like
        Gene names corresponding to columns of H.
    factor_prefix : str, optional (default: "data1_")
        Prefix for factor row names.
    pad : int, optional (default: 2)
        Zero-padding length for factor indices.

    Returns
    -------
    DataFrame
        Factors as rows, gene names as columns, with row names like
        'data1_01', 'data1_02', ...
    """
    if len(gene_names) != H.shape[1]:
        raise ValueError("len(gene_names) must match H.shape[1].")
    df = pd.DataFrame(H, columns=list(gene_names))
    n_factors = df.shape[0]
    idx = [f"{factor_prefix}{str(i+1).zfill(pad)}" for i in range(n_factors)]
    df.index = idx
    return df


def set_factor_index(df: pd.DataFrame, factor_prefix="data1_", pad=2) -> pd.DataFrame:
    """
    Reset row names of an existing H DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame where rows are factors and columns are genes.
    factor_prefix : str, optional (default: "data1_")
        Prefix for factor row names.
    pad : int, optional (default: 2)
        Zero-padding length for factor indices.

    Returns
    -------
    DataFrame
        Same DataFrame with updated row names:
        first row -> data1_01, second row -> data1_02, etc.
    """
    n_factors = df.shape[0]
    idx = [f"{factor_prefix}{str(i+1).zfill(pad)}" for i in range(n_factors)]
    df = df.copy()
    df.index = idx
    return df


# =============== 2) Select top-N genes per factor =================
def select_gene_sets_from_H(H_df: pd.DataFrame, top_n=200, by_abs=False) -> dict:
    """
    Select top genes for each factor based on H values.

    Parameters
    ----------
    H_df : pd.DataFrame
        DataFrame of shape (n_factors, n_genes), rows are factors, columns are genes.
    top_n : int, optional (default: 200)
        Number of top genes to select per factor.
    by_abs : bool, optional (default: False)
        If True, rank genes by absolute values instead of raw values.

    Returns
    -------
    dict
        Dictionary mapping factor_name -> set(top genes).
    """
    if H_df.columns.duplicated().any():
        # Drop duplicate gene names if any
        H_df = H_df.loc[:, ~H_df.columns.duplicated()].copy()

    gene_sets = {}
    for factor_name, row in H_df.iterrows():
        vals = row
        if by_abs:
            top_genes = vals.abs().nlargest(top_n).index
        else:
            top_genes = vals.nlargest(top_n).index
        gene_sets[factor_name] = set(top_genes)
    return gene_sets


# =============== 3) Compute Jaccard similarity matrix =================
def jaccard_matrix(gene_sets: dict) -> pd.DataFrame:
    """
    Compute Jaccard similarity matrix between gene sets.

    Parameters
    ----------
    gene_sets : dict
        Mapping from factor_name to a set of selected genes.

    Returns
    -------
    DataFrame
        Square Jaccard similarity matrix (n_factors x n_factors).
    """
    names = list(gene_sets.keys())
    n = len(names)
    mat = np.zeros((n, n), dtype=float)
    for i in range(n):
        A = gene_sets[names[i]]
        for j in range(n):
            B = gene_sets[names[j]]
            inter = len(A & B)
            union = len(A | B)
            mat[i, j] = inter / union if union > 0 else 0.0
    return pd.DataFrame(mat, index=names, columns=names)


# =============== 4) Visualization: hierarchical clustering heatmap =================
def plot_jaccard_clustermap(jaccard_df: pd.DataFrame, figsize=(16, 16),
                            vmin=0, vmax=1, cmap="Reds",
                            method="average", metric="euclidean",
                            cbar_pos=(1.02, 0.4, 0.02, 0.2), dendrogram_ratio=0.08):
    """
    Visualize Jaccard similarity matrix with hierarchical clustering.

    Parameters
    ----------
    jaccard_df : pd.DataFrame
        Jaccard similarity matrix (n_factors x n_factors).
    figsize : tuple, optional (default: (16,16))
        Figure size for the heatmap.
    vmin, vmax : float, optional
        Min and max values for colormap scaling.
    cmap : str, optional (default: "Reds")
        Colormap for heatmap.
    method : str, optional (default: "average")
        Linkage method for hierarchical clustering.
    metric : str, optional (default: "euclidean")
        Distance metric for clustering.
    cbar_pos : tuple, optional
        Colorbar position in the figure.
    dendrogram_ratio : float, optional
        Relative size of dendrograms.

    Returns
    -------
    ClusterGrid
        Seaborn ClusterGrid object.
    """
    g = sns.clustermap(
        jaccard_df, row_cluster=True, col_cluster=True,
        metric=metric, method=method,
        vmin=vmin, vmax=vmax, cmap=cmap,
        linewidths=0.05, figsize=figsize,
        dendrogram_ratio=dendrogram_ratio,
        cbar_pos=cbar_pos
    )
    g.ax_heatmap.grid(False)
    plt.show()
    return g
