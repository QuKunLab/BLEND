import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.neighbors import kneighbors_graph

def KNN_classify(
    adata,
    leiden_col: str = "leiden",
    spatial_key: str = "spatial",
    K: int = 10,
    default_label=np.nan,
    n_jobs: int = -1,
):
    """
    Fill missing cluster labels in `obs[leiden_col]` using majority vote of
    spatial KNN neighbors. The result is updated in place and standardized
    as string categories ("0", "1", "2", ...).

    Parameters
    ----------
    adata : AnnData
        AnnData object containing `obs[leiden_col]` (with missing entries) and
        `obsm[spatial_key]` (spatial coordinates).
    leiden_col : str
        The column in `adata.obs` to fill.
    spatial_key : str
        The key in `adata.obsm` with spatial coordinates (e.g., "spatial").
    K : int
        Number of neighbors for KNN.
    default_label : Any
        Fallback value when no valid neighbor labels are found (default: NaN).
    n_jobs : int
        Number of parallel jobs for sklearn neighbor graph construction.
        Default -1 uses all available cores.

    Returns
    -------
    None
        The filled column is written back to `adata.obs[leiden_col]` as a
        pandas Categorical with categories as strings ("0", "1", "2", ...).
    """
    if leiden_col not in adata.obs.columns:
        raise ValueError(f"'{leiden_col}' not found in adata.obs.")

    if spatial_key not in adata.obsm:
        raise ValueError(f"adata.obsm['{spatial_key}'] not found.")

    labels = adata.obs[leiden_col].to_numpy(copy=True)
    missing_mask = pd.isna(labels)

    if not missing_mask.any():
        # No missing values: just normalize to categorical strings
        non_na = pd.Series(labels).dropna().astype(str)
        cats = sorted(non_na.unique(), key=lambda x: (len(x), x))
        adata.obs[leiden_col] = pd.Categorical(pd.Series(labels).astype(str), categories=cats)
        return

    # --------- Build spatial-only KNN graph (sparse) ---------
    S = np.asarray(adata.obsm[spatial_key], dtype=float)
    G = kneighbors_graph(
        S, n_neighbors=int(K), mode="connectivity", include_self=False, n_jobs=n_jobs
    ).tolil()

    ser = pd.Series(labels, index=adata.obs_names)
    rows = np.where(missing_mask)[0]

    for r in rows:
        neigh = G.rows[r]
        if not neigh:
            labels[r] = default_label
            continue

        neigh_labels = ser.iloc[neigh].dropna()
        if len(neigh_labels) == 0:
            labels[r] = default_label
            continue

        # Majority vote among neighbors
        labels[r] = neigh_labels.mode(dropna=True).iloc[0]

    # --------- Standardize to string categories ---------
    labels = pd.Series(labels, index=adata.obs_names)

    # Convert non-missing values to int -> str (avoids "1.0")
    labels_non_na_str = labels.dropna().astype(float).astype(int).astype(str)

    # Align back to full index
    labels = labels_non_na_str.reindex(adata.obs_names).astype(str)

    # Create consistent category order
    cats = sorted(labels_non_na_str.unique(), key=lambda x: (len(x), x))
    adata.obs[leiden_col] = pd.Categorical(labels, categories=cats)
