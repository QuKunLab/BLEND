import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc


def blend_leiden_cluster(
    W,
    adata_ref,
    resolution: float = 0.8,
    n_neighbors: int = 30,
    n_pcs: int = 30,
    seed: int = None,
    log1p: bool = True,
    svd_solver: str = "arpack",
):
    """
    Perform Leiden clustering based on a given representation matrix.

    Parameters
    ----------
    W : np.ndarray
        Representation matrix of shape (cells x features).
    adata_ref : AnnData
        Reference AnnData object used to copy metadata such as spatial coordinates and cell indices.
    resolution : float, optional (default: 0.8)
        Resolution parameter for the Leiden algorithm.
    n_neighbors : int, optional (default: 30)
        Number of neighbors for kNN graph construction.
    n_pcs : int, optional (default: 30)
        Number of principal components for dimensionality reduction.
    seed : int, optional (default: 110)
        Random seed for reproducibility.
    log1p : bool, optional (default: True)
        Whether to apply log1p transformation to the data.
    svd_solver : str, optional (default: "arpack")
        SVD solver used in PCA.

    Returns
    -------
    cluster : pandas.Series
        A Series containing Leiden cluster assignments for each cell.
    """
    adata = ad.AnnData(W.copy())
    if "spatial" in adata_ref.obsm:
        adata.obsm["spatial"] = adata_ref.obsm["spatial"].copy()
    adata.obs.index = adata_ref.obs.index.copy()

    if log1p:
        sc.pp.log1p(adata)
    sc.tl.pca(adata, svd_solver=svd_solver, random_state=seed)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs, random_state=seed)
    sc.tl.leiden(adata, resolution=resolution, random_state=seed)
    sc.tl.umap(adata)

    return adata.obs["leiden"], adata.obsm['X_umap']


def resolve_colors(adata, key):
    """
    Resolve valid color keys for plotting with Scanpy.

    Parameters
    ----------
    adata : AnnData
        Input AnnData object.
    key : str, list, or tuple
        Column(s) in `adata.obs` to use for coloring.

    Returns
    -------
    colors : list
        Valid color keys. Includes 'leiden' first if available.
    """
    if key is None:
        keys = []
    elif isinstance(key, (list, tuple)):
        keys = list(key)
    else:
        keys = [key]

    keys = [k for k in keys if isinstance(k, str) and k in adata.obs.columns]

    colors = []
    if "blend" in adata.obs.columns:
        colors.append("blend")
    colors.extend([k for k in keys if k != "blend"])

    if not colors:
        raise ValueError(
            "No valid columns found for coloring. "
            "Neither 'blend' nor the provided keys exist in adata.obs."
        )
    return colors

def plot_umap(
    adata,
    key="annotation",
    palette=None,
    size: float = 10,
    legend_loc: str = "right margin",
    seed: int = 110,
    show: bool = True,
    save: str = None,
):
    """
    Plot a UMAP embedding of the given AnnData object.

    If UMAP has not been computed yet, it will be calculated automatically.

    Parameters
    ----------
    adata : AnnData
        Input AnnData object.
    key : str or list, optional (default: "annotation")
        Column(s) in `adata.obs` used for coloring.
    palette : list, optional
        List of colors to use for plotting.
    size : float, optional (default: 10)
        Point size.
    legend_loc : str, optional (default: "right margin")
        Location of the legend.
    seed : int, optional (default: 110)
        Random seed for reproducibility.
    show : bool, optional (default: True)
        Whether to display the plot.
    save : str, optional (default: None)
        File extension or filename to save the plot (e.g., ".png").

    Returns
    -------
    None
    """
    if "X_umap" not in adata.obsm:
        if "neighbors" not in adata.uns:
            sc.pp.neighbors(adata, random_state=seed)
        sc.tl.umap(adata, random_state=seed)

    colors = resolve_colors(adata, key)
    sc.pl.umap(
        adata,
        color=colors,
        palette=palette,
        legend_loc=legend_loc,
        size=size,
        show=show,
        save=save,
    )


def plot_spatial(
    adata,
    key="annotation",
    palette=None,
    size: float = 10,
    legend_loc: str = "right margin",
    show: bool = True,
    save: str = None,
):
    """
    Plot spatial scatter plots using `obsm['spatial']`.

    Parameters
    ----------
    adata : AnnData
        Input AnnData object with spatial coordinates.
    key : str or list, optional (default: "annotation")
        Column(s) in `adata.obs` used for coloring.
    palette : list, optional
        List of colors to use for plotting.
    size : float, optional (default: 10)
        Point size.
    legend_loc : str, optional (default: "right margin")
        Location of the legend.
    show : bool, optional (default: True)
        Whether to display the plot.
    save : str, optional (default: None)
        File extension or filename to save the plot (e.g., ".png").

    Returns
    -------
    None
    """
    if "spatial" not in adata.obsm:
        raise ValueError("`adata.obsm['spatial']` not found. Cannot plot spatial data.")

    colors = resolve_colors(adata, key)
    sc.pl.embedding(
        adata,
        basis="spatial",
        color=colors,
        palette=palette,
        legend_loc=legend_loc,
        size=size,
        show=show,
        save=save,
    )


'''
import scanpy as sc

def plot_umap(
    adata,
    key: str = "annotation",
    palette=None,
    size: float = 10,
    legend_loc: str = "right margin",
    seed: int = 110,
    show: bool = True,
    save: str = None,
):
    """
    Plot a UMAP embedding of the given AnnData object.

    If UMAP has not been computed yet, it will be calculated automatically.
    This function does NOT overwrite UMAP coordinates with spatial coordinates.

    Parameters
    ----------
    adata : AnnData
        Input AnnData object.
    key : str, optional (default: "annotation")
        Column in `adata.obs` to use for coloring the plot. If it does not exist,
        only 'leiden' will be plotted (if available).
    palette : list, optional
        List of colors to use for plotting.
    size : float, optional (default: 10)
        Point size.
    legend_loc : str, optional (default: "right margin")
        Location of the legend.
    seed : int, optional (default: 110)
        Random seed for reproducibility.
    show : bool, optional (default: True)
        Whether to display the plot.
    save : str, optional (default: None)
        File extension or filename to save the plot (e.g., ".png").

    Returns
    -------
    None
    """
    # Ensure neighbors and UMAP exist
    if "X_umap" not in adata.obsm:
        if "neighbors" not in adata.uns:
            sc.pp.neighbors(adata, random_state=seed)
        sc.tl.umap(adata, random_state=seed)

    # Choose valid color keys
    colors = ["leiden"] if "leiden" in adata.obs.columns else []
    if key in adata.obs.columns and key != "leiden":
        colors.append(key)

    sc.pl.umap(
        adata,
        color=colors,
        palette=palette,
        legend_loc=legend_loc,
        size=size,
        show=show,
        save=save,
    )


def plot_spatial(
    adata,
    key: str = "annotation",
    palette=None,
    size: float = 10,
    legend_loc: str = "right margin",
    show: bool = True,
    save: str = None,
):
    """
    Plot a spatial scatter plot using `obsm['spatial']` as coordinates.

    This function does NOT modify UMAP coordinates.

    Parameters
    ----------
    adata : AnnData
        Input AnnData object with spatial coordinates in `obsm['spatial']`.
    key : str, optional (default: "annotation")
        Column in `adata.obs` to use for coloring the plot. If it does not exist,
        only 'leiden' will be plotted (if available).
    palette : list, optional
        List of colors to use for plotting.
    size : float, optional (default: 10)
        Point size.
    legend_loc : str, optional (default: "right margin")
        Location of the legend.
    show : bool, optional (default: True)
        Whether to display the plot.
    save : str, optional (default: None)
        File extension or filename to save the plot (e.g., ".png").

    Returns
    -------
    None
    """
    if "spatial" not in adata.obsm:
        raise ValueError("`adata.obsm['spatial']` not found. Cannot plot spatial data.")

    # Choose valid color keys
    colors = ["leiden"] if "leiden" in adata.obs.columns else []
    if key in adata.obs.columns and key != "leiden":
        colors.append(key)

    sc.pl.embedding(
        adata,
        basis="spatial",
        color=colors,
        palette=palette,
        legend_loc=legend_loc,
        size=size,
        show=show,
        save=save,
    )

'''