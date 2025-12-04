"""
D-Optimal Design of Experiments for Chemical Complexes with Ligand Variants.

This module generates a D-optimal experimental design for chemical complexes defined by
hierarchical categorical factors (Element, Oxidation State, Coordination Number, Geometry)
and compositional factors (Ligand Types and Subtypes).

It handles the combinatorial explosion of ligand arrangements by treating them as
multisets (counts) rather than ordered positions, suitable for coordination chemistry
where ligand order is often chemically equivalent or fluxional.

Inputs:
    - Chemical constraints: Elements, Oxidation states, Coordination numbers (CN), Geometries.
    - Ligand definitions: Ligand types (scaffolds) and Subtypes (electronic properties).
    - Constraints: Number of ligands per CN, forbidden combinations.

Outputs:
    - A pandas DataFrame containing the selected subset of candidate complexes.
    - Columns include categorical factors and numeric counts for ligand types/subtypes.
    - A design matrix (X) suitable for regression modeling.

Usage:
    1. Define chemical space dictionaries (elements, coordination, etc.).
    2. Call `expand_candidates_with_variants` to generate the full candidate pool.
    3. Call `d_optimal_greedy` to select the most informative subset of runs.
"""

import itertools
import random
from collections import Counter

import numpy as np
import pandas as pd

def multisets_of_size(items, k):
    """
    Generate all multisets of size k from items.

    Parameters
    ----------
    items : iterable
        The items to choose from.
    k : int
        The size of the multiset.

    Returns
    -------
    list
        List of tuples representing the multisets.
    """
    return list(itertools.combinations_with_replacement(items, k))

def expand_candidates_with_variants(
    elements,
    coordination,
    ligand_types,
    ligand_subtypes=['W', 'N', 'D'],
    include_variant_counts=False,
    forbid_fn=None
):
    """
    Generate a DataFrame of candidate chemical complexes with ligand variants.

    Parameters
    ----------
    elements : dict
        Dictionary mapping element symbols (str) to lists of oxidation states (int).
    coordination : dict
        Dictionary mapping coordination numbers (int) to lists of geometry names (str).
    ligand_types : list of str
        List of ligand type identifiers (e.g., ['L1', 'L2']).
    ligand_subtypes : list of str, optional
        List of ligand subtype identifiers (e.g., ['W', 'N', 'D']). 
        Default is ['W', 'N', 'D'].
    include_variant_counts : bool, optional
        If True, include columns for counts of specific (type, subtype) combinations.
        Default is False.
    forbid_fn : callable, optional
        A function that takes a candidate dictionary and returns True if the 
        candidate should be excluded. Default is None.

    Returns
    -------
    pd.DataFrame
        DataFrame containing all valid candidate complexes.
    """
    # build full list of ligand variants as strings like 'L1|W'
    variants = []
    variant_to_components = {}
    for t in ligand_types:
        for s in ligand_subtypes:
            key = f"{t}|{s}"
            variants.append(key)
            variant_to_components[key] = {'type': t, 'prop': s}

    rows = []
    for elem, ox_list in elements.items():
        for ox in ox_list:
            for cn, geoms in coordination.items():
                # generate combinations (multisets) of variants sized m
                ligand_multisets = multisets_of_size(variants, cn)
                for geom in geoms:
                    for lig_multi in ligand_multisets:
                        counts_variant = Counter(lig_multi)
                        counts_type = Counter()
                        counts_prop = Counter()
                        # aggregate
                        for var_key, ct in counts_variant.items():
                            comp = variant_to_components[var_key]
                            counts_type[comp['type']] += ct
                            counts_prop[comp['prop']] += ct
                        candidate = {
                            'Element': elem,
                            'Ox': ox,
                            'CN': cn,
                            'Geometry': geom,
                            'Ligand_multiset_variants': tuple(lig_multi),
                        }
                        # counts per type (numeric)
                        for t in ligand_types:
                            candidate[f'count_type_{t}'] = counts_type.get(t, 0)
                        # counts per property
                        for p in ligand_subtypes:
                            candidate[f'count_prop_{p}'] = counts_prop.get(p, 0)
                        # optional: counts per variant
                        if include_variant_counts:
                            for v in variants:
                                # replace '|' in column names to avoid confusion
                                col = f"count_var_{v.replace('|','_')}"
                                candidate[col] = counts_variant.get(v, 0)
                        if forbid_fn and forbid_fn(candidate):
                            continue
                        rows.append(candidate)
    df = pd.DataFrame(rows)
    df.insert(0, 'cand_id', range(1, len(df)+1))
    return df

# ---------------------------
# Build model matrix (one-hot for categorical top-levels, numeric counts left as-is)
def build_model_matrix(df, model_terms=None, ligand_types=None, ligand_subtypes=None, include_variant_counts=False):
    """
    Construct the design matrix X from the candidate DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing candidate experiments.
    model_terms : list of str, optional
        List of terms to include (columns in df, or top-level categorical names).
        If None, defaults to ['Element', 'Ox', 'CN', 'Geometry'] + count_type_* + count_prop_*.
    ligand_types : list of str, optional
        List of ligand types to identify count columns if model_terms is None.
    ligand_subtypes : list of str, optional
        List of ligand subtypes to identify count columns if model_terms is None.
    include_variant_counts : bool, optional
        Whether to include any variant count columns already present. Default is False.

    Returns
    -------
    tuple
        (X, design) where X is the numpy array model matrix and design is the DataFrame version.
    """
    if model_terms is None:
        terms = ['Element','Ox','CN','Geometry']
        if ligand_types is None:
            ligand_types = sorted({c.replace('count_type_','') for c in df.columns if c.startswith('count_type_')})
        if ligand_subtypes is None:
            ligand_subtypes = sorted({c.replace('count_prop_','') for c in df.columns if c.startswith('count_prop_')})
        terms += [f'count_type_{t}' for t in ligand_types]
        terms += [f'count_prop_{p}' for p in ligand_subtypes]
        if include_variant_counts:
            variant_cols = [c for c in df.columns if c.startswith('count_var_')]
            terms += variant_cols
    else:
        terms = model_terms

    design = pd.DataFrame(index=df.index)
    for c in terms:
        if c in df.columns and df[c].dtype == object and not c.startswith('count_'):
            dummies = pd.get_dummies(df[c].astype(str), prefix=c, drop_first=True)
            design = pd.concat([design, dummies], axis=1)
        elif c in df.columns:
            design[c] = df[c]
        else:
            # unknown term: ignore (or raise)
            pass
    design.insert(0, 'Intercept', 1.0)
    X = design.to_numpy(dtype=float)
    return X, design

# ---------------------------
# D-optimal greedy exchange (same core idea as before)
def logdet(XtX):
    """
    Calculate the log-determinant of a matrix using SVD.

    Parameters
    ----------
    XtX : np.ndarray
        The matrix to calculate the log-determinant of.

    Returns
    -------
    float
        The log-determinant, or -inf if the matrix is singular.
    """
    try:
        s = np.linalg.svd(XtX, compute_uv=False)
        s = s[s > 1e-12]
        if len(s)==0:
            return -np.inf
        return float(np.sum(np.log(s)))
    except np.linalg.LinAlgError:
        return -np.inf

def d_optimal_greedy(
    df, 
    n_points, 
    model_terms=None, 
    ligand_types=None, 
    ligand_subtypes=None, 
    include_variant_counts=False, 
    max_iter=2000, 
    seed=123,
    previous_design=None
):
    """
    Select a D-optimal subset of runs using a greedy exchange algorithm.

    Parameters
    ----------
    df : pd.DataFrame
        The candidate DataFrame.
    n_points : int
        Number of structures to select.
    model_terms : list of str, optional
        List of model terms. If None, uses default terms.
    ligand_types : list of str, optional
        List of ligand types.
    ligand_subtypes : list of str, optional
        List of ligand subtypes.
    include_variant_counts : bool, optional
        Whether to include variant counts in the model.
    max_iter : int, optional
        Maximum number of iterations for the greedy algorithm. Default is 2000.
    seed : int, optional
        Random seed. Default is 123.
    previous_design : pd.DataFrame, optional
        A DataFrame of a previously generated design. Must contain columns 'success' (bool)
        and 'evaluated' (bool).
        - If 'evaluated' is True and 'success' is True: The point is kept and locked.
        - If 'evaluated' is True and 'success' is False: The point is discarded and replaced.
        - If 'evaluated' is False: The point is treated as a placeholder and can be swapped out.

    Returns
    -------
    tuple
        (selected_indices, X_final, design_final)
    """
    X_all, design_all = build_model_matrix(df, model_terms, ligand_types, ligand_subtypes, include_variant_counts)
    N = len(df)
    rng = random.Random(seed)

    # Initialize selection mask
    selected_mask = np.zeros(N, dtype=bool)
    locked_mask = np.zeros(N, dtype=bool)  # Points that MUST be kept

    if previous_design is not None:
        # Map previous design rows back to candidate df indices
        # We assume 'cand_id' or index alignment is preserved. 
        # Ideally, merge on unique features, but here we assume cand_id matches.
        if 'cand_id' in previous_design.columns and 'cand_id' in df.columns:
             # Create a map from cand_id to df index
            cand_id_to_idx = {cid: idx for idx, cid in zip(df.index, df['cand_id'])}
            
            for _, row in previous_design.iterrows():
                cid = row.get('cand_id')
                if cid in cand_id_to_idx:
                    idx = cand_id_to_idx[cid]
                    evaluated = row.get('evaluated', False)
                    success = row.get('success', False)

                    if evaluated and success:
                        # Keep and lock
                        selected_mask[idx] = True
                        locked_mask[idx] = True
                    elif evaluated and not success:
                        # Discard (do not select, do not lock) - effectively "replace"
                        selected_mask[idx] = False
                    elif not evaluated:
                        # Keep initially, but allow swapping (do not lock)
                        selected_mask[idx] = True
        else:
            raise ValueError("previous_design provided but 'cand_id' column missing in design or candidates.")

    # Fill remaining spots to reach n_points
    current_count = np.sum(selected_mask)
    if current_count < n_points:
        available_indices = [i for i in range(N) if not selected_mask[i]]
        needed = n_points - current_count
        if len(available_indices) >= needed:
            new_picks = rng.sample(available_indices, needed)
            selected_mask[new_picks] = True
        else:
            # Take all available if not enough
            selected_mask[available_indices] = True
    elif current_count > n_points:
        # If we have too many locked/pre-selected points, we might have to trim non-locked ones
        # If all are locked, we can't reduce size.
        indices = np.where(selected_mask & ~locked_mask)[0]
        to_remove = current_count - n_points
        if len(indices) >= to_remove:
            remove_idx = rng.sample(list(indices), to_remove)
            selected_mask[remove_idx] = False
        else:
            # Warning: n_points is smaller than the number of locked successful runs.
            # We keep all locked runs.
            pass

    XtX = X_all[selected_mask,:].T.dot(X_all[selected_mask,:])
    best_score = logdet(XtX)
    improved = True
    it = 0
    
    while improved and it < max_iter:
        improved = False
        it += 1
        
        # Only swap points that are NOT locked
        sel_indices = np.where(selected_mask & ~locked_mask)[0]
        not_sel_indices = np.where(~selected_mask)[0]
        
        rng.shuffle(sel_indices)
        rng.shuffle(not_sel_indices)
        
        for i in sel_indices:
            for j in not_sel_indices:
                new_mask = selected_mask.copy()
                new_mask[i] = False
                new_mask[j] = True
                XtXp = X_all[new_mask,:].T.dot(X_all[new_mask,:])
                score = logdet(XtXp)
                if score > best_score + 1e-8:
                    best_score = score
                    selected_mask = new_mask
                    improved = True
                    break
            if improved:
                break
                
    final_indices = list(np.where(selected_mask)[0])
    df_indices = df.index[final_indices].tolist()
    X_final, design_final = build_model_matrix(df.loc[df_indices].reset_index(drop=True), model_terms, ligand_types, ligand_subtypes, include_variant_counts)
    return df_indices, X_final, design_final

# ---------------------------
# Continuous Relaxation (Gradient-Based D-Optimal)
def d_optimal_convex(
    df, 
    n_points, 
    model_terms=None, 
    ligand_types=None, 
    ligand_subtypes=None, 
    include_variant_counts=False, 
    max_iter=1000, 
    tol=1e-5
):
    """
    Select a D-optimal subset using convex optimization (Titterington's Algorithm).
    
    This is a "gradient-based" approach that relaxes the discrete problem to a continuous one.
    It assigns a weight w_i to each candidate and iteratively updates weights using the 
    gradient of the determinant. It is deterministic and often finds better optima than greedy search.

    Parameters
    ----------
    df : pd.DataFrame
        Candidate DataFrame.
    n_points : int
        Number of points to select (used for rounding the continuous weights).
    max_iter : int
        Maximum iterations.
    tol : float
        Convergence tolerance for weights.

    Returns
    -------
    tuple
        (selected_indices, X_final, design_final)
    """
    X, design = build_model_matrix(df, model_terms, ligand_types, ligand_subtypes, include_variant_counts)
    N, p = X.shape
    
    # Initialize weights uniformly
    w = np.ones(N) / N
    
    for it in range(max_iter):
        # Compute Information Matrix M = X.T * W * X
        # Efficient way: (X * w[:, None]).T @ X
        WX = X * w[:, np.newaxis]
        M = WX.T @ X
        
        try:
            M_inv = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            # If singular, add small regularization or perturb weights
            M_inv = np.linalg.pinv(M)

        # Compute variance of prediction d_i = x_i.T * M_inv * x_i
        # Vectorized: sum((X @ M_inv) * X, axis=1)
        d = np.sum((X @ M_inv) * X, axis=1)
        
        # Titterington's multiplicative update (gradient ascent step)
        # w_new = w_old * (d_i / p)
        w_new = w * (d / p)
        
        # Check convergence
        if np.max(np.abs(w_new - w)) < tol:
            w = w_new
            break
        w = w_new

    # Select top n_points with highest weights
    # (This is a simple rounding strategy; more complex ones exist)
    top_indices = np.argsort(w)[-n_points:]
    top_indices = sorted(top_indices)
    
    df_indices = df.index[top_indices].tolist()
    X_final, design_final = build_model_matrix(df.loc[df_indices].reset_index(drop=True), model_terms, ligand_types, ligand_subtypes, include_variant_counts)
    
    return df_indices, X_final, design_final

# ---------------------------
# Space-Filling Design (k-Means Clustering)
def space_filling_kmeans(
    df, 
    n_points, 
    model_terms=None, 
    ligand_types=None, 
    ligand_subtypes=None, 
    include_variant_counts=False,
    seed=123
):
    """
    Select a subset of points that are "spread evenly" across the design space using k-Means.
    
    This uses Lloyd's algorithm (gradient-based on cluster distortion) to find n_points centroids,
    then selects the actual candidate closest to each centroid. This is ideal for coverage 
    rather than parameter estimation.

    Returns
    -------
    tuple
        (selected_indices, X_final, design_final)
    """
    from sklearn.cluster import KMeans
    
    X, design = build_model_matrix(df, model_terms, ligand_types, ligand_subtypes, include_variant_counts)
    
    # Normalize X to ensure equal weighting of factors (optional but recommended for distance-based methods)
    # Here we use raw X (one-hot + counts) which is usually acceptable for this domain.
    
    kmeans = KMeans(n_clusters=n_points, random_state=seed, n_init=10)
    kmeans.fit(X)
    
    # Find closest candidate to each centroid
    selected_indices = []
    centers = kmeans.cluster_centers_
    
    # For each center, find the nearest neighbor in X
    # We can use cdist
    from scipy.spatial.distance import cdist
    dists = cdist(centers, X, metric='euclidean')
    
    # Greedy assignment to ensure unique points
    # (Simple argmin might pick same point for two close centers)
    assigned_mask = np.zeros(len(X), dtype=bool)
    
    for i in range(n_points):
        # Sort candidates by distance to this center
        candidates = np.argsort(dists[i])
        for idx in candidates:
            if not assigned_mask[idx]:
                selected_indices.append(idx)
                assigned_mask[idx] = True
                break
                
    selected_indices = sorted(selected_indices)
    df_indices = df.index[selected_indices].tolist()
    X_final, design_final = build_model_matrix(df.loc[df_indices].reset_index(drop=True), model_terms, ligand_types, ligand_subtypes, include_variant_counts)
    
    return df_indices, X_final, design_final

# ---------------------------
# Example usage
if __name__ == '__main__':
    # example inputs
    elements = {'Fe':[2,3], 'Co':[2,3]}
    coordination = {4:['square_planar','tetra'], 6:['octa']}
    ligand_types = ['L1','L2','L3']   # N = 3 in this example
    ligand_subtypes = ['W','N','D']
    # expand candidates, include variant counts if you want explicit variant columns
    df = expand_candidates_with_variants(
        elements, 
        coordination, 
        ligand_types, 
        ligand_subtypes=ligand_subtypes, 
        include_variant_counts=False
    )
    print("Total feasible candidates:", len(df))
    # choose desired runs
    desired = 30
    
    print("\n--- Method 1: D-Optimal Greedy (Randomized Exchange) ---")
    sel_df_indices, X_sel, design_sel = d_optimal_greedy(
        df, 
        desired, 
        model_terms=None, 
        ligand_types=ligand_types, 
        ligand_subtypes=ligand_subtypes, 
        include_variant_counts=False,
        previous_design=None
    )
    print("Selected df indices:", sel_df_indices)
    
    print("\n--- Method 2: D-Optimal Convex (Gradient-Based Relaxation) ---")
    sel_cvx, X_cvx, design_cvx = d_optimal_convex(
        df,
        desired,
        model_terms=None,
        ligand_types=ligand_types,
        ligand_subtypes=ligand_subtypes
    )
    print("Selected df indices:", sel_cvx)

    # Note: Method 3 (Space Filling) requires sklearn, uncomment if installed
    # print("\n--- Method 3: Space Filling (k-Means) ---")
    # sel_sf, X_sf, design_sf = space_filling_kmeans(df, desired, ligand_types=ligand_types, ligand_subtypes=ligand_subtypes)
    # print("Selected df indices:", sel_sf)

    # optionally export selected runs from greedy
    selected_runs = df.loc[sel_df_indices].reset_index(drop=True)
    selected_runs.to_csv('selected_design.csv', index=False)
    print("\nSelected runs (Greedy) saved to selected_design.csv")