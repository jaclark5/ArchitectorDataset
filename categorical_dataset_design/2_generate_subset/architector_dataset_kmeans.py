"""
Space-Filling Design of Experiments for Chemical Complexes using k-Means Clustering.

This module generates a space-filling experimental design for chemical complexes defined by
hierarchical categorical factors (Element, Oxidation State, Coordination Number, Geometry)
and compositional factors (Ligand Types and Subtypes).

It uses k-Means clustering to select a representative subset of candidates that evenly
covers the chemical design space, which is ideal for training neural networks or other
non-linear models where interpolation is key.

Inputs
------
- Chemical constraints: Elements, Oxidation states, Coordination numbers (CN), Geometries.
- Ligand definitions: Ligand types (scaffolds) and Subtypes (electronic properties).
- Constraints: Number of ligands per CN, forbidden combinations.

Outputs
-------
- A pandas DataFrame containing the selected subset of candidate complexes.
- Columns include categorical factors and numeric counts for ligand types/subtypes.
- A design matrix (X) suitable for regression modeling.

Usage
-----
1. Define chemical space dictionaries (elements, coordination, etc.).
2. Call `expand_candidates_with_variants` to generate the full candidate pool.
3. Call `space_filling_kmeans` to select the most representative subset of runs.
"""

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist



def space_filling_kmeans(
    jsonl_file_path, 
    n_points, 
    model_terms=None, 
    ligand_types=None, 
    ligand_subtypes=None, 
    include_variant_counts=False,
    seed=123,
    previous_design=None
):
    """
    Select a subset of points that are "spread evenly" across the design space using k-Means.
    
    This uses Lloyd's algorithm (gradient-based on cluster distortion) to find n_points centroids,
    then selects the actual candidate closest to each centroid. This is ideal for coverage 
    rather than parameter estimation.

    If a previous_design is provided, points marked as 'evaluated' and 'success' are locked (kept).
    The algorithm then fills the remaining slots by prioritizing centroids that are farthest 
    from the locked points, ensuring the new points fill the "holes" in the design space.

    Parameters
    ----------
    jsonl_file_path : str
        Path to the JSONL file containing candidate data.
    n_points : int
        Number of points to select.
    model_terms : list of str, optional
        List of model terms.
    ligand_types : list of str, optional
        List of ligand types.
    ligand_subtypes : list of str, optional
        List of ligand subtypes.
    include_variant_counts : bool, optional
        Whether to include variant counts.
        
        Pros:
        - Captures specific interactions between ligand scaffold and subtype.
        
        Cons:
        - Increases dimensionality and may lead to sparsity.
    seed : int, optional
        Random seed.
    previous_design : pd.DataFrame, optional
        DataFrame with 'cand_id', 'success', 'evaluated'.
        - evaluated=True, success=True: Locked (kept).
        - evaluated=True, success=False: Discarded (replaced).
        - evaluated=False: Available for replacement (treated as candidate).

    Returns
    -------
    tuple
        (selected_indices, X_final, design_final)
    """

    # Read JSONL file in chunks to build the design matrix
    chunks = []
    with open(jsonl_file_path, 'r') as file:
        for line in file:
            chunks.append(json.loads(line))

    df = pd.DataFrame(chunks)
    X, _ = build_model_matrix(df, model_terms, ligand_types, ligand_subtypes, include_variant_counts)

    # 1. Identify locked points from previous design
    locked_indices = set()
    if previous_design is not None:
        if 'cand_id' in previous_design.columns and 'cand_id' in df.columns:
            cand_id_to_idx = {cid: idx for idx, cid in zip(df.index, df['cand_id'])}
            for _, row in previous_design.iterrows():
                cid = row.get('cand_id')
                if cid in cand_id_to_idx:
                    idx = cand_id_to_idx[cid]
                    evaluated = row.get('evaluated', False)
                    success = row.get('success', False)

                    if evaluated and success:
                        locked_indices.add(idx)
                    # If evaluated=True/success=False -> Discard (don't add to locked)
                    # If evaluated=False -> Don't add to locked (can be re-selected if optimal)
        else:
            raise ValueError("previous_design provided but 'cand_id' column missing in design or candidates.")

    final_indices = list(locked_indices)

    # If we already have enough points, return them
    if len(final_indices) >= n_points:
        df_indices = df.index[final_indices].tolist()
        X_final, design_final = build_model_matrix(
            df.loc[df_indices].reset_index(drop=True), 
            model_terms, 
            ligand_types, 
            ligand_subtypes, 
            include_variant_counts
        )
        return df_indices, X_final, design_final

    # 2. Run KMeans to find N ideal centroids representing the full space
    kmeans = KMeans(n_clusters=n_points, random_state=seed, n_init=10)
    kmeans.fit(X)
    centers = kmeans.cluster_centers_

    # 3. Prioritize centroids that are far from locked points
    if len(locked_indices) > 0:
        X_locked = X[list(locked_indices)]
        # Calculate distance from each centroid to the NEAREST locked point
        # dists: (n_clusters, n_locked)
        dists_to_locked = cdist(centers, X_locked, metric='euclidean')
        min_dists = np.min(dists_to_locked, axis=1)
    else:
        min_dists = np.full(n_points, np.inf)

    sorted_center_indices = np.argsort(min_dists)[::-1]

    # 4. Fill remaining slots
    for center_idx in sorted_center_indices:
        if len(final_indices) >= n_points:
            break
            
        # Find nearest candidate to this centroid
        center = centers[center_idx].reshape(1, -1)
        dists_to_candidates = cdist(center, X, metric='euclidean').flatten()
        
        # Sort candidates by distance to this centroid
        candidate_ranking = np.argsort(dists_to_candidates)
        
        # Pick the first one that isn't already in final_indices
        for cand_idx in candidate_ranking:
            if cand_idx not in final_indices:
                final_indices.append(cand_idx)
                break

    final_indices = sorted(final_indices)
    df_indices = df.index[final_indices].tolist()
    X_final, design_final = build_model_matrix(
        df.loc[df_indices].reset_index(drop=True), 
        model_terms, 
        ligand_types, 
        ligand_subtypes, 
        include_variant_counts
    )

    return df_indices, X_final, design_final


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
        
        Pros:
        - Captures specific interactions between ligand scaffold and subtype.
        
        Cons:
        - Increases dimensionality and may lead to sparsity.

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


def main():
    """
    Command-line interface for executing space_filling_kmeans.

    This function allows users to run the space_filling_kmeans function from the command line.
    """
    parser = argparse.ArgumentParser(
        description="Run space-filling k-Means clustering for chemical design experiments."
    )
    parser.add_argument(
        "--jsonl_file_path",
        type=str,
        required=True,
        help="Path to the JSONL file containing candidate data."
    )
    parser.add_argument(
        "--n_points",
        type=int,
        required=True,
        help="Number of points to select."
    )
    parser.add_argument(
        "--model_terms",
        type=str,
        nargs="*",
        default=None,
        help="List of model terms."
    )
    parser.add_argument(
        "--ligand_types",
        type=str,
        nargs="*",
        default=None,
        help="List of ligand types."
    )
    parser.add_argument(
        "--ligand_subtypes",
        type=str,
        nargs="*",
        default=None,
        help="List of ligand subtypes."
    )
    parser.add_argument(
        "--include_variant_counts",
        action="store_true",
        help="Whether to include variant counts."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed."
    )
    parser.add_argument(
        "--previous_design",
        type=str,
        default=None,
        help="Path to a CSV file containing the previous design."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to save the selected design as a CSV file."
    )

    args = parser.parse_args()

    # Load previous design if provided
    previous_design = None
    if args.previous_design:
        previous_design = pd.read_csv(args.previous_design)

    # Run space_filling_kmeans
    selected_indices, X_final, design_final = space_filling_kmeans(
        jsonl_file_path=args.jsonl_file_path,
        n_points=args.n_points,
        model_terms=args.model_terms,
        ligand_types=args.ligand_types,
        ligand_subtypes=args.ligand_subtypes,
        include_variant_counts=args.include_variant_counts,
        seed=args.seed,
        previous_design=previous_design
    )

    # Save the selected design
    design_final.to_csv(args.output_file, index=False)
    print(f"Selected design saved to {args.output_file}")



