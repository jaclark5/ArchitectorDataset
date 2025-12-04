"""
Space-Filling Design of Experiments for Chemical Complexes using k-Means Clustering.

This module generates a space-filling experimental design for chemical complexes defined by
hierarchical categorical factors (Element, Oxidation State, Coordination Number, Geometry)
and compositional factors (Ligand Types and Subtypes).

It uses k-Means clustering to select a representative subset of candidates that evenly
covers the chemical design space, which is ideal for training neural networks or other
non-linear models where interpolation is key.

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
    3. Call `space_filling_kmeans` to select the most representative subset of runs.
"""

import itertools
from collections import Counter
import math
import concurrent.futures
import json

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist


def expand_candidates_with_variants(
    elements,
    coordination,
    ligand_types,
    ligand_subtypes=['W', 'N', 'D'],
    include_variant_counts=False,
    forbid_fn=None,
    max_abs_charge=None,
    ligand_charges=None,
    n_jobs=None,
    output_file=None
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
        
        Pros:
        - Captures specific interactions between ligand scaffold and subtype (e.g., "L1-W").
        - Provides higher resolution of the chemical space.
        
        Cons:
        - Significantly increases dimensionality (combinatorial explosion).
        - Can lead to sparse matrices and overfitting if data is limited.
        - May be redundant if effects are primarily additive (Type + Subtype).
    forbid_fn : callable, optional
        A function that takes two arguments:
        - candidate (dict): A dictionary representing a chemical complex candidate.
        - variant_to_components (dict): A dictionary mapping ligand variant identifiers to their properties.
        Returns True if the candidate should be excluded, False otherwise. Default is None.
    max_abs_charge : int, optional
        Maximum absolute charge allowed for a complex. If the absolute value of the oxidation
        state plus the total charge of the ligands is greater than this value, the complex is 
        not recorded. Default is None.
    ligand_charges : list[int], optional
        List of the same length as the ligand types containing their charge as if they've taken
        any metal electrons so that X-type ligands have a charge of -1. Default is None.
    n_jobs : int, optional
        Number of parallel jobs to run. If None, uses default ProcessPoolExecutor behavior.
        If 1, runs serially.
    output_file : str, optional
        Path to a file (JSONL or CSV) to save candidates incrementally. 
        If provided, the function will write results to this file as they are generated
        and return None (or an empty DataFrame) to avoid memory issues.
        JSONL (.jsonl) is recommended for robustness.

    Returns
    -------
    pd.DataFrame or None
        DataFrame containing all valid candidate complexes, or None if output_file is used.
    """
    
    if max_abs_charge is not None:
        if ligand_charges is None:
            raise ValueError("ligand_charges must be provided if max_abs_charge is not None.")
        if len(ligand_charges) != len(ligand_types):
            raise ValueError("ligand_charges must map to all ligand_types, lists are not the same length.")
    else:
        if ligand_charges is not None:
            ligand_charges = None
        

    # build full list of ligand variants as strings like 'L1|W'
    variants = []
    variant_to_components = {}
    for i,t in enumerate(ligand_types):
        for s in ligand_subtypes:
            key = f"{t}|{s}"
            variants.append(key)
            variant_to_components[key] = {'type': t, 'prop': s}
            if max_abs_charge is not None:
                variant_to_components[key]['chg'] = ligand_charges[i]

    def task_generator():
        for elem, ox_list in elements.items():
            for ox in ox_list:
                for cn, geoms in coordination.items():
                    for geom in geoms:
                        # Split tasks by the first ligand to reduce task size and allow better parallelization
                        # We fix the first variant, and choose the remaining (cn-1) from variants[i:]
                        # This corresponds to the logic of combinations_with_replacement
                        for i, first_variant in enumerate(variants):
                            # OPTIONAL OPTIMIZATION: Check forbid_fn on partial candidate (1st ligand selected)
                            # This allows us to skip creating tasks for ligands that are immediately invalid.
                            # However, this DOES NOT replace the check in _generate_candidates_batch, 
                            # which validates the full ligand set.
                            if forbid_fn is not None:
                                comp = variant_to_components[first_variant]
                                # Construct minimal candidate with 0s for all counts
                                cand = {
                                    'Element': elem,
                                    'Ox': ox,
                                    'CN': cn,
                                    'Geometry': geom,
                                    'Ligand_multiset_variants': (first_variant,),
                                }
                                # counts per type
                                for t in ligand_types:
                                    cand[f'count_type_{t}'] = 1 if t == comp['type'] else 0
                                # counts per property
                                for p in ligand_subtypes:
                                    cand[f'count_prop_{p}'] = 1 if p == comp['prop'] else 0
                                # optional: counts per variant
                                if include_variant_counts:
                                    for v in variants:
                                        col = f"count_var_{v.replace('|','_')}"
                                        cand[col] = 1 if v == first_variant else 0

                            yield (
                                elem, ox, cn, geom,
                                (first_variant,), variants[i:], cn - 1,
                                variant_to_components, 
                                ligand_types, ligand_subtypes, 
                                include_variant_counts, forbid_fn, max_abs_charge
                            )

    # Helper to write batch to file
    def write_batch(batch_rows, file_handle, is_jsonl):
        if not batch_rows:
            return
        if is_jsonl:
            for row in batch_rows:
                file_handle.write(json.dumps(row) + '\n')
        else:
            # CSV mode (less robust for complex types like tuples, but standard)
            # We need to ensure header is written only once.
            # This simple helper assumes file_handle is open.
            # For CSV, we'd need a DictWriter and consistent keys.
            # JSONL is preferred for nested data like 'Ligand_multiset_variants'.
            pass

    # Prepare output file if requested
    f_out = None
    is_jsonl = False
    if output_file:
        if output_file.endswith('.jsonl'):
            is_jsonl = True
            f_out = open(output_file, 'w')
        else:
            # Fallback or error? Let's support JSONL primarily for now as it handles the tuple data best.
            # If user asks for CSV, we might need to flatten the tuple or stringify it.
            print("Warning: output_file should end in .jsonl for best results. Writing as JSONL anyway.")
            is_jsonl = True
            f_out = open(output_file, 'w')

    rows = []
    try:
        if n_jobs == 1:
            for task in task_generator():
                batch = _generate_candidates_batch(*task)
                if f_out:
                    write_batch(batch, f_out, is_jsonl)
                else:
                    rows.extend(batch)
        else:
            # Use ProcessPoolExecutor for parallel execution
            with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
                # Submit all tasks
                futures = [executor.submit(_generate_candidates_batch, *task) for task in task_generator()]
                
                for future in concurrent.futures.as_completed(futures):
                    batch = future.result()
                    if f_out:
                        write_batch(batch, f_out, is_jsonl)
                    else:
                        rows.extend(batch)
    finally:
        if f_out:
            f_out.close()

    if output_file:
        print(f"Candidates saved to {output_file}")
        return None

    df = pd.DataFrame(rows)
    df.insert(0, 'cand_id', range(1, len(df)+1))
    return df

def calculate_candidate_count(
    elements,
    coordination,
    ligand_types,
    ligand_subtypes=['W', 'N', 'D'],
    include_variant_counts=False,
    forbid_fn=None,
    max_abs_charge=None,
    ligand_charges=None,
):
    """
    Calculate the exact number of candidates that will be generated.

    Parameters
    ----------
    elements : dict
        Dictionary mapping element symbols to lists of oxidation states.
    coordination : dict
        Dictionary mapping coordination numbers to lists of geometry names.
    ligand_types : list of str
        List of ligand type identifiers.
    ligand_subtypes : list of str, optional
        List of ligand subtype identifiers. Default is ['W', 'N', 'D'].
    include_variant_counts : bool, optional
        Whether to include variant counts (only relevant if forbid_fn uses them). Default is False.
    forbid_fn : callable, optional
        A function that takes two arguments:
        - candidate (dict): A dictionary representing a chemical complex candidate.
        - variant_to_components (dict): A dictionary mapping ligand variant identifiers to their properties.
        Returns True if the candidate should be excluded, False otherwise. Default is None.
    max_abs_charge : int, optional
        Maximum absolute charge allowed for a complex. If the absolute value of the oxidation
        state plus the total charge of the ligands is greater than this value, the complex is 
        not recorded. Default is None.
    ligand_charges : list[int], optional
        List of the same length as the ligand types containing their charge as if they've taken
        any metal electrons so that X-type ligands have a charge of -1. Default is None.

    Returns
    -------
    int
        Number of candidates.
    """

    # If no forbid_fn, use combinatorial formula (much faster)
    if forbid_fn is None and max_abs_charge is None:
        num_variants = len(ligand_types) * len(ligand_subtypes)
        num_metal_ox = sum(len(oxs) for oxs in elements.values())
        
        total_structs = 0
        for cn, geoms in coordination.items():
            # (n + k - 1) choose k
            n_multisets = math.comb(num_variants + cn - 1, cn)
            total_structs += n_multisets * len(geoms)
            
        return num_metal_ox * total_structs

    if max_abs_charge is not None:
        if ligand_charges is None:
            raise ValueError("ligand_charges must be provided if max_abs_charge is not None.")
        if len(ligand_charges) != len(ligand_types):
            raise ValueError("ligand_charges must map to all ligand_types, lists are not the same length.")
    else:
        if ligand_charges is not None:
            ligand_charges = None

    # If forbid_fn exists, we must iterate and check
    variants = []
    variant_to_components = {}
    for i,t in enumerate(ligand_types):
        for s in ligand_subtypes:
            key = f"{t}|{s}"
            variants.append(key)
            variant_to_components[key] = {'type': t, 'prop': s}
            if max_abs_charge is not None:
                variant_to_components[key]['chg'] = ligand_charges[i]

    count = 0
    for elem, ox_list in elements.items():
        for ox in ox_list:
            for cn, geoms in coordination.items():
                for geom in geoms:
                    ligand_multisets = itertools.combinations_with_replacement(variants, cn)
                    for lig_multi in ligand_multisets:
                        # Construct minimal candidate for forbid_fn
                        counts_variant = Counter(lig_multi)
                        counts_type = Counter()
                        counts_prop = Counter()
                        charge = 0
                        for var_key, ct in counts_variant.items():
                            comp = variant_to_components[var_key]
                            counts_type[comp['type']] += ct
                            counts_prop[comp['prop']] += ct
                            if max_abs_charge is not None:
                                charge += comp['chg']
                        if max_abs_charge is not None and abs(charge) > max_abs_charge: # No warning that it is skipped
                            continue
                            
                        candidate = {
                            'Element': elem,
                            'Ox': ox,
                            'CN': cn,
                            'Geometry': geom,
                            'Ligand_multiset_variants': tuple(lig_multi),
                        }
                        for t in ligand_types:
                            candidate[f'count_type_{t}'] = counts_type.get(t, 0)
                        for p in ligand_subtypes:
                            candidate[f'count_prop_{p}'] = counts_prop.get(p, 0)
                            
                        if include_variant_counts:
                            for v in variants:
                                col = f"count_var_{v.replace('|','_')}"
                                candidate[col] = counts_variant.get(v, 0)
                        
                        if forbid_fn is not None and forbid_fn(candidate):
                            continue
                        count += 1
    return count

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


def space_filling_kmeans(
    df, 
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
    df : pd.DataFrame
        Candidate DataFrame.
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
        X_final, design_final = build_model_matrix(df.loc[df_indices].reset_index(drop=True), model_terms, ligand_types, ligand_subtypes, include_variant_counts)
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
        # min_dists: (n_clusters,)
        min_dists = np.min(dists_to_locked, axis=1)
    else:
        # If no locked points, all centroids are equally valid (distance infinity)
        min_dists = np.full(n_points, np.inf)
        
    # Sort centroids by distance to locked set (descending) -> fill holes first
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
    
    print("\n--- Method: Space Filling (k-Means) ---")
    # Note: Requires sklearn
    try:
        sel_sf, X_sf, design_sf = space_filling_kmeans(df, desired, ligand_types=ligand_types, ligand_subtypes=ligand_subtypes)
        print("Selected df indices:", sel_sf)
        
        # optionally export selected runs
        selected_runs = df.loc[sel_sf].reset_index(drop=True)
        selected_runs.to_csv('selected_design.csv', index=False)
        print("\nSelected runs saved to selected_design.csv")
    except ImportError:
        print("Error: scikit-learn is required for space_filling_kmeans.")

def _generate_candidates_batch(
    elem, 
    ox, 
    cn, 
    geom, 
    fixed_ligands,
    pool_variants,
    n_remaining,
    variant_to_components, 
    ligand_types, 
    ligand_subtypes, 
    include_variant_counts, 
    forbid_fn, 
    max_abs_charge
):
    rows = []
    
    # If max_abs_charge is set, use a recursive generator with pruning
    if max_abs_charge is not None:
        min_target = -max_abs_charge - ox
        max_target = max_abs_charge - ox
        
        # Pre-fetch charges for pool variants to avoid dict lookups in inner loop
        pool_with_charges = []
        for v in pool_variants:
            c = variant_to_components[v].get('chg', 0)
            pool_with_charges.append((v, c))
            
        if pool_with_charges:
            charges = [c for _, c in pool_with_charges]
            min_pool_chg = min(charges)
            max_pool_chg = max(charges)
        else:
            min_pool_chg = 0
            max_pool_chg = 0
            
        # Calculate charge of fixed ligands
        current_charge = 0
        for v in fixed_ligands:
            current_charge += variant_to_components[v].get('chg', 0)
            
        def generate_combinations_recursive(start_idx, current_n, current_q):
            if current_n < 0:
                return
            if current_n == 0:
                if min_target <= current_q <= max_target:
                    yield ()
                return

            # Pruning: check if it's possible to reach the target range
            # If even adding the max possible charge is not enough to reach min_target
            if current_q + (current_n * max_pool_chg) < min_target:
                return
            # If even adding the min possible charge exceeds max_target
            if current_q + (current_n * min_pool_chg) > max_target:
                return

            for i in range(start_idx, len(pool_with_charges)):
                v, c = pool_with_charges[i]
                # Recurse
                for tail in generate_combinations_recursive(i, current_n - 1, current_q + c):
                    yield (v,) + tail

        combinations_iter = generate_combinations_recursive(0, n_remaining, current_charge)
        
    else:
        combinations_iter = itertools.combinations_with_replacement(pool_variants, n_remaining)

    for tail in combinations_iter:
        lig_multi = fixed_ligands + tail
        counts_variant = Counter(lig_multi)
        counts_type = Counter()
        counts_prop = Counter()
        
        charge = ox
        for var_key, ct in counts_variant.items():
            comp = variant_to_components[var_key]
            counts_type[comp['type']] += ct
            counts_prop[comp['prop']] += ct
            if max_abs_charge is not None:
                charge += comp['chg']
        
        # Note: If max_abs_charge is not None, the generator guarantees validity,
        # so we don't strictly need to check again, but we calculate 'charge' anyway.
        
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
            for v in variant_to_components:
                col = f"count_var_{v.replace('|','_')}"
                candidate[col] = counts_variant.get(v, 0)

        if forbid_fn and forbid_fn(candidate, variant_to_components):
            continue
        rows.append(candidate)
    return rows

def skip_function(candidate, variant_to_components):
    """
    Determine whether a candidate should be excluded based on ligand properties.

    Parameters
    ----------
    candidate : dict
        A dictionary representing a chemical complex candidate. It must include the key
        'Ligand_multiset_variants', which is a tuple of ligand variant identifiers.
    variant_to_components : dict
        A dictionary mapping ligand variant identifiers (str) to their properties.
        Each entry is a dictionary with at least the following keys:
        - 'type' (str): The ligand type.
        - 'prop' (str): The ligand property (e.g., electronic property).

    Returns
    -------
    bool
        True if the candidate should be excluded, False otherwise.

    Notes
    -----
    This function specifically excludes candidates containing ligands from the
    "neutral_only_ligands" list that do not have the property "N".
    """

    neutral_only_ligands = ["choride", "oxo", "sulfido"]
    for lig_key in candidate['Ligand_multiset_variants']:
        comp = variant_to_components[lig_key]
        if comp["type"] in neutral_only_ligands and comp["prop"] != "N":
            return True

    return False