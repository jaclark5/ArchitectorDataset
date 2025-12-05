"""
Space-Filling Design of Experiments for Chemical Complexes using Stratified Farthest Point Sampling.

This module generates a space-filling experimental design for chemical complexes defined by
hierarchical categorical factors (Element, Oxidation State, Coordination Number, Geometry)
and compositional factors (Ligand Types and Subtypes).

It uses Two-Phase Stratified Farthest Point Sampling to select a representative subset of 
candidates that evenly covers the chemical design space, ensuring that rare chemical 
combinations are not lost. This is superior to k-means for space-filling because it:
1. Explicitly guarantees representation of rare strata (chemical categories)
2. Uses maximin spacing within each stratum to maximize coverage
3. Does not bias toward dense regions like k-means does

Major Inputs
------------
- jsonl_file_path : str
    Path to the JSONL file containing candidate data (can be larger than memory).
- n_points : int
    Number of points to select for the space-filling design.
- model_terms : list of str, optional
    List of model terms to include in the design matrix.
- ligand_types : list of str, optional
    List of ligand types for count columns.
- ligand_subtypes : list of str, optional
    List of ligand subtypes for count columns.
- previous_design : pd.DataFrame, optional
    DataFrame with 'cand_id', 'success', 'evaluated' for incremental design.

Major Outputs
-------------
- selected_indices : list of int
    Indices of selected candidates in the original file.
- selected_records : list of dict
    The actual records of selected candidates.
- design_final : pd.DataFrame
    DataFrame of selected candidates with design matrix columns.

Algorithm
---------
Phase 1: Global Stratum Census (Parallel)
    - Each worker scans its chunk of the file
    - Reports stratum counts (Element × Ox × CN × Geometry)
    - Counts are aggregated globally

Phase 2: Quota Allocation
    - Allocate selection quotas per stratum using square-root weighting
    - Ensures rare strata get proportionally more representation
    - Guarantees minimum representation for every stratum

Phase 3: Within-Stratum Farthest Point Sampling (Parallel)
    - For each stratum, run FPS to select maximally spread points
    - FPS iteratively selects the point farthest from all currently selected points
    - Parallelized across strata for efficiency

Usage
-----
1. Generate candidates using `expand_candidates_with_variants` and save to JSONL.
2. Call `stratified_farthest_point_sampling` to select a representative subset.
3. Use the command-line interface for batch processing.

Example
-------
    python architector_dataset_stratified_sampling.py \\
        --jsonl_file_path candidates.jsonl \\
        --n_points 10000 \\
        --output_file selected_design.csv \\
        --n_cores 32
"""

import argparse
import json
import os
import random
import math
from collections import defaultdict
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


def stratified_farthest_point_sampling(
    jsonl_file_path,
    n_points,
    model_terms=None,
    ligand_types=None,
    ligand_subtypes=None,
    include_variant_counts=False,
    seed=123,
    previous_design=None,
    n_cores=None,
    weighting='sqrt',
    min_per_stratum=1,
    chunk_size=100000,
    max_samples_per_stratum=10000
):
    """
    Select a subset of points using Two-Phase Stratified Farthest Point Sampling.

    This method ensures even coverage of the chemical design space by:
    1. Guaranteeing representation of all strata (Element × Ox × CN × Geometry)
    2. Using farthest point sampling within each stratum for maximal spread
    3. Weighting quotas to give rare strata proportionally more representation

    Parameters
    ----------
    jsonl_file_path : str
        Path to the JSONL file containing candidate data.
    n_points : int
        Total number of points to select.
    model_terms : list of str, optional
        List of model terms for the design matrix.
    ligand_types : list of str, optional
        List of ligand types.
    ligand_subtypes : list of str, optional
        List of ligand subtypes.
    include_variant_counts : bool, optional
        Whether to include variant counts in the design matrix.
    seed : int, optional
        Random seed for reproducibility. Default is 123.
    previous_design : pd.DataFrame, optional
        DataFrame with 'cand_id', 'success', 'evaluated' columns.
        - evaluated=True, success=True: Locked (kept in final selection).
        - evaluated=True, success=False: Discarded (not selected).
        - evaluated=False: Available for selection.
    n_cores : int, optional
        Number of CPU cores to use for parallel processing. Default is all available.
    weighting : str, optional
        Quota allocation strategy. One of:
        - 'sqrt': Quota proportional to sqrt(stratum_size). Recommended.
        - 'proportional': Quota proportional to stratum_size.
        - 'equal': Equal quota per stratum.
        Default is 'sqrt'.
    min_per_stratum : int, optional
        Minimum number of points to select from each stratum. Default is 1.
    chunk_size : int, optional
        Number of records to process per batch. Default is 100000.
    max_samples_per_stratum : int, optional
        Maximum samples to keep in memory per stratum during sampling. Default is 10000.

    Returns
    -------
    tuple
        (selected_indices, selected_records, design_final) where:
        - selected_indices: List of global indices of selected candidates.
        - selected_records: List of dictionaries containing the selected records.
        - design_final: pandas DataFrame with the design matrix for selected candidates.

    Notes
    -----
    The algorithm runs in three phases:
    
    Phase 1 (Parallel): Scan the file to count candidates per stratum and collect
    category metadata. This is a lightweight pass that only aggregates counts.
    
    Phase 2: Allocate quotas per stratum based on the weighting strategy. Rare strata
    receive proportionally more representation with 'sqrt' weighting.
    
    Phase 3 (Parallel): For each stratum, perform farthest point sampling to select
    maximally spread points. This is parallelized across strata.
    """
    random.seed(seed)
    np.random.seed(seed)

    if n_cores is None:
        n_cores = cpu_count()

    print(f"Stratified Farthest Point Sampling with {n_cores} cores")
    print(f"Target: {n_points} points")

    # =========================================================================
    # PHASE 1: Parallel Global Stratum Census
    # =========================================================================
    print("\n=== Phase 1: Global Stratum Census ===")

    byte_offsets = _get_file_byte_offsets(jsonl_file_path, n_cores)

    census_args = [
        (jsonl_file_path, start, end, chunk_id)
        for chunk_id, (start, end) in enumerate(byte_offsets)
    ]

    with Pool(n_cores) as pool:
        census_results = pool.map(_census_worker, census_args)

    # Merge census results
    global_stratum_counts = defaultdict(int)
    categories_meta = {'Element': set(), 'Geometry': set()}
    total_rows = 0

    for chunk_counts, chunk_categories, chunk_rows in census_results:
        total_rows += chunk_rows
        for stratum, count in chunk_counts.items():
            global_stratum_counts[stratum] += count
        for col, vals in chunk_categories.items():
            categories_meta[col].update(vals)

    categories_meta = {k: sorted(list(v)) for k, v in categories_meta.items()}
    n_strata = len(global_stratum_counts)

    print(f"  Total candidates: {total_rows:,}")
    print(f"  Unique strata: {n_strata:,}")

    # Get locked candidate IDs from previous design
    locked_cand_ids = set()
    if previous_design is not None and 'cand_id' in previous_design.columns:
        for _, row in previous_design.iterrows():
            if row.get('evaluated', False) and row.get('success', False):
                locked_cand_ids.add(row.get('cand_id'))
    print(f"  Locked candidates: {len(locked_cand_ids)}")

    # =========================================================================
    # PHASE 2: Quota Allocation
    # =========================================================================
    print("\n=== Phase 2: Quota Allocation ===")

    # Calculate effective n_points (excluding locked)
    n_to_select = max(0, n_points - len(locked_cand_ids))

    if n_to_select == 0:
        print("  Already have enough locked points.")
        # Just retrieve locked candidates
        return _retrieve_locked_candidates(
            jsonl_file_path, locked_cand_ids, model_terms, ligand_types,
            ligand_subtypes, include_variant_counts, categories_meta, n_cores, byte_offsets
        )

    # Calculate quotas per stratum
    stratum_quotas = _calculate_quotas(
        global_stratum_counts, n_to_select, weighting, min_per_stratum
    )

    # Print stratum statistics
    quota_values = list(stratum_quotas.values())
    print(f"  Weighting strategy: {weighting}")
    print(f"  Points to select: {n_to_select}")
    print(f"  Quota range: {min(quota_values)} - {max(quota_values)}")
    print(f"  Total quota: {sum(quota_values)}")

    # =========================================================================
    # PHASE 3: Parallel Stratified Sampling + Farthest Point Selection
    # =========================================================================
    print("\n=== Phase 3: Stratified Farthest Point Sampling ===")

    # First, collect samples from each stratum (parallel by file chunk)
    print("  Collecting samples from each stratum...")

    sample_args = [
        (jsonl_file_path, start, end, stratum_quotas, max_samples_per_stratum,
         locked_cand_ids, seed, chunk_id)
        for chunk_id, (start, end) in enumerate(byte_offsets)
    ]

    with Pool(n_cores) as pool:
        sample_results = pool.map(_sample_strata_worker, sample_args)

    # Merge samples by stratum
    stratum_samples = defaultdict(list)
    locked_records = []

    for chunk_samples, chunk_locked in sample_results:
        for stratum, records in chunk_samples.items():
            stratum_samples[stratum].extend(records)
        locked_records.extend(chunk_locked)

    print(f"  Strata with samples: {len(stratum_samples)}")
    print(f"  Locked records found: {len(locked_records)}")

    # Now run FPS within each stratum (parallel by stratum)
    print("  Running Farthest Point Sampling within each stratum...")

    # Prepare FPS arguments
    fps_args = []
    for stratum, records in stratum_samples.items():
        quota = stratum_quotas.get(stratum, min_per_stratum)
        if len(records) > 0:
            fps_args.append((
                stratum, records, quota, model_terms, ligand_types,
                ligand_subtypes, include_variant_counts, categories_meta, seed
            ))

    # Run FPS in parallel across strata
    with Pool(min(n_cores, len(fps_args))) as pool:
        fps_results = pool.map(_fps_worker, fps_args)

    # Collect all selected records
    selected_records = locked_records.copy()
    selected_indices = [r['_original_idx'] for r in locked_records]

    for stratum, selected in fps_results:
        for record in selected:
            if record['_original_idx'] not in selected_indices:
                selected_records.append(record)
                selected_indices.append(record['_original_idx'])

    # Trim to n_points if we have too many
    if len(selected_records) > n_points:
        # Keep all locked, then take from FPS results
        selected_records = selected_records[:n_points]
        selected_indices = selected_indices[:n_points]

    print(f"\n  Total selected: {len(selected_records)}")

    # =========================================================================
    # Build final design matrix
    # =========================================================================
    print("\n=== Building Final Design Matrix ===")

    # Remove internal tracking fields
    clean_records = []
    for record in selected_records:
        clean_record = {k: v for k, v in record.items() if not k.startswith('_')}
        clean_records.append(clean_record)

    df_final = pd.DataFrame(clean_records)
    X_final, design_final = build_model_matrix(
        df_final, model_terms, ligand_types, ligand_subtypes,
        include_variant_counts, categories_meta
    )

    print(f"  Design matrix shape: {X_final.shape}")

    return selected_indices, selected_records, design_final


# =============================================================================
# Worker Functions for Parallel Processing
# =============================================================================

def _census_worker(args):
    """
    Worker function for Phase 1: Count strata in a file chunk.

    Parameters
    ----------
    args : tuple
        (jsonl_file_path, start_byte, end_byte, chunk_id)

    Returns
    -------
    tuple
        (stratum_counts, categories, row_count)
    """
    jsonl_file_path, start_byte, end_byte, chunk_id = args

    stratum_counts = defaultdict(int)
    categories = {'Element': set(), 'Geometry': set()}
    row_count = 0

    with open(jsonl_file_path, 'rb') as f:
        f.seek(start_byte)

        # Skip partial line if not at start
        if start_byte > 0:
            f.readline()

        while f.tell() < end_byte:
            line = f.readline()
            if not line:
                break

            try:
                record = json.loads(line.decode('utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue

            row_count += 1

            # Extract stratum key
            stratum = (
                record.get('Element', ''),
                record.get('Ox', ''),
                record.get('CN', ''),
                record.get('Geometry', '')
            )
            stratum_counts[stratum] += 1

            # Collect category values
            for col in categories:
                if col in record:
                    categories[col].add(record[col])

    return dict(stratum_counts), categories, row_count


def _sample_strata_worker(args):
    """
    Worker function for Phase 3: Collect samples from each stratum using reservoir sampling.

    Parameters
    ----------
    args : tuple
        (jsonl_file_path, start_byte, end_byte, stratum_quotas, max_samples,
         locked_cand_ids, seed, chunk_id)

    Returns
    -------
    tuple
        (stratum_samples, locked_records)
    """
    (jsonl_file_path, start_byte, end_byte, stratum_quotas, max_samples,
     locked_cand_ids, seed, chunk_id) = args

    random.seed(seed + chunk_id)

    stratum_samples = defaultdict(list)
    stratum_counts = defaultdict(int)
    locked_records = []

    # Calculate global index offset
    global_idx_start = 0
    if start_byte > 0:
        with open(jsonl_file_path, 'r') as f:
            for line in f:
                if f.tell() >= start_byte:
                    break
                global_idx_start += 1

    with open(jsonl_file_path, 'rb') as f:
        f.seek(start_byte)

        if start_byte > 0:
            f.readline()

        local_idx = 0
        while f.tell() < end_byte:
            line = f.readline()
            if not line:
                break

            try:
                record = json.loads(line.decode('utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue

            global_idx = global_idx_start + local_idx
            record['_original_idx'] = global_idx
            local_idx += 1

            # Check if this is a locked candidate
            if 'cand_id' in record and record['cand_id'] in locked_cand_ids:
                locked_records.append(record)
                continue

            # Extract stratum key
            stratum = (
                record.get('Element', ''),
                record.get('Ox', ''),
                record.get('CN', ''),
                record.get('Geometry', '')
            )

            stratum_counts[stratum] += 1

            # Reservoir sampling: keep up to max_samples per stratum
            # Sample more than quota to allow FPS to select the best ones
            sample_limit = min(max_samples, stratum_quotas.get(stratum, 1) * 10)

            if len(stratum_samples[stratum]) < sample_limit:
                stratum_samples[stratum].append(record)
            else:
                # Reservoir sampling replacement
                j = random.randint(0, stratum_counts[stratum] - 1)
                if j < sample_limit:
                    stratum_samples[stratum][j] = record

    return dict(stratum_samples), locked_records


def _fps_worker(args):
    """
    Worker function: Run Farthest Point Sampling within a stratum.

    Parameters
    ----------
    args : tuple
        (stratum, records, quota, model_terms, ligand_types, ligand_subtypes,
         include_variant_counts, categories_meta, seed)

    Returns
    -------
    tuple
        (stratum, selected_records)
    """
    (stratum, records, quota, model_terms, ligand_types, ligand_subtypes,
     include_variant_counts, categories_meta, seed) = args

    if len(records) == 0:
        return (stratum, [])

    # If we have fewer records than quota, return all
    if len(records) <= quota:
        return (stratum, records)

    np.random.seed(seed + hash(stratum) % (2**31))

    # Build feature matrix for this stratum
    df_stratum = pd.DataFrame(records)
    X_stratum, _ = build_model_matrix(
        df_stratum, model_terms, ligand_types, ligand_subtypes,
        include_variant_counts, categories_meta
    )

    # Run Farthest Point Sampling
    selected_indices = _farthest_point_sampling(X_stratum, quota, seed)

    selected_records = [records[i] for i in selected_indices]

    return (stratum, selected_records)


def _farthest_point_sampling(X, n_select, seed=None):
    """
    Select n_select points from X using Farthest Point Sampling (FPS).

    FPS iteratively selects the point that is farthest from all currently
    selected points, ensuring maximal spread in the feature space.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    n_select : int
        Number of points to select.
    seed : int, optional
        Random seed for initial point selection.

    Returns
    -------
    list
        Indices of selected points.

    Notes
    -----
    Time complexity: O(n_select * n_samples)
    Space complexity: O(n_samples)
    """
    n_samples = X.shape[0]

    if n_select >= n_samples:
        return list(range(n_samples))

    if seed is not None:
        np.random.seed(seed)

    # Start with a random point
    selected = [np.random.randint(n_samples)]

    # Track minimum distance from each point to the selected set
    min_distances = np.full(n_samples, np.inf)

    # Update distances from the first selected point
    first_point = X[selected[0]].reshape(1, -1)
    min_distances = cdist(X, first_point, metric='euclidean').flatten()
    min_distances[selected[0]] = -np.inf  # Exclude already selected

    for _ in range(n_select - 1):
        # Select the point with maximum minimum distance
        next_idx = np.argmax(min_distances)
        selected.append(next_idx)

        # Update minimum distances
        new_point = X[next_idx].reshape(1, -1)
        new_distances = cdist(X, new_point, metric='euclidean').flatten()
        min_distances = np.minimum(min_distances, new_distances)
        min_distances[next_idx] = -np.inf  # Exclude already selected

    return selected


# =============================================================================
# Helper Functions
# =============================================================================

def _get_file_byte_offsets(jsonl_file_path, n_chunks):
    """
    Get byte offsets to split a JSONL file into roughly equal chunks.

    Parameters
    ----------
    jsonl_file_path : str
        Path to the JSONL file.
    n_chunks : int
        Number of chunks to create.

    Returns
    -------
    list of tuple
        List of (start_byte, end_byte) tuples.
    """
    file_size = os.path.getsize(jsonl_file_path)
    chunk_size = file_size // n_chunks

    offsets = []
    with open(jsonl_file_path, 'rb') as f:
        start = 0
        for i in range(n_chunks):
            if i == n_chunks - 1:
                end = file_size
            else:
                target = (i + 1) * chunk_size
                f.seek(target)
                f.readline()  # Move to end of current line
                end = f.tell()

            offsets.append((start, end))
            start = end

    return offsets


def _calculate_quotas(stratum_counts, n_points, weighting, min_per_stratum):
    """
    Calculate selection quotas per stratum.

    Parameters
    ----------
    stratum_counts : dict
        Dictionary mapping stratum keys to counts.
    n_points : int
        Total number of points to select.
    weighting : str
        Weighting strategy: 'sqrt', 'proportional', or 'equal'.
    min_per_stratum : int
        Minimum quota per stratum.

    Returns
    -------
    dict
        Dictionary mapping stratum keys to quotas.
    """
    n_strata = len(stratum_counts)
    total_count = sum(stratum_counts.values())

    if weighting == 'equal':
        base_quota = n_points // n_strata
        quotas = {s: max(min_per_stratum, base_quota) for s in stratum_counts}

    elif weighting == 'proportional':
        quotas = {}
        for stratum, count in stratum_counts.items():
            proportion = count / total_count
            quota = max(min_per_stratum, int(n_points * proportion))
            quotas[stratum] = min(quota, count)  # Can't select more than exist

    elif weighting == 'sqrt':
        # Square-root weighting gives rare strata more representation
        sqrt_counts = {s: math.sqrt(c) for s, c in stratum_counts.items()}
        total_sqrt = sum(sqrt_counts.values())

        quotas = {}
        for stratum, count in stratum_counts.items():
            proportion = sqrt_counts[stratum] / total_sqrt
            quota = max(min_per_stratum, int(n_points * proportion))
            quotas[stratum] = min(quota, count)

    else:
        raise ValueError(f"Unknown weighting strategy: {weighting}")

    # Adjust quotas to sum to n_points
    total_quota = sum(quotas.values())
    if total_quota > n_points:
        # Scale down proportionally, respecting minimums
        scale = n_points / total_quota
        for stratum in quotas:
            quotas[stratum] = max(min_per_stratum, int(quotas[stratum] * scale))
    elif total_quota < n_points:
        # Distribute remaining quota to largest strata
        remaining = n_points - total_quota
        sorted_strata = sorted(stratum_counts.keys(), key=lambda s: -stratum_counts[s])
        for stratum in sorted_strata:
            if remaining <= 0:
                break
            can_add = min(remaining, stratum_counts[stratum] - quotas[stratum])
            if can_add > 0:
                quotas[stratum] += can_add
                remaining -= can_add

    return quotas


def _retrieve_locked_candidates(
    jsonl_file_path, locked_cand_ids, model_terms, ligand_types,
    ligand_subtypes, include_variant_counts, categories_meta, n_cores, byte_offsets
):
    """
    Retrieve locked candidates from the file.

    Parameters
    ----------
    jsonl_file_path : str
        Path to JSONL file.
    locked_cand_ids : set
        Set of locked candidate IDs.
    model_terms, ligand_types, ligand_subtypes, include_variant_counts, categories_meta
        Parameters for build_model_matrix.
    n_cores : int
        Number of cores for parallel retrieval.
    byte_offsets : list
        File byte offsets.

    Returns
    -------
    tuple
        (selected_indices, selected_records, design_final)
    """
    retrieve_args = [
        (jsonl_file_path, start, end, locked_cand_ids, chunk_id)
        for chunk_id, (start, end) in enumerate(byte_offsets)
    ]

    with Pool(n_cores) as pool:
        results = pool.map(_retrieve_locked_worker, retrieve_args)

    locked_records = []
    for chunk_locked in results:
        locked_records.extend(chunk_locked)

    selected_indices = [r['_original_idx'] for r in locked_records]

    # Clean records
    clean_records = [{k: v for k, v in r.items() if not k.startswith('_')} for r in locked_records]

    df_final = pd.DataFrame(clean_records)
    X_final, design_final = build_model_matrix(
        df_final, model_terms, ligand_types, ligand_subtypes,
        include_variant_counts, categories_meta
    )

    return selected_indices, locked_records, design_final


def _retrieve_locked_worker(args):
    """
    Worker to retrieve locked candidates from a file chunk.

    Parameters
    ----------
    args : tuple
        (jsonl_file_path, start_byte, end_byte, locked_cand_ids, chunk_id)

    Returns
    -------
    list
        List of locked records.
    """
    jsonl_file_path, start_byte, end_byte, locked_cand_ids, chunk_id = args

    locked_records = []

    global_idx_start = 0
    if start_byte > 0:
        with open(jsonl_file_path, 'r') as f:
            for line in f:
                if f.tell() >= start_byte:
                    break
                global_idx_start += 1

    with open(jsonl_file_path, 'rb') as f:
        f.seek(start_byte)

        if start_byte > 0:
            f.readline()

        local_idx = 0
        while f.tell() < end_byte:
            line = f.readline()
            if not line:
                break

            try:
                record = json.loads(line.decode('utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue

            if 'cand_id' in record and record['cand_id'] in locked_cand_ids:
                record['_original_idx'] = global_idx_start + local_idx
                locked_records.append(record)

            local_idx += 1

    return locked_records


def build_model_matrix(df, model_terms=None, ligand_types=None, ligand_subtypes=None,
                       include_variant_counts=False, categories_meta=None):
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
    categories_meta : dict, optional
        Dictionary mapping categorical column names to their possible categories.
        Used for consistent one-hot encoding across different subsets.

    Returns
    -------
    tuple
        (X, design) where X is the numpy array model matrix and design is the DataFrame version.
    """
    if len(df) == 0:
        return np.array([]).reshape(0, 0), pd.DataFrame()

    if model_terms is None:
        terms = ['Element', 'Ox', 'CN', 'Geometry']
        if ligand_types is None:
            ligand_types = sorted({
                c.replace('count_type_', '')
                for c in df.columns if c.startswith('count_type_')
            })
        if ligand_subtypes is None:
            ligand_subtypes = sorted({
                c.replace('count_prop_', '')
                for c in df.columns if c.startswith('count_prop_')
            })
        terms += [f'count_type_{t}' for t in ligand_types]
        terms += [f'count_prop_{p}' for p in ligand_subtypes]
        if include_variant_counts:
            variant_cols = [c for c in df.columns if c.startswith('count_var_')]
            terms += variant_cols
    else:
        terms = model_terms

    design = pd.DataFrame(index=df.index)

    for c in terms:
        if c not in df.columns:
            continue

        if df[c].dtype == object and not c.startswith('count_'):
            # Categorical column
            if categories_meta and c in categories_meta:
                # Use consistent categories
                cat_type = pd.CategoricalDtype(categories=categories_meta[c], ordered=False)
                col_cat = df[c].astype(cat_type)
                dummies = pd.get_dummies(col_cat, prefix=c, drop_first=True)
                # Ensure all expected columns exist
                expected_cols = [f"{c}_{cat}" for cat in categories_meta[c][1:]]
                for exp_col in expected_cols:
                    if exp_col not in dummies.columns:
                        dummies[exp_col] = 0
                dummies = dummies[expected_cols]
            else:
                dummies = pd.get_dummies(df[c].astype(str), prefix=c, drop_first=True)
            design = pd.concat([design, dummies], axis=1)
        else:
            design[c] = df[c]

    design.insert(0, 'Intercept', 1.0)
    X = design.to_numpy(dtype=float)
    return X, design


if __name__ == "__main__":
    """
    Command-line interface for executing stratified_farthest_point_sampling.

    This function allows users to run the stratified FPS algorithm from the command line
    with support for large datasets that don't fit in memory.
    """
    parser = argparse.ArgumentParser(
        description="Run Stratified Farthest Point Sampling for chemical design experiments."
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
        help="Random seed. Default is 123."
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
    parser.add_argument(
        "--n_cores",
        type=int,
        default=None,
        help="Number of CPU cores to use. Default is all available."
    )
    parser.add_argument(
        "--weighting",
        type=str,
        choices=['sqrt', 'proportional', 'equal'],
        default='sqrt',
        help="Quota allocation strategy. Default is 'sqrt'."
    )
    parser.add_argument(
        "--min_per_stratum",
        type=int,
        default=1,
        help="Minimum points to select from each stratum. Default is 1."
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=100000,
        help="Number of records per processing batch. Default is 100000."
    )
    parser.add_argument(
        "--max_samples_per_stratum",
        type=int,
        default=10000,
        help="Maximum samples to keep per stratum during sampling. Default is 10000."
    )

    args = parser.parse_args()

    # Load previous design if provided
    previous_design = None
    if args.previous_design:
        previous_design = pd.read_csv(args.previous_design)

    # Run stratified FPS
    selected_indices, selected_records, design_final = stratified_farthest_point_sampling(
        jsonl_file_path=args.jsonl_file_path,
        n_points=args.n_points,
        model_terms=args.model_terms,
        ligand_types=args.ligand_types,
        ligand_subtypes=args.ligand_subtypes,
        include_variant_counts=args.include_variant_counts,
        seed=args.seed,
        previous_design=previous_design,
        n_cores=args.n_cores,
        weighting=args.weighting,
        min_per_stratum=args.min_per_stratum,
        chunk_size=args.chunk_size,
        max_samples_per_stratum=args.max_samples_per_stratum
    )

    # Save the selected design
    # Create DataFrame from selected records (without internal fields)
    clean_records = [{k: v for k, v in r.items() if not k.startswith('_')} for r in selected_records]
    df_output = pd.DataFrame(clean_records)
    df_output.to_csv(args.output_file, index=False)

    print(f"\nSelected design saved to {args.output_file}")
    print(f"Total selected: {len(selected_indices)} candidates")

