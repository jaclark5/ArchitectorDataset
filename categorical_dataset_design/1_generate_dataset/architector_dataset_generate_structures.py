"""
Module: Generate Chemical Complex Structures

This module provides functionality to generate chemical complex structures based on hierarchical
categorical factors (e.g., Element, Oxidation State, Coordination Number, Geometry) and compositional
factors (e.g., Ligand Types and Subtypes). It supports the generation of candidate structures and
space-filling experimental designs for chemical complexes.

Major Inputs
------------
- Elements dictionary: Maps element symbols to their possible oxidation states.
- Coordination dictionary: Maps coordination numbers to geometry names.
- Ligand types: List of ligand type identifiers.
- Ligand subtypes: List of ligand subtype identifiers.
- Optional constraints: Maximum absolute charge, forbidden combinations, etc.

Major Outputs
-------------
- A pandas DataFrame containing the generated candidate structures.
- Optionally, the candidates can be saved to a JSONL or CSV file.
- If `count_only` is enabled, the total number of structures is returned instead of generating them.

Usage
-----
1. Define the chemical space dictionaries (elements, coordination, etc.).
2. Use the `expand_candidates_with_variants` function to generate the full candidate pool.
3. Optionally, save the generated candidates to a file or count the total number of structures.
4. Use the command-line interface to execute the module directly.
"""

import argparse
import itertools
from collections import Counter
import json
import multiprocessing

import pandas as pd


def _generate_ligand_combinations(cn, variants, variant_to_components, ox, max_abs_charge):
    """
    Generate all valid ligand multisets for a given coordination number.
    
    Yields tuples of ligand variant keys that satisfy charge constraints.
    
    Parameters
    ----------
    cn : int
        Coordination number (number of ligands).
    variants : list of str
        List of ligand variant identifiers (e.g., 'L1|W').
    variant_to_components : dict
        Maps variant keys to their properties including charge.
    ox : int
        Oxidation state of the metal center.
    max_abs_charge : int or None
        Maximum absolute charge constraint. If None, no charge filtering.
        
    Yields
    ------
    tuple of str
        A multiset of ligand variants as a sorted tuple.
    """
    if max_abs_charge is not None:
        min_target = -max_abs_charge - ox
        max_target = max_abs_charge - ox
        
        # Pre-fetch charges for variants
        pool_with_charges = []
        for v in variants:
            c = variant_to_components[v].get('chg', 0)
            pool_with_charges.append((v, c))
        
        if pool_with_charges:
            charges = [c for _, c in pool_with_charges]
            min_pool_chg = min(charges)
            max_pool_chg = max(charges)
        else:
            min_pool_chg = 0
            max_pool_chg = 0
        
        def generate_recursive(start_idx, current_n, current_q):
            if current_n == 0:
                if min_target <= current_q <= max_target:
                    yield ()
                return
            
            # Pruning
            if current_q + (current_n * max_pool_chg) < min_target:
                return
            if current_q + (current_n * min_pool_chg) > max_target:
                return
            
            for i in range(start_idx, len(pool_with_charges)):
                v, c = pool_with_charges[i]
                for tail in generate_recursive(i, current_n - 1, current_q + c):
                    yield (v,) + tail
        
        yield from generate_recursive(0, cn, 0)
    else:
        yield from itertools.combinations_with_replacement(variants, cn)


def candidate_generator(elements, coordination, variants, variant_to_components, max_abs_charge, forbid_fn=None):
    """
    Generator that yields individual candidate structures one at a time.
    
    This is the core generator for memory-efficient iteration over all possible
    chemical complex structures. Each yielded item contains all information
    needed to construct a candidate dictionary.
    
    Parameters
    ----------
    elements : dict
        Maps element symbols to lists of oxidation states.
    coordination : dict
        Maps coordination numbers to lists of geometry names.
    variants : list of str
        List of all ligand variant identifiers.
    variant_to_components : dict
        Maps variant keys to their properties.
    max_abs_charge : int or None
        Maximum absolute charge constraint.
    forbid_fn : callable, optional
        A function that takes (candidate, variant_to_components) and returns True
        if the candidate should be excluded. If provided, forbidden candidates
        are never yielded.
        
    Yields
    ------
    tuple
        (elem, ox, cn, geom, ligand_multiset) for each valid candidate.
    """
    for elem, ox_list in elements.items():
        for ox in ox_list:
            for cn, geoms in coordination.items():
                for geom in geoms:
                    for lig_multi in _generate_ligand_combinations(
                        cn, variants, variant_to_components, ox, max_abs_charge
                    ):
                        # Early filtering with forbid_fn if provided
                        if forbid_fn is not None:
                            # Build minimal candidate dict for forbid_fn check
                            candidate = {
                                'Element': elem,
                                'Ox': ox,
                                'CN': cn,
                                'Geometry': geom,
                                'Ligand_multiset_variants': list(lig_multi),
                            }
                            if forbid_fn(candidate, variant_to_components):
                                continue
                        yield (elem, ox, cn, geom, lig_multi)


# Module-level shared data for worker processes (set by initializer)
_worker_data = {}


def _init_worker(variant_to_components, ligand_types, ligand_subtypes, 
                 include_variant_counts, max_abs_charge):
    """Initialize worker process with shared data."""
    global _worker_data
    _worker_data['variant_to_components'] = variant_to_components
    _worker_data['ligand_types'] = ligand_types
    _worker_data['ligand_subtypes'] = ligand_subtypes
    _worker_data['include_variant_counts'] = include_variant_counts
    _worker_data['max_abs_charge'] = max_abs_charge


def _process_single_candidate(args):
    """
    Process a single candidate structure.
    
    Note: forbid_fn filtering is done in the generator before candidates
    reach this function, so all candidates here are valid.
    
    Parameters
    ----------
    args : tuple
        (elem, ox, cn, geom, lig_multi) tuple from candidate_generator.
        
    Returns
    -------
    dict
        Candidate dictionary with all computed fields.
    """
    elem, ox, cn, geom, lig_multi = args
    
    # Access shared worker data
    variant_to_components = _worker_data['variant_to_components']
    ligand_types = _worker_data['ligand_types']
    ligand_subtypes = _worker_data['ligand_subtypes']
    include_variant_counts = _worker_data['include_variant_counts']
    max_abs_charge = _worker_data['max_abs_charge']
    
    counts_variant = Counter(lig_multi)
    counts_type = Counter()
    counts_prop = Counter()
    
    charge = ox
    for var_key, ct in counts_variant.items():
        comp = variant_to_components[var_key]
        counts_type[comp['type']] += ct
        counts_prop[comp['prop']] += ct
        if max_abs_charge is not None:
            charge += comp.get('chg', 0) * ct
    
    candidate = {
        'Element': elem,
        'Ox': ox,
        'CN': cn,
        'Geometry': geom,
        'Charge': charge,
        'Ligand_multiset_variants': list(lig_multi),
    }
    
    # counts per type
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
    
    return candidate


def count_candidates(
    elements,
    coordination,
    ligand_types,
    ligand_subtypes=['W', 'N', 'D'],
    forbid_fn=None,
    forbid_fn_name=None,
    max_abs_charge=None,
    ligand_charges=None,
    progress_interval=1000000,
):
    """
    Efficiently count the number of valid candidate structures.
    
    This is much faster than expand_candidates_with_variants with count_only=True
    because it skips all the multiprocessing setup and candidate dict construction.
    
    Parameters
    ----------
    elements : dict
        Dictionary mapping element symbols (str) to lists of oxidation states (int).
    coordination : dict
        Dictionary mapping coordination numbers (int) to lists of geometry names (str).
    ligand_types : list of str
        List of ligand type identifiers.
    ligand_subtypes : list of str, optional
        List of ligand subtype identifiers. Default is ['W', 'N', 'D'].
    forbid_fn : callable, optional
        A function that takes (candidate, variant_to_components) and returns True
        if the candidate should be excluded.
    forbid_fn_name : str, optional
        Module path to the forbid function (e.g., "module.submodule.function").
    max_abs_charge : int, optional
        Maximum absolute charge allowed for a complex.
    ligand_charges : list[int], optional
        List of charges for each ligand type.
    progress_interval : int, optional
        Print progress every N candidates. Default is 1,000,000. Set to 0 to disable.
        
    Returns
    -------
    int
        Total number of valid candidates.
    """
    if max_abs_charge is not None:
        if ligand_charges is None:
            raise ValueError("ligand_charges must be provided if max_abs_charge is not None.")
        if len(ligand_charges) != len(ligand_types):
            raise ValueError("ligand_charges must map to all ligand_types.")
    else:
        ligand_charges = None

    # Build variant list and mapping
    variants = []
    variant_to_components = {}
    for i, t in enumerate(ligand_types):
        for s in ligand_subtypes:
            key = f"{t}|{s}"
            variants.append(key)
            variant_to_components[key] = {'type': t, 'prop': s}
            if max_abs_charge is not None and ligand_charges is not None:
                variant_to_components[key]['chg'] = ligand_charges[i]

    # Load forbid function if needed
    generator_forbid_fn = forbid_fn
    if generator_forbid_fn is None and forbid_fn_name:
        if "." in forbid_fn_name and not forbid_fn_name.endswith(".py"):
            parts = forbid_fn_name.rsplit(".", 1)
            module_path, function_name = parts[0], parts[1]
            module = __import__(module_path, fromlist=[function_name])
            generator_forbid_fn = getattr(module, function_name)
        else:
            import importlib.util
            spec = importlib.util.spec_from_file_location("forbid_module", forbid_fn_name)
            if spec and spec.loader:
                forbid_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(forbid_module)
                generator_forbid_fn = forbid_module.forbid_fn

    # Simply iterate and count - no dict building, no multiprocessing overhead
    count = 0
    for _ in candidate_generator(
        elements, coordination, variants, variant_to_components, max_abs_charge,
        forbid_fn=generator_forbid_fn
    ):
        count += 1
        if progress_interval and count % progress_interval == 0:
            print(f"Counted {count:,} candidates...")
    
    print(f"Total: {count:,} candidates")
    return count


def expand_candidates_with_variants(
    elements,
    coordination,
    ligand_types,
    ligand_subtypes=['W', 'N', 'D'],
    include_variant_counts=False,
    forbid_fn=None,
    forbid_fn_name=None,
    max_abs_charge=None,
    ligand_charges=None,
    n_cores=None,
    output_file=None,
    count_only=True,
    chunksize=1000,
):
    """
    Generate a DataFrame of candidate chemical complexes with ligand variants.
    
    Uses a memory-efficient streaming approach where each structure is generated
    lazily and processed one at a time by parallel workers.

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
    forbid_fn_name : str, optional
        Module path to the forbid function (e.g., "module.submodule.function").
        Used for parallel execution where functions cannot be pickled directly.
    max_abs_charge : int, optional
        Maximum absolute charge allowed for a complex.
    ligand_charges : list[int], optional
        List of charges for each ligand type.
    n_cores : int, optional
        Number of parallel jobs. If None, uses all available cores. If 1, runs serially.
    output_file : str, optional
        Path to a JSONL file to save candidates incrementally.
    chunksize : int, optional
        Number of candidates to process per chunk in parallel mode. Default is 1000.

    Returns
    -------
    pd.DataFrame or None
        DataFrame of candidates, None if output_file used.
    """
    
    if max_abs_charge is not None:
        if ligand_charges is None:
            raise ValueError("ligand_charges must be provided if max_abs_charge is not None.")
        if len(ligand_charges) != len(ligand_types):
            raise ValueError("ligand_charges must map to all ligand_types.")
    else:
        ligand_charges = None

    # Build variant list and mapping
    variants = []
    variant_to_components = {}
    for i, t in enumerate(ligand_types):
        for s in ligand_subtypes:
            key = f"{t}|{s}"
            variants.append(key)
            variant_to_components[key] = {'type': t, 'prop': s}
            if max_abs_charge is not None and ligand_charges is not None:
                variant_to_components[key]['chg'] = ligand_charges[i]

    # Prepare output file
    f_out = None
    if output_file:
        if not output_file.endswith('.jsonl'):
            print("Warning: output_file should end in .jsonl. Writing as JSONL anyway.")
        f_out = open(output_file, 'w')

    rows = []
    count = 0
    processed = 0
    
    # Progress tracking
    def print_progress(n_processed):
        if n_processed > 0 and n_processed % 100000 == 0:
            print(f"Processed {n_processed:,} candidates...")

    # Load forbid function for generator filtering (needed for both serial and parallel)
    generator_forbid_fn = forbid_fn  # Use provided function if available
    if generator_forbid_fn is None and forbid_fn_name:
        # Load from name for parallel mode
        if "." in forbid_fn_name and not forbid_fn_name.endswith(".py"):
            parts = forbid_fn_name.rsplit(".", 1)
            module_path, function_name = parts[0], parts[1]
            module = __import__(module_path, fromlist=[function_name])
            generator_forbid_fn = getattr(module, function_name)
        else:
            import importlib.util
            spec = importlib.util.spec_from_file_location("forbid_module", forbid_fn_name)
            if spec and spec.loader:
                forbid_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(forbid_module)
                generator_forbid_fn = forbid_module.forbid_fn

    try:
        if n_cores == 1:
            # Serial execution - forbid_fn filtering happens in generator
            for args in candidate_generator(
                elements, coordination, variants, variant_to_components, max_abs_charge,
                forbid_fn=generator_forbid_fn
            ):
                elem, ox, cn, geom, lig_multi = args
                
                counts_variant = Counter(lig_multi)
                counts_type = Counter()
                counts_prop = Counter()
                
                charge = ox
                for var_key, ct in counts_variant.items():
                    comp = variant_to_components[var_key]
                    counts_type[comp['type']] += ct
                    counts_prop[comp['prop']] += ct
                    if max_abs_charge is not None:
                        charge += comp.get('chg', 0) * ct
                
                candidate = {
                    'Element': elem,
                    'Ox': ox,
                    'CN': cn,
                    'Geometry': geom,
                    'Charge': charge,
                    'Ligand_multiset_variants': list(lig_multi),
                }
                
                for t in ligand_types:
                    candidate[f'count_type_{t}'] = counts_type.get(t, 0)
                for p in ligand_subtypes:
                    candidate[f'count_prop_{p}'] = counts_prop.get(p, 0)
                if include_variant_counts:
                    for v in variant_to_components:
                        col = f"count_var_{v.replace('|','_')}"
                        candidate[col] = counts_variant.get(v, 0)
                

                
                count += 1
                processed += 1
                print_progress(processed)
                if f_out:
                    f_out.write(json.dumps(candidate) + '\n')
                else:
                    rows.append(candidate)
        else:
            # Parallel execution using Pool.imap_unordered for memory efficiency
            n_workers = n_cores if n_cores else multiprocessing.cpu_count()
            
            with multiprocessing.Pool(
                processes=n_workers,
                initializer=_init_worker,
                initargs=(
                    variant_to_components, 
                    ligand_types, 
                    ligand_subtypes,
                    include_variant_counts, 
                    max_abs_charge
                )
            ) as pool:
                # Generator filters candidates before they're sent to workers
                gen = candidate_generator(
                    elements, coordination, variants, variant_to_components, max_abs_charge,
                    forbid_fn=generator_forbid_fn
                )
                
                for candidate in pool.imap_unordered(
                    _process_single_candidate, gen, chunksize=chunksize
                ):
                    processed += 1
                    print_progress(processed)
                    
                    # All candidates from generator are valid (forbid_fn already applied)
                    if candidate is None:
                        continue
                    
                    count += 1
                    if f_out:
                        f_out.write(json.dumps(candidate) + '\n')
                    else:
                        rows.append(candidate)
    finally:
        if f_out:
            f_out.close()

    print(f"There are {count:,} valid candidates (from {processed:,} generated)")
    
    if output_file:
        print(f"{count:,} candidates saved to {output_file}")
        return None
    
    df = pd.DataFrame(rows)
    df.insert(0, 'cand_id', range(1, len(df) + 1))
    return df


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


if __name__ == "__main__":
    """
    Command-line interface for executing expand_candidates_with_variants.

    This function allows users to run the expand_candidates_with_variants function from the command line.
    """
    parser = argparse.ArgumentParser(
        description="Generate candidate chemical complexes with ligand variants."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to a JSON file containing all input parameters."
    )
    parser.add_argument(
        "--n_cores",
        type=int,
        default=None,
        help="Number of parallel jobs to run. Default is None (uses all available cores)."
    )

    args = parser.parse_args()

    # Load input parameters from JSON file
    with open(args.input_file, 'r') as f:
        input_data = json.load(f)

    # Extract parameters from the input JSON
    n_cores = int(args.n_cores)
    elements = {k: [int(v) for v in vals] if isinstance(vals, list) else [int(vals)] 
                for k, vals in input_data["elements"].items()}
    coordination = {int(k): v for k, v in input_data["coordination"].items()}
    ligand_types = input_data["ligand_types"]
    ligand_subtypes = input_data.get("ligand_subtypes", ['W', 'N', 'D'])
    include_variant_counts = input_data.get("include_variant_counts", False)
    max_abs_charge = input_data.get("max_abs_charge", None)
    ligand_charges = input_data.get("ligand_charges", None)
    if ligand_charges is not None:
        ligand_charges = [int(c) for c in ligand_charges]
    output_file = input_data.get("output_file", None)
    count_only = input_data.get("count_only", True)
    chunksize = input_data.get("chunksize", 1000)
    if "n_cores" in input_data:
        raise ValueError("n_cores must be entered as a command line argument.")

    # Get forbid function name - loading is handled inside expand_candidates_with_variants
    forbid_fn_name = input_data.get("forbid_fn", None)

    # Use fast counting if requested and only counting
    if count_only:
        result = count_candidates(
            elements=elements,
            coordination=coordination,
            ligand_types=ligand_types,
            ligand_subtypes=ligand_subtypes,
            forbid_fn_name=forbid_fn_name,
            max_abs_charge=max_abs_charge,
            ligand_charges=ligand_charges,
        )
        print(f"Total number of structures: {result}")
    else:
        result = expand_candidates_with_variants(
            elements=elements,
            coordination=coordination,
            ligand_types=ligand_types,
            ligand_subtypes=ligand_subtypes,
            include_variant_counts=include_variant_counts,
            forbid_fn_name=forbid_fn_name,
            max_abs_charge=max_abs_charge,
            ligand_charges=ligand_charges,
            n_cores=n_cores,
            output_file=output_file,
            chunksize=chunksize
        )

        if output_file:
            print(f"Candidates saved to {output_file}")
        else:
            print("Candidates generated successfully.")



