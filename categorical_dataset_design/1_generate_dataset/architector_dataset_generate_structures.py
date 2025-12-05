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
import concurrent.futures
import json

import pandas as pd


def expand_candidates_with_variants(
    elements,
    coordination,
    ligand_types,
    ligand_subtypes=['W', 'N', 'D'],
    include_variant_counts=False,
    forbid_fn=None,
    max_abs_charge=None,
    ligand_charges=None,
    n_cores=None,
    output_file=None,
    count_only=True,
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
    n_cores : int, optional
        Number of parallel jobs to run. If None, uses default ProcessPoolExecutor behavior.
        If 1, runs serially.
    output_file : str, optional
        Path to a file (JSONL or CSV) to save candidates incrementally. 
        If provided, the function will write results to this file as they are generated
        and return None (or an empty DataFrame) to avoid memory issues.
        JSONL (.jsonl) is recommended for robustness.
    count_only : bool, optional
        If true, the candidates are not recorded, but the resulting structures are counted
        and the total returned

    Returns
    -------
    pd.DataFrame or None or int
        DataFrame containing all valid candidate complexes, or None if output_file is used, or 
        an integer of the number of structures if count_only is used.
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
            raise ValueError("Only jsonl file types are currently accepted.")

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
    total_tasks = sum(1 for _ in task_generator())
    completed_tasks = 0

    def task_progress():
        nonlocal completed_tasks
        completed_tasks += 1
        progress = (completed_tasks / total_tasks) * 100
        if completed_tasks % (total_tasks // 100) == 0:
            print(f"Progress: {progress:.2f}% complete")

    try:
        count = 0
        if n_cores == 1:
            for task in task_generator():
                batch = _generate_candidates_batch(*task)
                task_progress()
                count += len(batch)
                if count_only:
                    pass
                elif f_out:
                    write_batch(batch, f_out, is_jsonl)
                else:
                    rows.extend(batch)
        else:
            # Use ProcessPoolExecutor for parallel execution
            with concurrent.futures.ProcessPoolExecutor(max_workers=n_cores) as executor:
                # Submit all tasks
                futures = [executor.submit(_generate_candidates_batch, *task) for task in task_generator()]
                count = 0
                for future in concurrent.futures.as_completed(futures):
                    batch = future.result()
                    task_progress()
                    count += len(batch)
                    if count_only:
                        pass
                    elif f_out:
                        write_batch(batch, f_out, is_jsonl)
                    else:
                        rows.extend(batch)
    finally:
        if f_out:
            f_out.close()
    if count_only:
        print(f"There are {count} candidates")
        return count
    elif output_file:
        print(f"{count} candidates saved to {output_file}")
        return None

    df = pd.DataFrame(rows)
    print(f"There are {count} candidates")
    df.insert(0, 'cand_id', range(1, len(df)+1))
    return df


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
    if "n_cores" in input_data:
        raise ValueError("n_cores must be entered as a command line argument.")

    # Load forbid function if provided
    forbid_fn = None
    if "forbid_fn" in input_data:
        forbid_fn_path = input_data["forbid_fn"]
        
        # Check if it's a module path (e.g., "module.submodule.function")
        if "." in forbid_fn_path and not forbid_fn_path.endswith(".py"):
            # Import from module path
            parts = forbid_fn_path.rsplit(".", 1)
            module_path = parts[0]
            function_name = parts[1]
            
            module = __import__(module_path, fromlist=[function_name])
            forbid_fn = getattr(module, function_name)
        else:
            # Import from file path
            import importlib.util
            spec = importlib.util.spec_from_file_location("forbid_module", forbid_fn_path)
            forbid_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(forbid_module)
            forbid_fn = forbid_module.forbid_fn


    result = expand_candidates_with_variants(
        elements=elements,
        coordination=coordination,
        ligand_types=ligand_types,
        ligand_subtypes=ligand_subtypes,
        include_variant_counts=include_variant_counts,
        forbid_fn=forbid_fn,
        max_abs_charge=max_abs_charge,
        ligand_charges=ligand_charges,
        n_cores=input_data.get("n_cores", None),
        output_file=output_file,
        count_only=count_only
    )

    if count_only:
        print(f"Total number of structures: {result}")
    elif output_file:
        print(f"Candidates saved to {output_file}")
    else:
        print("Candidates generated successfully.")



