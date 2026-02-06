#!/usr/bin/env python3
"""
Dataset Generation Script for Organometallic Complexes using Architector

This script processes a design CSV file to generate organometallic complexes 
using the Architector package. It supports parallelization with robust
error handling and structure capture for failed cases.

Usage:
    python generate_dataset.py <input_csv> [options]

Example:
    python generate_dataset.py ../2_generate_subset_metadata/a_subset1/selected_design.csv \
        --output-dir results \
        --workers 4
"""

import argparse
import os
import sys
import copy
import time
import traceback
import json
import pandas as pd
from pathlib import Path
from multiprocessing import Pool, Manager
from typing import Optional
from ase import Atoms
import qcelemental as qcel
from multiprocessing import Manager

from architector import build_complex


class DatasetGenerator:
    """
    Main class for generating organometallic complex datasets.
    
    This class handles the generation of large datasets using Architector,
    with robust error handling and preliminary structure capture for failed cases.
    
    Key Features for Error Recovery:
    - Captures initial mol2 structures before optimization (save_init_geos=True)
    - More lenient sanity check parameters to allow more structures to be generated
    - Saves preliminary structures even when final validation fails
    - Memory-efficient storage using only mol2 format
    """
    
    def __init__(self, 
                 input_csv: str,
                 ligand_dict_path: str = "../../ligand_dictionaries/ligands.json",
                 output_dir: str = "dataset_output",
                 output_chunk_size: int = 100000):
        """
        Initialize the dataset generator.
        
        Args:
            input_csv: Path to the input CSV with design specifications
            ligand_dict_path: Path to the ligand dictionary JSON file
            output_dir: Directory to store output files
            output_chunk_size: Maximum number of structures per output file chunk
        """
        self.input_csv = input_csv
        self.ligand_dict_path = ligand_dict_path
        self.output_dir = Path(output_dir)
        self.output_chunk_size = output_chunk_size
        
        # Create output directory structure
        self.output_dir.mkdir(exist_ok=True)
        
        # Load ligand dictionary
        self.ligand_dict = self._load_ligand_dict()
        
        # Initialize data containers
        self.successful_structures = []
        self.error_structures = []
        
        # Chunk tracking
        self.success_chunk_index = 0
        self.error_chunk_index = 0
        self.success_count_in_chunk = 0
        self.error_count_in_chunk = 0
        
    def _load_ligand_dict(self) -> dict:
        """Load and sort the ligand dictionary."""
        with open(self.ligand_dict_path, "r") as f:
            ligands = json.load(f)
        return dict(sorted(ligands.items()))
    
    def _get_success_file_path(self, chunk_index: int) -> Path:
        """Get the path for a success file chunk."""
        return self.output_dir / f"successful_structures_chunk_{chunk_index:04d}.jsonl"
    
    def _get_error_file_path(self, chunk_index: int) -> Path:
        """Get the path for an error file chunk."""
        return self.output_dir / f"error_structures_chunk_{chunk_index:04d}.jsonl"
    
    @staticmethod
    def make_state_key(metal: str, ox_state: int, cn: int, 
                       ligand_labels: list[str], core_type: str) -> str:
        """Create unique key for current state."""
        return f"{metal}_{ox_state}_{cn}_{core_type}_{'_'.join(sorted(ligand_labels))}"
    
    @staticmethod
    def ligand_dicts_from_labels(ligand_labels: list[str], ligand_dict: dict) -> list[dict]:
        """
        Process ligand labels into dicts that are usable by Architector.
        
        Args:
            ligand_labels: List of ligand labels in format "type|subtype"
            ligand_dict: Dictionary mapping ligand types to their definitions
            
        Returns:
            List of ligand dictionaries ready for Architector
        """
        ligand_dicts = []
        for lig_label in ligand_labels:
            lig_type, lig_subtype = lig_label.split("|")
            ligand_dict_copy = copy.deepcopy(ligand_dict[lig_type])
            
            if 'functional_inds' in ligand_dict_copy:
                if lig_subtype != "D":  # if not electron donating
                    if lig_type == "halide":
                        ligand_dict_copy['smiles'] = '[F-]' if lig_subtype == "W" else "[Cl-]"
                    else:
                        ligand_dict_copy['smiles'] = ligand_dict_copy['smiles'].replace(
                            "[H]",
                            '[F]' if lig_subtype == "W" else '[C]([H])([H])[H]'
                        )
                del ligand_dict_copy['functional_inds']
            
            ligand_dicts.append(ligand_dict_copy)
        return ligand_dicts
    
    def _save_successful_structure(self, structure_data: dict):
        """Save a successful structure to the JSONL file with chunking."""
        # Check if we need to start a new chunk
        if self.success_count_in_chunk >= self.output_chunk_size:
            self.success_chunk_index += 1
            self.success_count_in_chunk = 0
            print(f"  → Starting new success chunk {self.success_chunk_index}")
        
        success_file = self._get_success_file_path(self.success_chunk_index)
        with open(success_file, 'a') as f:
            f.write(json.dumps(structure_data) + '\n')
        self.success_count_in_chunk += 1
    
    def _save_error_structure(self, error_data: dict):
        """Save an error structure to the JSONL file with chunking."""
        # Check if we need to start a new chunk
        if self.error_count_in_chunk >= self.output_chunk_size:
            self.error_chunk_index += 1
            self.error_count_in_chunk = 0
            print(f"  → Starting new error chunk {self.error_chunk_index}")
        
        error_file = self._get_error_file_path(self.error_chunk_index)
        with open(error_file, 'a') as f:
            f.write(json.dumps(error_data) + '\n')
        self.error_count_in_chunk += 1
    
    def process_single_structure(self, structure_params: dict, ligand_dict: dict) -> tuple[bool, dict]:
        """
        Process a single structure. This function is designed to be called
        in parallel processes and is now thread-safe.

        Args:
            structure_params: dictionary containing structure parameters
            ligand_dict: ligand dictionary passed explicitly to avoid shared state

        Returns:
            tuple of (success: bool, result_data: dict)
        """
        metal = structure_params["metal"]
        ox_state = structure_params["ox_state"]
        cn = structure_params["cn"]
        core_type = structure_params["core_type"]
        ligand_labels = structure_params["ligand_labels"]
        state_key = structure_params["state_key"]

        try:
            # Create ligand dictionaries
            ligand_dicts = self.ligand_dicts_from_labels(ligand_labels, ligand_dict)

            # Create input dictionary for Architector
            input_dict = {
                "core": {"metal": metal, 'coreCN': cn, "coreType": [core_type]},
                "ligands": ligand_dicts,
                'parameters': {
                    "debug": False,
                    "metal_ox": ox_state,
                    "full_method": "GFN2-xTB",
                    'assemble_method': 'GFN2-xTB',
                    'n_symmetries': 100,
                    'n_conformers': 1,
                    'return_only_1': True,
                    'save_init_geos': True,
                    'assemble_sanity_checks': True,
                    'assemble_graph_sanity_cutoff': 2.0,
                    'assemble_smallest_dist_cutoff': 0.25,
                    'full_sanity_checks': True,
                    'full_graph_sanity_cutoff': 1.8,
                    'full_smallest_dist_cutoff': 0.5,
                    'force_generation': False,
                },
            }

            # Call Architector
            output = build_complex(input_dict)
            if not output:
                raise ValueError("build_complex returned empty result")

            if len(output) > 0:
                print(f"  build_complex completed, {len(output)} options available, taking the first", flush=True)
                result = next(iter(output.values()))
            else:
                raise ValueError("No build_complex output was produced.")

            # Custom sanity checks
            n_el = [result['xtb_n_unpaired_electrons'], result['calc_n_unpaired_electrons']]
            if len(set(n_el)) > 1:
                raise ValueError(f"N_unpaired_el should agree between keys: 'xtb_n_unpaired_electrons', 'calc_n_unpaired_electrons', 'metal_spin', resulting {n_el}")
            
            chg = [result['total_charge'], result['xtb_total_charge']]
            if len(set(chg)) > 1:
                raise ValueError(f"Charge should agree between keys: 'total_charge', 'xtb_total_charge', resulting {chg}")
            
            if ox_state != result["metal_ox"]:
                raise ValueError("Metal oxidation state is not equal to assigned.")

            success_data = {
                'state_key': state_key,
                'metal': metal,
                'oxidation_state': ox_state,
                'coordination_number': cn,
                'geometry': core_type,
                'ligand_labels': ligand_labels,
                'total_charge': result['total_charge'],
                'multiplicity': result['xtb_n_unpaired_electrons'] + 1,
                'xtb_energy': result['energy'],
                'mol2_string': result['mol2string'],
                'init_mol2_string': result.get('init_mol2string', ''),
                'timestamp': time.time()
            }

            return True, success_data
        except Exception as e:
            error_data = {
                'state_key': state_key,
                'metal': metal,
                'oxidation_state': ox_state,
                'coordination_number': cn,
                'geometry': core_type,
                'ligand_labels': ligand_labels,
                'error_type': type(e).__name__,
                'error_message': str(e),
                'traceback': traceback.format_exc(),
                'timestamp': time.time()
            }

            return False, error_data

    def safe_process(self, params, processed_state_keys):
        """
        Safely process a structure, handling errors gracefully.

        Args:
            params (dict): Structure parameters.
            processed_state_keys (dict): Shared dictionary to track processed state keys.

        Returns:
            tuple: (success, result_data or error_data)
        """
        state_key = params['state_key']
        if state_key in processed_state_keys:
            print(f"Skipping already processed state_key: {state_key}")
            return False, {}

        try:
            success, result_data = self.process_single_structure(params, self.ligand_dict)
            processed_state_keys[state_key] = success
            return success, result_data
        except Exception as e:
            print(f"Error processing state_key {state_key}: {e}")
            traceback.print_exc()
            error_data = {
                'state_key': state_key,
                'error_type': type(e).__name__,
                'error_message': str(e),
                'traceback': traceback.format_exc(),
                'timestamp': time.time()
            }
            return False, error_data

    def generate_dataset(self, n_workers: int = 1) -> None:
        """
        Generate the complete dataset.
        
        Args:
            n_workers: Number of parallel workers to use
        """
        print("Loading input data...")
        
        # Load input CSV
        df = pd.read_csv(self.input_csv, converters={'Ligand_multiset_variants': eval})
        print(f"Loaded {len(df)} structures from {self.input_csv}")
        
        # Prepare structure parameters for processing
        structure_params_list = []
        for i, row in df.iterrows():
            row_dict = row.to_dict()
            metal = row_dict["Element"]
            ox_state = row_dict["Ox"]
            cn = row_dict["CN"]
            core_type = row_dict["Geometry"]
            ligand_labels = row_dict["Ligand_multiset_variants"]
            state_key = self.make_state_key(metal, ox_state, cn, ligand_labels, core_type)
            
            structure_params = {
                'metal': metal,
                'ox_state': ox_state,
                'cn': cn,
                'core_type': core_type,
                'ligand_labels': ligand_labels,
                'state_key': state_key
            }
            structure_params_list.append(structure_params)
        
        print(f"Processing {len(structure_params_list)} structures with {n_workers} workers...")
        
        # Shared state_keys set to prevent duplicate work
        manager = Manager()
        processed_state_keys = manager.dict()
        
        # Process structures
        successful_count = 0
        error_count = 0
        
        if n_workers == 1:
            # Sequential processing
            for params in structure_params_list:
                result = self.safe_process(params, processed_state_keys)
                if result:
                    success, result_data = result
                    self._handle_result(success, result_data)
                    successful_count += success
                    error_count += not success
        else:
            # Parallel processing using multiprocessing.Pool
            with Pool(processes=n_workers) as pool:
                results = pool.starmap(self.safe_process, [(params, processed_state_keys) for params in structure_params_list])

                for result in results:
                    if result:
                        success, result_data = result
                        self._handle_result(success, result_data)
                        successful_count += success
                        error_count += not success

        print(f"\nDataset generation completed!")
        print(f"Successful structures: {successful_count}")
        print(f"Failed structures: {error_count}")
        print(f"Total processed: {successful_count + error_count}")

        # Create final dataset CSV
        self._create_final_dataset()
    
    def _handle_result(self, success: bool, result_data: dict):
        """Handle the result of processing a single structure."""
        if success:
            self._save_successful_structure(result_data)
            self.successful_structures.append(result_data)
        else:
            self._save_error_structure(result_data)
            self.error_structures.append(result_data)
    
    def _create_final_dataset(self):
        """Create final CSV dataset from all successful structure chunks."""
        # Find all success chunk files
        success_files = sorted(self.output_dir.glob("successful_structures_chunk_*.jsonl"))
        
        if not success_files:
            print("No successful structures found")
            return
        
        print(f"Reading {len(success_files)} success chunk file(s)...")
        
        # Read all successful structures from all chunks
        successful_data = []
        for success_file in success_files:
            with open(success_file, 'r') as f:
                for line in f:
                    successful_data.append(json.loads(line))
        
        if not successful_data:
            print("No successful structures found")
            return
        
        # Create DataFrame
        df_columns = [
            'metal', 'oxidation_state', 'coordination_number', 'geometry',
            'ligand_labels', 'total_charge', 'multiplicity', 'xtb_energy'
        ]
        
        df_data = {}
        for col in df_columns:
            df_data[col] = [structure[col] for structure in successful_data]
        
        df_final = pd.DataFrame(df_data)
        final_dataset_path = self.output_dir / "final_dataset.csv"
        df_final.to_csv(final_dataset_path, index=False)
        print(f"Final dataset saved to {final_dataset_path}")
        print(f"Dataset contains {len(df_final)} successful structures")


def main(input_csv: str, ligand_dict_path: str, output_dir: str, workers: int, output_chunk_size: int):
    """
    Main function to generate the dataset.

    Args:
        input_csv: Path to the input CSV file with design specifications.
        ligand_dict_path: Path to the ligand dictionary JSON file.
        output_dir: Directory to store output files.
        workers: Number of parallel workers to use.
        output_chunk_size: Maximum number of structures per output file chunk.
    """
    # Create and run dataset generator
    generator = DatasetGenerator(
        input_csv=input_csv,
        ligand_dict_path=ligand_dict_path,
        output_dir=output_dir,
        output_chunk_size=output_chunk_size
    )

    print(f"Starting dataset generation...")
    print(f"Input: {input_csv}")
    print(f"Output directory: {output_dir}")
    print(f"Workers: {workers}")
    print(f"Output chunk size: {output_chunk_size} structures per file")

    start_time = time.time()

    try:
        generator.generate_dataset(n_workers=workers)
    except KeyboardInterrupt:
        print("\nInterrupted by user. Progress has been saved.")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\nTotal time: {elapsed:.2f} seconds ({elapsed/60:.1f} minutes)")


def mol2_to_ase(mol2_string: str) -> Optional['Atoms']:
    """
    Convert mol2 string to ASE Atoms object.
    
    This function can be used to convert mol2 strings from the dataset
    to ASE Atoms objects for further analysis.
    
    Args:
        mol2_string: mol2 format string
        
    Returns:
        ASE Atoms object or None if conversion fails
        
    Example:
        ```python
        import json
        from generate_dataset import mol2_to_ase
        
        # Load a structure from the dataset
        with open('dataset_output/successful_structures.jsonl', 'r') as f:
            structure = json.loads(f.readline())
        
        # Convert to ASE
        atoms = mol2_to_ase(structure['mol2_string'])
        if atoms:
            print(f"Loaded molecule with {len(atoms)} atoms")
        ```
    """
    try:
        # Parse mol2 string to extract atomic information
        lines = mol2_string.split('\n')
        atom_section = False
        atoms = []
        positions = []
        
        for line in lines:
            if '@<TRIPOS>ATOM' in line:
                atom_section = True
                continue
            elif '@<TRIPOS>' in line and atom_section:
                break
            elif atom_section and line.strip():
                parts = line.split()
                if len(parts) >= 6:
                    symbol = parts[5].split('.')[0]  # Remove atom type suffix
                    x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                    atoms.append(symbol)
                    positions.append([x, y, z])
        
        if atoms and positions:
            return Atoms(symbols=atoms, positions=positions)
    except Exception as e:
        print(f"Warning: Failed to convert mol2 to ASE: {e}")
    
    return None


def mol2_to_qcelemental(mol2_string: str) -> Optional[dict]:
    """
    Convert mol2 string to QCElemental molecule dictionary.
    
    This function can be used to convert mol2 strings from the dataset
    to QCElemental molecule objects for quantum chemistry workflows.
    
    Args:
        mol2_string: mol2 format string
        
    Returns:
        QCElemental molecule dictionary or None if conversion fails
        
    Example:
        ```python
        import json
        from generate_dataset import mol2_to_qcelemental
        
        # Load a structure from the dataset
        with open('dataset_output/successful_structures.jsonl', 'r') as f:
            structure = json.loads(f.readline())
        
        # Convert to QCElemental
        qce_mol_dict = mol2_to_qcelemental(structure['mol2_string'])
        if qce_mol_dict:
            # Reconstruct QCElemental molecule
            import qcelemental as qcel
            qce_mol = qcel.models.Molecule(**qce_mol_dict)
            print(f"Loaded molecule: {qce_mol.molecular_formula}")
        ```
    """
    # First convert to ASE, then to QCElemental
    ase_mol = mol2_to_ase(mol2_string)
    if ase_mol is None:
        return None
        
    try:
        qce_mol = qcel.models.Molecule(
            symbols=list(ase_mol.get_chemical_symbols()),
            geometry=ase_mol.get_positions().flatten(),
            connectivity=None  # Could be enhanced to include connectivity
        )
        return qce_mol.dict()
    except Exception as e:
        print(f"Warning: Failed to convert to QCElemental: {e}")
    
    return None


def load_dataset_as_ase(output_dir: str) -> list[tuple[dict, 'Atoms']]:
    """
    Load entire dataset from chunked files and convert all mol2 strings to ASE Atoms objects.
    
    Args:
        output_dir: Path to the output directory containing chunked JSONL files
        
    Returns:
        List of (metadata_dict, ase_atoms) tuples
        
    Example:
        ```python
        from generate_dataset import load_dataset_as_ase
        
        # Load all structures as ASE objects
        dataset = load_dataset_as_ase('dataset_output')
        
        for metadata, atoms in dataset:
            print(f"Metal: {metadata['metal']}, Atoms: {len(atoms)}")
        ```
    """
    from pathlib import Path
    output_path = Path(output_dir)
    
    # Find all success chunk files
    success_files = sorted(output_path.glob("successful_structures_chunk_*.jsonl"))
    
    dataset = []
    for success_file in success_files:
        with open(success_file, 'r') as f:
            for line in f:
                structure = json.loads(line)
                atoms = mol2_to_ase(structure['mol2_string'])
                if atoms:
                    # Remove mol2_string from metadata to save memory
                    metadata = {k: v for k, v in structure.items() if k != 'mol2_string'}
                    dataset.append((metadata, atoms))
    
    return dataset


def load_dataset_as_qcelemental(output_dir: str) -> list[tuple[dict, dict]]:
    """
    Load entire dataset from chunked files and convert all mol2 strings to QCElemental molecules.
    
    Args:
        output_dir: Path to the output directory containing chunked JSONL files
        
    Returns:
        List of (metadata_dict, qcelemental_dict) tuples
        
    Example:
        ```python
        from generate_dataset import load_dataset_as_qcelemental
        import qcelemental as qcel
        
        # Load all structures as QCElemental objects
        dataset = load_dataset_as_qcelemental('dataset_output')
        
        for metadata, qce_dict in dataset:
            qce_mol = qcel.models.Molecule(**qce_dict)
            print(f"Metal: {metadata['metal']}, Formula: {qce_mol.molecular_formula}")
        ```
    """
    from pathlib import Path
    output_path = Path(output_dir)
    
    # Find all success chunk files
    success_files = sorted(output_path.glob("successful_structures_chunk_*.jsonl"))
    
    dataset = []
    for success_file in success_files:
        with open(success_file, 'r') as f:
            for line in f:
                structure = json.loads(line)
                qce_dict = mol2_to_qcelemental(structure['mol2_string'])
                if qce_dict:
                    # Remove mol2_string from metadata to save memory
                    metadata = {k: v for k, v in structure.items() if k != 'mol2_string'}
                    dataset.append((metadata, qce_dict))
    
    return dataset


def make_state_key(metal: str, ox_state: int, cn: int, 
                   ligand_labels: list[str], core_type: str) -> str:
    """
    Convenience function to create unique state key without class instantiation.
    
    Args:
        metal: Metal element symbol
        ox_state: Oxidation state
        cn: Coordination number
        ligand_labels: List of ligand labels
        core_type: Core geometry type
        
    Returns:
        Unique string identifier for the structure state
    """
    return DatasetGenerator.make_state_key(metal, ox_state, cn, ligand_labels, core_type)


def ligand_dicts_from_labels(ligand_labels: list[str], ligand_dict_path: str = "../../ligand_dictionaries/ligands.json") -> list[dict]:
    """
    Convenience function to process ligand labels without class instantiation.
    
    Args:
        ligand_labels: List of ligand labels in format "type|subtype"
        ligand_dict_path: Path to ligand dictionary JSON file
        
    Returns:
        List of ligand dictionaries ready for Architector
        
    Example:
        ligands = ligand_dicts_from_labels(["halide|W", "water|D"])
    """
    with open(ligand_dict_path, "r") as f:
        ligand_dict = json.load(f)
    ligand_dict = dict(sorted(ligand_dict.items()))
    
    return DatasetGenerator.ligand_dicts_from_labels(ligand_labels, ligand_dict)


if __name__ == "__main__":
    # Command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Generate organometallic complex dataset using Architector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    python generate_dataset.py ../2_generate_subset_metadata/a_subset1/selected_design.csv

    # With parallel processing and custom output
    python generate_dataset.py input.csv --workers 4 --output-dir my_results
    
    # Large dataset with custom chunk size (50k structures per file)
    python generate_dataset.py input.csv --workers 64 --output-chunk-size 50000
        """
    )

    parser.add_argument(
        "input_csv",
        help="Path to input CSV file with design specifications"
    )

    parser.add_argument(
        "--ligand-dict",
        default="../../ligand_dictionaries/ligands.json",
        help="Path to ligand dictionary JSON file (default: ../../ligand_dictionaries/ligands.json)"
    )

    parser.add_argument(
        "--output-dir",
        default="dataset_output",
        help="Output directory for results (default: dataset_output)"
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)"
    )

    parser.add_argument(
        "--output-chunk-size",
        type=int,
        default=100000,
        help="Maximum number of structures per output file chunk (default: 100000)"
    )

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.input_csv):
        print(f"Error: Input CSV file not found: {args.input_csv}")
        sys.exit(1)

    if not os.path.exists(args.ligand_dict):
        print(f"Error: Ligand dictionary not found: {args.ligand_dict}")
        sys.exit(1)

    if args.workers < 1:
        print("Error: Number of workers must be at least 1")
        sys.exit(1)

    if args.workers > 1:
        # Warn about potential issues with parallel processing
        print(f"Warning: Using {args.workers} parallel workers.")
        print("Note: Architector may have threading issues. Monitor for stability.")

    # Call the main function with parsed arguments
    main(
        input_csv=args.input_csv,
        ligand_dict_path=args.ligand_dict,
        output_dir=args.output_dir,
        workers=args.workers,
        output_chunk_size=args.output_chunk_size
    )