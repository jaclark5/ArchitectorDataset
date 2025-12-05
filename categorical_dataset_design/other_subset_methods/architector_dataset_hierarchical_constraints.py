"""
Nested Categorical Design of Experiments for Chemical Complexes
Uses covering arrays with hierarchical constraints

Greedy Covering Array Algorithm: Efficiently covers the parameter space with minimal experiments while respecting nested dependencies

These are specifically designed for situations where factor levels depend on settings of other factors, which matches your system perfectly:

Oxidation state → depends on element
Geometry → depends on coordination number
Ligand combinations → constrained by coordination number

"""

import itertools
from typing import Dict, List, Tuple, Set
import random
from collections import defaultdict
import pandas as pd

class NestedCategoricalDOE:
    """
    Creates optimal experimental designs for categorical factors with nested dependencies.
    Uses greedy covering array algorithm to minimize experiments while maximizing coverage.
    """
    
    def __init__(self):
        # Define chemical space with nested dependencies
        self.chemical_space = {
            'Fe': {
                'oxidation_states': [2, 3],
                'coordination_numbers': {
                    4: ['tetrahedral', 'square_planar'],
                    5: ['trigonal_bipyramidal', 'square_pyramidal'],
                    6: ['octahedral']
                }
            },
            'Co': {
                'oxidation_states': [2, 3],
                'coordination_numbers': {
                    4: ['tetrahedral', 'square_planar'],
                    5: ['trigonal_bipyramidal', 'square_pyramidal'],
                    6: ['octahedral']
                }
            },
            'Ni': {
                'oxidation_states': [2],
                'coordination_numbers': {
                    4: ['tetrahedral', 'square_planar'],
                    5: ['trigonal_bipyramidal', 'square_pyramidal'],
                    6: ['octahedral']
                }
            },
            'Cu': {
                'oxidation_states': [1, 2],
                'coordination_numbers': {
                    4: ['tetrahedral', 'square_planar'],
                    5: ['trigonal_bipyramidal', 'square_pyramidal'],
                    6: ['octahedral']
                }
            }
        }
        
        # Ligand scaffolds - coordination sites marked with *
        self.ligand_scaffolds = [
            '*C(*)(*)(*)',  # Carbon center with 3 functionalizable positions
            '*N(*)*',       # Nitrogen center with 2 functionalizable positions
            '*O*',          # Oxygen center with 1 functionalizable position
            '*P(*)(*)(*)',  # Phosphorus center with 3 functionalizable positions
        ]
        
        # Functional groups for substitution
        self.functional_groups = {
            'electron_withdrawing': ['F', 'CF3', 'NO2', 'CN'],
            'neutral': ['H', 'CH3', 'Ph'],
            'electron_donating': ['OH', 'OCH3', 'NH2', 'N(CH3)2']
        }
        
        self.electronic_properties = ['electron_withdrawing', 'neutral', 'electron_donating']
    
    def generate_ligand_library(self) -> List[Tuple[str, str]]:
        """
        Generate library of ligands with their electronic properties.
        Returns: List of (ligand_scaffold, electronic_property) tuples
        """
        ligand_library = []
        for scaffold in self.ligand_scaffolds:
            for prop in self.electronic_properties:
                ligand_library.append((scaffold, prop))
        return ligand_library
    
    def generate_valid_combinations(self) -> List[Dict]:
        """
        Generate all valid combinations respecting nested dependencies.
        Returns list of valid experimental configurations.
        """
        valid_combinations = []
        
        for element, element_data in self.chemical_space.items():
            for ox_state in element_data['oxidation_states']:
                for cn, geometries in element_data['coordination_numbers'].items():
                    for geometry in geometries:
                        # Add this valid combination
                        valid_combinations.append({
                            'element': element,
                            'oxidation_state': ox_state,
                            'coordination_number': cn,
                            'geometry': geometry
                        })
        
        return valid_combinations
    
    def generate_ligand_combinations(self, coordination_number: int, 
                                    with_replacement: bool = True) -> List[Tuple]:
        """
        Generate ligand combinations for a given coordination number.
        
        Args:
            coordination_number: Number of ligands to select
            with_replacement: Allow same ligand type multiple times
        
        Returns:
            List of ligand combinations (each as tuple of (scaffold, property) pairs)
        """
        ligand_library = self.generate_ligand_library()
        
        if with_replacement:
            # Combinations with replacement - order doesn't matter
            combinations = list(itertools.combinations_with_replacement(
                ligand_library, coordination_number
            ))
        else:
            # Combinations without replacement
            combinations = list(itertools.combinations(
                ligand_library, coordination_number
            ))
        
        return combinations
    
    def compute_pairwise_interactions(self, design_point: Dict) -> Set[Tuple]:
        """
        Compute all pairwise interactions for a design point.
        Used for coverage analysis.
        """
        interactions = set()
        keys = list(design_point.keys())
        
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                key1, key2 = keys[i], keys[j]
                val1, val2 = design_point[key1], design_point[key2]
                # Create hashable interaction tuple
                interaction = tuple(sorted([(key1, str(val1)), (key2, str(val2))]))
                interactions.add(interaction)
        
        return interactions
    
    def greedy_covering_array(self, valid_combinations: List[Dict],
                             target_size: int = None,
                             strength: int = 2,
                             max_cn: int = 6) -> List[Dict]:
        """
        Generate a covering array using greedy algorithm.
        
        Args:
            valid_combinations: List of valid metal-geometry combinations
            target_size: Target number of experiments (None for automatic)
            strength: Coverage strength (2 = pairwise, 3 = 3-way)
            max_cn: Maximum coordination number to consider
        
        Returns:
            List of complete design points including ligand combinations
        """
        # Generate full design space with ligands
        full_design_space = []
        
        for combo in valid_combinations:
            cn = combo['coordination_number']
            if cn > max_cn:
                continue
            
            # Get subset of ligand combinations (sample for efficiency)
            ligand_combos = self.generate_ligand_combinations(cn, with_replacement=True)
            
            # Sample ligand combinations if too many
            if len(ligand_combos) > 100:
                ligand_combos = random.sample(ligand_combos, 100)
            
            for lig_combo in ligand_combos:
                # Extract electronic properties from ligand combination
                electronic_props = [prop for scaffold, prop in lig_combo]
                
                design_point = combo.copy()
                design_point['ligands'] = lig_combo
                design_point['ligand_electronic_distribution'] = tuple(sorted(electronic_props))
                
                full_design_space.append(design_point)
        
        print(f"Full design space size: {len(full_design_space)}")
        
        # Compute all possible interactions we want to cover
        all_interactions = set()
        for point in full_design_space:
            all_interactions.update(self.compute_pairwise_interactions(point))
        
        print(f"Total pairwise interactions to cover: {len(all_interactions)}")
        
        # Greedy selection
        selected_designs = []
        uncovered_interactions = all_interactions.copy()
        
        if target_size is None:
            # Estimate target size based on factors
            target_size = min(50, len(full_design_space) // 10)
        
        # Greedy algorithm
        available_points = full_design_space.copy()
        
        while uncovered_interactions and len(selected_designs) < target_size and available_points:
            # Find point that covers most uncovered interactions
            best_point = None
            best_coverage = 0
            best_idx = -1
            
            for idx, point in enumerate(available_points):
                point_interactions = self.compute_pairwise_interactions(point)
                new_coverage = len(point_interactions & uncovered_interactions)
                
                if new_coverage > best_coverage:
                    best_coverage = new_coverage
                    best_point = point
                    best_idx = idx
            
            if best_point is None:
                break
            
            # Add best point to design
            selected_designs.append(best_point)
            uncovered_interactions -= self.compute_pairwise_interactions(best_point)
            available_points.pop(best_idx)
            
            if len(selected_designs) % 5 == 0:
                coverage_pct = 100 * (1 - len(uncovered_interactions) / len(all_interactions))
                print(f"Selected {len(selected_designs)} points, coverage: {coverage_pct:.1f}%")
        
        final_coverage = 100 * (1 - len(uncovered_interactions) / len(all_interactions))
        print(f"\nFinal design size: {len(selected_designs)}")
        print(f"Final coverage: {final_coverage:.1f}%")
        
        return selected_designs
    
    def design_to_dataframe(self, design: List[Dict]) -> pd.DataFrame:
        """Convert design to pandas DataFrame for easy viewing."""
        rows = []
        for point in design:
            row = {
                'element': point['element'],
                'oxidation_state': point['oxidation_state'],
                'coordination_number': point['coordination_number'],
                'geometry': point['geometry'],
                'ligand_electronic_dist': point['ligand_electronic_distribution']
            }
            # Add individual ligands
            for i, (scaffold, prop) in enumerate(point['ligands'], 1):
                row[f'ligand_{i}_scaffold'] = scaffold
                row[f'ligand_{i}_property'] = prop
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def generate_design(self, target_size: int = 30, max_cn: int = 6) -> pd.DataFrame:
        """
        Main method to generate experimental design.
        
        Args:
            target_size: Target number of experiments
            max_cn: Maximum coordination number to include
        
        Returns:
            DataFrame with experimental design
        """
        print("Generating valid metal-geometry combinations...")
        valid_combos = self.generate_valid_combinations()
        print(f"Valid combinations: {len(valid_combos)}")
        
        print("\nGenerating covering array design...")
        design = self.greedy_covering_array(valid_combos, target_size=target_size, max_cn=max_cn)
        
        print("\nConverting to DataFrame...")
        df = self.design_to_dataframe(design)
        
        return df


# Example usage
if __name__ == "__main__":
    # Create DOE object
    doe = NestedCategoricalDOE()
    
    # Generate design with 25 experiments, max coordination number 6
    design_df = doe.generate_design(target_size=25, max_cn=6)
    
    # Display results
    print("\n" + "="*80)
    print("EXPERIMENTAL DESIGN")
    print("="*80)
    print(design_df.to_string())
    
    # Save to CSV
    design_df.to_csv('chemical_complex_design.csv', index=False)
    print("\nDesign saved to 'chemical_complex_design.csv'")
    
    # Summary statistics
    print("\n" + "="*80)
    print("DESIGN SUMMARY")
    print("="*80)
    print(f"Total experiments: {len(design_df)}")
    print(f"\nElements: {design_df['element'].value_counts().to_dict()}")
    print(f"\nGeometries: {design_df['geometry'].value_counts().to_dict()}")
    print(f"\nCoordination numbers: {design_df['coordination_number'].value_counts().to_dict()}")