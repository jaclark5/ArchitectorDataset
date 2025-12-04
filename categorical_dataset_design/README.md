Based on the two files provided, here is a comparison of the methods, their suitability for your specific chemical problem, and answers to your specific questions.

1. Preferred Method for Categorical Factors with Nested Dependencies
For chemical datasets where the goal is usually to train a regression model (e.g., predicting stability, energy, or gap) or to sample the space for diversity:

D-Optimal (architector_dataset_d-optimal_design.py) is generally preferred for scientific modeling.

Why: It selects data points that maximize the determinant of the information matrix (
$X^T X$). This minimizes the variance of the regression coefficients. In plain English, it picks the set of molecules that makes your future statistical model as accurate as possible.
Nesting: It handles nesting implicitly. By generating only valid candidates (via expand_candidates_with_variants), the algorithm only optimizes over chemically feasible structures.
Covering Arrays (architector_dataset_hierarchical_constraints.py) are preferred for screening/bug-finding.

Why: This method guarantees that every pair of parameters (e.g., Fe + Octahedral) appears together at least once. This is standard in software testing to find "bugs" caused by specific interactions.
Nesting: It handles nesting explicitly in the generation loops.
Recommendation: If you plan to fit a linear model or machine learning model to your data later, use architector_dataset_d-optimal_design.py (D-Optimal). If you simply want to ensure you have "seen" every combination of pairs at least once without a specific model in mind, use architector_dataset_hierarchical_constraints.py.

2. Does architector_dataset_hierarchical_constraints.py function well with a chosen number of data points?
Yes, but with a caveat.

The greedy_covering_array function accepts a target_size.
How it works: It adds the "best" point (the one that covers the most previously unseen interactions) iteratively until it hits target_size.
Limitation: If target_size is too small, the algorithm will stop before achieving 100% coverage of all pairwise interactions. If target_size is very large, it may run out of unique interactions to cover and simply stop adding useful information, potentially resulting in fewer points than requested if the space is small.

3. How can coverage be tested between these methods?
You can measure "2-way coverage" (percentage of all valid pairwise combinations present in the design).

architector_dataset_hierarchical_constraints.py already contains the logic to do this. You can adapt the compute_pairwise_interactions method to test the output of architector_dataset_d-optimal_design.py.

1. Generate the full set of all possible valid pairs (Interaction Set $S_{total}$).
2. For the selected design rows, generate their pairs (Interaction Set $S_{design}$).
3. Coverage % = $\frac{∣S_{design}∣}{∣S_{total}∣} × 100$.
4. Is one faster than another?
Small/Medium Spaces: architector_dataset_d-optimal_design.py (D-Optimal) is often faster because it uses numpy linear algebra, which is highly optimized.
Massive Spaces: architector_dataset_hierarchical_constraints.py may be faster or more memory efficient.
  - architector_dataset_d-optimal_design.py generates every possible candidate in memory first (expand_candidates_with_variants). If you have high coordination numbers and many ligands, this creates a combinatorial explosion (millions of rows) that can crash memory before selection begins.
  - architector_dataset_hierarchical_constraints.py includes a sampling step (if len(ligand_combos) > 100: random.sample...) which prevents the candidate list from exploding, making it safer for very large combinatorial spaces.
5. How can the number of data points be controlled in architector_dataset_d-optimal_design.py?
In architector_dataset_d-optimal_design.py, the number of data points is controlled by the n_runs argument passed to the d_optimal_greedy function.

In the if __name__ == '__main__': block at the bottom of the file, look for the variable `desired`.

If you set desired = 50, the algorithm will select the 50 statistically best candidates from the pool.