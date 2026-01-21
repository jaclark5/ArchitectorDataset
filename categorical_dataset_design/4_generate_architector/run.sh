#!/bin/bash

OMP_NUM_THREADS=1  # OpenMP threads (used by xTB and other codes)
MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
VECLIB_MAXIMUM_THREADS=1

python -u generate_dataset.py ../2_generate_subset_metadata/a_subset1/selected_design.csv \
                               --ligand-dict ../../ligand_dictionaries/ligands.json \
                               --workers $SLURM_CPUS_ON_NODE 2>&1 | tee log.txt


