#!/bin/bash

#SBATCH --job-name=architector  ## job name
#SBATCH -p standard              ## use free partition
#SBATCH -t 7-00:00:00
#SBATCH --nodes=1            ## use 1 node, don't ask for multiple
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=32
#SBATCH --account DMOBLEY_LAB
#SBATCH --mem-per-cpu=5G     ## ask for 1Gb memory per CPU
#SBATCH --constraint=fastscratch
#SBATCH -o stdout.txt
#SBATCH -e stderr.txt

OMP_NUM_THREADS=1  # OpenMP threads (used by xTB and other codes)
MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
VECLIB_MAXIMUM_THREADS=1

date
hn=`hostname`
echo "Running job: ${SLURM_JOB_ID} on host $hn"

source ~/.bashrc
mamba activate architector

python -u generate_dataset.py ../2_generate_subset_metadata/a_subset1/selected_design.csv \
                               --ligand-dict ../../ligand_dictionaries/ligands.json \
                               --workers $SLURM_CPUS_ON_NODE 2>&1 | tee log.txt

date

