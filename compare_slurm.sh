#!/bin/bash -l
#SBATCH --partition hpc_bigmem_a
#SBATCH -c 192
#SBATCH --mem 1028G
#SBATCH --time 1-0:0:0

micromamba activate sptnano_10
./notebook_cmds.py
