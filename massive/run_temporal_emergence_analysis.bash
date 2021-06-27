#!/bin/bash
# Usage: sbatch slurm-serial-job-script
# Prepared By: Kai Xi,  Oct 2014
#              help@massive.org.au
# NOTE: To activate a SLURM option, remove the whitespace between the '#' and 'SBATCH'
# $1: line counter
# Need to use variables OUTSIDE of this script, #SBATCH doesn't support variables: https://help.rc.ufl.edu/doc/Using_Variables_in_SLURM_Jobs

#SBATCH --job-name=compute_phis

# To set a project account for credit charging, 
#SBATCH --account=ot95

# Request CPU resource for a serial job
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
# SBATCH --cpus-per-task=5

# Memory usage (MB)
#SBATCH --mem-per-cpu=1000

# Set your minimum acceptable walltime, format: day-hours:minutes:seconds
#SBATCH --time=0-08:00:00

# SBATCH --qos=shortq
# SBATCH --partition=short,comp

# To receive an email when job completes or fails
# SBATCH --mail-user=mmas0026@student.monash.edu
# SBATCH --mail-type=END
# SBATCH --mail-type=FAIL

# Set the file for output (stdout)
#SBATCH --output=compute.out

# Set the file for error log (stderr)
#SBATCH --error=compute.err

# Use reserved node to run job when a node reservation is made for you already

# Job script

# Compute phis
module load python/3.6.2
source /home/mmasques/ot95/MarcelMasque/TemporalEmergence/bin/activate
python run_temporal_emergence_analysis.py 20
deactivate