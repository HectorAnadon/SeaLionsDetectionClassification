#!/bin/bash -l
# The -l above is required to get the full environment with modules

# Set the allocation to be charged for this job
# not required if you have set a default allocation
#SBATCH -A edu17.DD2424

# The name of the script is myjob
#SBATCH -J myjob

# Only 1 hour wall-clock time will be given to this job
#SBATCH -t 1:00:00

# Number of nodes
#SBATCH --nodes=1

#SBATCH -e error_file.e
#SBATCH -o output_file.o

# load the anaconda module
module add cudnn/5.1-cuda-8.0
module load anaconda/py35/4.2.0

# if you need the tensorflow environment:
source activate tensorflow

# add modules
pip install --user -r requirements3.txt

# execute the program
# (on Beskow use aprun instead)
mpirun -np 1 python3 some_script.py

# to deactivate the Anaconda environment
source deactivate