#!/bin/bash -l
# The -l above is required to get the full environment with modules

# Set the allocation to be charged for this job
# not required if you have set a default allocation
#SBATCH -A edu17.DD2438

# The name of the script
#SBATCH -J combine

# Email
#SBATCH --mail-type=BEGIN,END

# Only 1 hour wall-clock time will be given to this job
#SBATCH -t 8:00:00

# Number of nodes
#SBATCH --nodes=1

#SBATCH -e error_file_make_datasets_combine_jun17.e
#SBATCH -o output_file_make_datasets_combine_jun17.o

# load the anaconda module
module add cudnn/5.1-cuda-8.0
module load anaconda/py35/4.2.0

# if you need the tensorflow environment:
source activate tensorflow

# add modules
pip install --user -r requirements3.txt

# execute the program
# (on Beskow use aprun instead)
#mpirun -np 1 python make_datasets.py neg
python make_datasets.py combine

# to deactivate the Anaconda environment
source deactivate
