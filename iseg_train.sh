#!/bin/bash -l
# # Resource request
#SBATCH --job-name=iseg_train
#SBATCH -p gpu-mxian
#SBATCH --gres=gpu:1
#SBATCH --nodelist=node01
#SBATCH --mem=32G
#SBATCH --output=./logs/sbd_vit_base_55_epochs-%x-%j.txt     # where to write output, %x give job name, %j names job id
#SBATCH --error=./logs/sbd_vit_base_55_epochs-%x-%j.err      # where to write slurm error

# change directory to directory we submit job from:
cd $SLURM_SUBMIT_DIR

# Setup conda:
echo "Beginning to initialize SEM environment"

source /lfs/danq7270.ui/miniconda3/etc/profile.d/conda.sh
conda activate iseg

echo "pytorch environment activated"

echo "Loading  python executable file"

echo "Current node: ${SLURM_NODELIST}"

START=$(date +%s)


python ./train.py models/plainvit_base448_sbd.py \
    --batch-size=140 \
    --ngpus=4 \
    --workers=8 \
    --epochs=55 \
    --exp-name=sbd_vit_base_55_epochs


let RUNTIME=$(date +%s)-$START
echo "Training time: $RUNTIME"
echo "*--done--*"

# Printing out some info.
echo "Home directory: ${HOME}"

echo "Submitted job with sbatch from directory: ${SLURM_SUBMIT_DIR}"

echo "Working directory: $PWD"

# Print out values of the current jobs SLURM environment variables
env | grep SLURM

# Print out final statistics about resource use before job exits                                                                              
scontrol show job ${SLURM_JOB_ID}

### Run script with command: 
### 'sbatch test.sh' 
