#!/bin/bash -l
# # Resource request
#SBATCH --job-name=UCFR_eval
#SBATCH -p gpu-mxian
#SBATCH --gres=gpu:1
#SBATCH --nodelist=node01
#SBATCH --mem=32G
##SBATCH --mail-type=all              # send mail when job begins and ends
#SBATCH --mail-user=danq7270@vandals.uidaho.edu  # TODO: change this to your mailaddress!
#SBATCH --output=./logs/SBD_vit_base_edge_model_50_epochs_05_05-%x-%j.txt     # where to write output, %x give job name, %j names job id
#SBATCH --error=./logs/SBD_vit_base_edge_model_50_epochs_05_05-%x-%j.err      # where to write slurm error

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

python ./scripts/evaluate_model.py NoBRS \
    --gpus=0 \
    --checkpoint=./weights/SBD_vit_base_edge_model_50_epochs.pth \
    --datasets=GrabCut,Berkeley,DAVIS,BraTS,OAIZIB,ssTEM,COCO_MVal,PascalVOC,SBD \
    --print-ious \
    --iou-analysis \
    --thresh=0.5
    

#--cf-n=1 \
#--cf-click \
#--iou-analysis \
#--print-ious
# cf-n: CFR steps
# cf-click: UCFR
# acf: adaptive CFR

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
