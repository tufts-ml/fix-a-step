#!/bin/bash
#
# Usage
# -----
# $ bash launch_experiments.sh ACTION_NAME
#
# where ACTION_NAME is either 'list' or 'submit' or 'run_here'

if [[ -z $1 ]]; then
    ACTION_NAME='list'
else
    ACTION_NAME=$1
fi

export normalization='unnormalized_HWC'

export script="$ROOT_PATH/src_CIFAR10/build_datasets/CIFAR10Contamination_ssl/$normalization/build_dataset.py"

export data_rootdir="$ROOT_PATH/ML_DATA/"

export seed=3
export dataset='cifar10'
export nlabels=500
export CIFAR10ContaminationLevel=25

## NOTE all env vars that have been 'export'-ed will be passed along to the .slurm file

if [[ $ACTION_NAME == 'submit' ]]; then
    ## Use this line to submit the experiment to the batch scheduler
    sbatch < $ROOT_PATH/src_CIFAR10/build_datasets/CIFAR10Contamination_ssl/$normalization/do_BuildContaminationSSL.slurm

elif [[ $ACTION_NAME == 'run_here' ]]; then
    ## Use this line to just run interactively
    bash $ROOT_PATH/src_CIFAR10/build_datasets/CIFAR10Contamination_ssl/$normalization/do_BuildContaminationSSL.slurm
fi

