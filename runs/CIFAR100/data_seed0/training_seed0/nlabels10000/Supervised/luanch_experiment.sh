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

export implementation='Supervised'
export experiment='CIFAR100Contamination_ssl'
export normalization='unnormalized_HWC'

#experiment setting
export dataset_name='cifar100'
export data_seed=0
export training_seed=0
export nlabels=10000

#hyperparameters inherit from Echo_ClinicalManualScript_torch style
export resume='last_checkpoint.pth.tar'

export train_dir="$ROOT_PATH/experiments/$experiment/$normalization/data_seed$data_seed/training_seed$training_seed/nlabels$nlabels/$implementation"
mkdir -p $train_dir

export script="$ROOT_PATH/src_CIFAR100/algos/$implementation/$implementation.py"


export arch='wideresnet'
export train_epoch=10416 #27962 equal to train for 1<<16 kimg in MixMatch and FixMatch repo #10416 equal to train for 500,000 iterations in Oliver et al 2018
export nimg_per_epoch=5000 #size of the labeled train set, 50 classes, 100 per class
export start_epoch=0


#data paths
export l_train_dataset_path="$ROOT_PATH/ML_DATA/$experiment/$normalization/$dataset_name/data_seed$data_seed/nlabels$nlabels/CIFAR100ContaminationLevel50/l_train.npy"

export val_dataset_path="$ROOT_PATH/ML_DATA/$experiment/$normalization/$dataset_name/data_seed$data_seed/nlabels$nlabels/CIFAR100ContaminationLevel50/val.npy"

export test_dataset_path="$ROOT_PATH/ML_DATA/$experiment/$normalization/$dataset_name/data_seed$data_seed/nlabels$nlabels/CIFAR100ContaminationLevel50/test.npy"


#shared config
export flip='True' #default
export r_crop='True' #default
export g_noise='True' #default
export labeledtrain_batchsize=64 #default


#PL config, candidate hypers to search
export lr=0.003
export wd=0.002
export lr_warmup_img=0
export optimizer_type='SGD'
export lr_schedule_type='CosineLR'
export lr_cycle_length=390600



if [[ $ACTION_NAME == 'submit' ]]; then
    ## Use this line to submit the experiment to the batch scheduler
    sbatch <./do_experiment.slurm

elif [[ $ACTION_NAME == 'run_here' ]]; then
    ## Use this line to just run interactively
    bash ./do_experiment.slurm
fi


