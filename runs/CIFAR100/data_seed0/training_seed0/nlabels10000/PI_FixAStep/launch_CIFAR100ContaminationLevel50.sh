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

export implementation='PI_FixAStep'
export experiment='CIFAR100Contamination_ssl'
export normalization='unnormalized_HWC'

#experiment setting
export dataset_name='cifar100'
export data_seed=0
export training_seed=0
export nlabels=10000
export CIFAR100ContaminationLevel=50

#hyperparameters inherit from Echo_ClinicalManualScript_torch style
export resume='last_checkpoint.pth.tar'

export train_dir="$ROOT_PATH/experiments/$experiment/$normalization/data_seed$data_seed/training_seed$training_seed/nlabels$nlabels/$implementation/CIFAR100ContaminationLevel$CIFAR100ContaminationLevel"
mkdir -p $train_dir

export script="$ROOT_PATH/src_CIFAR100/algos/$implementation/$implementation.py"


export arch='wideresnet'
export train_epoch=10416 #27962 equal to train for 1<<16 kimg in MixMatch and FixMatch repo #10416 equal to train for 500,000 iterations in Oliver et al 2018
export nimg_per_epoch=5000 #size of the labeled train set
export start_epoch=0


#data paths
export l_train_dataset_path="$ROOT_PATH/ML_DATA/$experiment/$normalization/$dataset_name/data_seed$data_seed/nlabels$nlabels/CIFAR100ContaminationLevel$CIFAR100ContaminationLevel/l_train.npy"

export u_train_dataset_path="$ROOT_PATH/ML_DATA/$experiment/$normalization/$dataset_name/data_seed$data_seed/nlabels$nlabels/CIFAR100ContaminationLevel$CIFAR100ContaminationLevel/u_train.npy"

export val_dataset_path="$ROOT_PATH/ML_DATA/$experiment/$normalization/$dataset_name/data_seed$data_seed/nlabels$nlabels/CIFAR100ContaminationLevel$CIFAR100ContaminationLevel/val.npy"

export test_dataset_path="$ROOT_PATH/ML_DATA/$experiment/$normalization/$dataset_name/data_seed$data_seed/nlabels$nlabels/CIFAR100ContaminationLevel$CIFAR100ContaminationLevel/test.npy"


#shared config
export flip='True' #default
export r_crop='True' #default
export g_noise='True' #default
export labeledtrain_batchsize=64 #default
export unlabeledtrain_batchsize=64 #default
export em=0 #default


#PL config, candidate hypers to search
export temperature=0.5
export alpha=0.5

export lr=0.03
export wd=0.0005
export lambda_u_max=10.0
export lr_warmup_img=0

export optimizer_type='SGD'
export lr_schedule_type='CosineLR'
export lr_cycle_length=1048576
export unlabeledloss_warmup_schedule_type='GoogleFixmatchMixmatchRepo_Exact'
export unlabeledloss_warmup_iterations='NA' 
export gradient_align_start_iterations='0'



if [[ $ACTION_NAME == 'submit' ]]; then
    ## Use this line to submit the experiment to the batch scheduler
    sbatch <./do_experiment.slurm

elif [[ $ACTION_NAME == 'run_here' ]]; then
    ## Use this line to just run interactively
    bash ./do_experiment.slurm
fi


