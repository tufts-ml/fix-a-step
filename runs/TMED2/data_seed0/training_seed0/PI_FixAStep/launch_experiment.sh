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
export experiment='TMED2' 
export normalization='unnormalized_HWC'

#hyperparameters inherit from Echo_ClinicalManualScript_torch style
export resume='last_checkpoint.pth.tar'

#experiment setting
export use_class_weights='True'
export dataset_name='echo' 
export data_seed=0
export training_seed=0
export development_size='DEV56'


export train_dir="$ROOT_PATH/experiments/$experiment/$normalization/data_seed$data_seed/training_seed$training_seed/$development_size/$implementation"
mkdir -p $train_dir

export script="$ROOT_PATH/src_TMED2/algos/$implementation/$implementation.py"


export arch='wideresnet_scale4'
export train_epoch=1000 
export start_epoch=0
export eval_every_Xepoch=8

#data paths
export l_train_dataset_path="$ROOT_PATH/ML_DATA/$experiment/$normalization/$dataset_name/data_seed$data_seed/$development_size/train.npy"

export u_train_dataset_path="$ROOT_PATH/ML_DATA/$experiment/$normalization/$dataset_name/all_shared_unlabeledset/u_train.npy"

export val_dataset_path="$ROOT_PATH/ML_DATA/$experiment/$normalization/$dataset_name/data_seed$data_seed/$development_size/val.npy"

export test_dataset_path="$ROOT_PATH/ML_DATA/$experiment/$normalization/$dataset_name/data_seed$data_seed/this_seed_shared_testset/test.npy"




#shared config
export flip='True' #default
export r_crop='True' #default
export g_noise='True' #default
export labeledtrain_batchsize=64 #default
export unlabeledtrain_batchsize=64 #default
export em=0 #default


#PL config, candidate hypers to search
export temperature=0.5
export alpha=0.75

export lr=0.003
export wd=0.0
export lambda_u_max=1.0
export lr_warmup_img=0

export optimizer_type='Adam'
export lr_schedule_type='CosineLR'
export lr_cycle_length=1048576
export unlabeledloss_warmup_schedule_type='GoogleFixmatchMixmatchRepo_Like'
export unlabeledloss_warmup_iterations='0' 
export gradient_align_start_iterations='0'


if [[ $ACTION_NAME == 'submit' ]]; then
    ## Use this line to submit the experiment to the batch scheduler
    sbatch <./do_experiment.slurm

elif [[ $ACTION_NAME == 'run_here' ]]; then
    ## Use this line to just run interactively
    bash ./do_experiment.slurm
fi


