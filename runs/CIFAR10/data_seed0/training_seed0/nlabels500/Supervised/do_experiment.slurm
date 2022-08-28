#!/usr/bin/env bash

# Pass along all ENV variables as arguments to my Python script
python $script \
    --dataset_name $dataset_name\
    --data_seed $data_seed\
    --training_seed $training_seed\
    --nlabels $nlabels\
    --arch $arch\
    --train_epoch $train_epoch\
    --nimg_per_epoch $nimg_per_epoch\
    --start_epoch $start_epoch\
    --eval_every_Xepoch $eval_every_Xepoch\
    --resume $resume\
    --train_dir $train_dir\
    --l_train_dataset_path $l_train_dataset_path\
    --val_dataset_path $val_dataset_path\
    --test_dataset_path $test_dataset_path\
    --flip $flip\
    --r_crop $r_crop\
    --g_noise $g_noise\
    --labeledtrain_batchsize $labeledtrain_batchsize\
    --lr $lr\
    --wd $wd\
    --lr_warmup_img $lr_warmup_img\
    --optimizer_type $optimizer_type\
    --lr_schedule_type $lr_schedule_type\
    --lr_cycle_length $lr_cycle_length\
    
    