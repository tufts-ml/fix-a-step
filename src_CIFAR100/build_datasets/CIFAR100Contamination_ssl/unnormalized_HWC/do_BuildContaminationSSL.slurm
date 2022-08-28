#!/usr/bin/env bash


# Pass along all ENV variables as arguments to my Python script
python3 $script\
    --data_rootdir $data_rootdir\
    --seed $seed\
    --dataset $dataset\
    --nlabels $nlabels\
    --CIFAR100ContaminationLevel $CIFAR100ContaminationLevel\
    --normalization $normalization\
