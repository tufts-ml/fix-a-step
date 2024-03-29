# Fix-A-Step

Code for AISTATS 2023 paper: Fix-A-Step: Semi-supervised Learning From Uncurated Unlabeled Data 

Paper PDF: https://arxiv.org/abs/2208.11870

Video: https://www.youtube.com/watch?v=JUnfeisSi9M

# Repo structure
1. src_CIFAR10/: CIFAR-10 experiments source code
2. src_CIFAR100/: CIFAR-100 experiments source code 
3. src_TMED2/:  TMED-2 experiments source code: 
4. Heart2Heart_Transfering/:  Transfering to Unity and CAMUS source code: 
5. runs/: commands and hyper-parameters to run the experiments

# Setup
### Prepare datasets
- CIFAR-10: use code provided in src_CIFAR10/build_datasets to prepare dataset needed for CIFAR-10 experiments
- CIFAR-100: use code provided in src_CIFAR100/build_datasets to prepare dataset needed for CIFAR-100 experiments
- TMED2: please visit https://TMED.cs.tufts.edu and follow the instruction to apply for access, any researcher can apply
- Unity: please visit https://data.unityimaging.net and download the data
- CAMUS: please visit http://camus.creatis.insa-lyon.fr/challenge/#challenges and follow the instructions to register 

### Install Anaconda
Follow the instructions here: https://conda.io/projects/conda/en/latest/user-guide/install/index.html

### Environment
packages needed are specified in environment.yml

# Running experiments
### Define the environment variable
```export ROOT_PATH="paths to this repo" ```
(e.g., '/ab/cd/fix-a-step', then do export ROOT_PATH = '/ab/cd/fix-a-step')

### Example
For example if you want to run Mean Teacher with Fix-A-Step for CIFAR-10 400labels/class , go to [runs/CIFAR10/data_seed0/training_seed0/nlabels4000/MT_FixAStep](runs/CIFAR10/data_seed0/training_seed0/nlabels4000/MT_FixAStep/)

``` bash CIFAR10ContaminationLevelX.sh run_here ```

X is the corresponding experiment you want to run

### A note on reproducibility
While the focus of our paper is reproducibility, ultimately exact comparison to the results in our paper will be conflated by subtle differences such as the version of Pytorch etc (see https://pytorch.org/docs/stable/notes/randomness.html for more detail). We found in our experiment even with same random seed, result can vary sligtly between different runs (but usually less than 1%).


## Citing this work
@article{huang2022fix,
  title={Fix-A-Step: Effective Semi-supervised Learning from Uncurated Unlabeled Sets},
  author={Huang, Zhe and Sidhom, Mary-Joy and Wessler, Benjamin S and Hughes, Michael C},
  journal={arXiv preprint arXiv:2208.11870},
  year={2022}
}
