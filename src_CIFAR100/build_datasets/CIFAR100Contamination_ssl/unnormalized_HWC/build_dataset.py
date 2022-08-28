'''
This script is for CIFAR100, uses the setting inspired by UASD, following same coding style as for CIFAR10 experiments

0% contamination: --labeled_classes=0-49 --unlabeled_classes=0-49
50% contamination: --labeled_classes=0-49 --unlabeled_classes_filter=25-74
100% contamination: --labeled_classes_filter=0-49 --unlabeled_classes_filter=50-99 
'''

from torchvision import datasets
import argparse, os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--data_rootdir", default='', type=str, help="folder to store the data")
parser.add_argument("--seed", "-s", default=1, type=int, help="random seed")
parser.add_argument("--dataset", "-d", default="cifar100", type=str, help="dataset name : [cifar10]")
parser.add_argument("--nlabels", "-n", default=10000, type=int, help="the number of total labeled data for 100 classes")
parser.add_argument("--CIFAR100ContaminationLevel", default=0, type=int, help="contamination setting similar to UASD: [50,100]")
parser.add_argument("--normalization", default='unnormalized_HWC', type=str, help="[gcn_zca_normalized_CHW, unnormalized_HWC]")

#confirmed: smilar to cifar10 usage
COUNTS = {
    "cifar100": {"train": 50000, "test": 10000, "valid": 5000, "extra": 0},
}



def split_l_u(train_set, n_labels, CIFAR100ContaminationLevel):
    # NOTE: this function assume that the train_set is shuffled
    images = train_set["images"]
    labels = train_set["labels"]
    classes = np.unique(labels)
    n_labels_per_cls = n_labels // len(classes)
    print('Inside split_l_u, n_labels_per_cls: {}'.format(n_labels_per_cls))
    l_images = []
    l_labels = []
    u_images = []
    u_labels = []

    #labeled classes  #array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
    #17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
    #34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49])
    labeled_classes = np.arange(0, 50) #classes 0-49
#     labeled_classes = [2,3,4,5,6,7]
        
    if CIFAR100ContaminationLevel == 50:
        unlabeled_classes = np.arange(25, 75) #classes 25-74
        
    elif CIFAR100ContaminationLevel == 100:
        unlabeled_classes = np.arange(50, 100) #classes 50-99
    else:
        raise NameError('no such CIFAR100ContaminationLevel')
    
    for c in classes:
        cls_mask = (labels==c)
        c_images = images[cls_mask]
        c_labels = labels[cls_mask]
        l_images += [c_images[:n_labels_per_cls]]
        l_labels += [c_labels[:n_labels_per_cls]]
        u_images += [c_images[n_labels_per_cls:]]
        u_labels += [c_labels[n_labels_per_cls:]]
    
    l_images = np.concatenate(l_images, 0)
    l_labels = np.concatenate(l_labels, 0)
    u_images = np.concatenate(u_images, 0)
    u_labels = np.concatenate(u_labels, 0)
    
    labeled_cls_mask = [True if element in labeled_classes else False for element in l_labels]
    unlabeled_cls_mask = [True if element in unlabeled_classes else False for element in u_labels]
    
    l_images = l_images[labeled_cls_mask]
    l_labels = l_labels[labeled_cls_mask]
    
    u_images = u_images[unlabeled_cls_mask]
    u_labels_true = u_labels[unlabeled_cls_mask] #for debugging
    u_labels = np.zeros_like(u_labels_true)-1

    l_train_set = {"images": l_images, "labels": l_labels}
    u_train_set = {"images": u_images, "labels": u_labels_true}
    
    return l_train_set, u_train_set
    



#confirmed with pytorch.org and several other repos: this is the correct way to download with torchvision dataset
def _load_cifar100(data_rootdir): 
    splits = {}
    for train in [True, False]:
        tv_data = datasets.CIFAR100(data_rootdir, train, download=True)
        data = {}
        data["images"] = tv_data.data
        data["labels"] = np.array(tv_data.targets)        
        splits["train" if train else "test"] = data
    return splits.values()





if __name__=='__main__':
    args = parser.parse_args()
    
    #confirmed: similar as CIFAR10;
    #random seed for separating train and val set
    rng = np.random.RandomState(args.seed)

    #confirmed: similar as CIFAR10;
    validation_count = COUNTS[args.dataset]["valid"] #5000 for 100 classes

    
    train_set, test_set = _load_cifar100(args.data_rootdir)
    
    #Add test set class filtering for CIFAR100ContaminationSetting, so that test set contains only class 0-49
    test_cls_mask = [True if element in np.arange(0, 50) else False for element in test_set["labels"]]
    test_set["images"] = test_set["images"][test_cls_mask]
    test_set["labels"] = test_set["labels"][test_cls_mask]

    
    
    #confirmed: same as CIFAR10;
    # permute index of training set
    indices = rng.permutation(len(train_set["images"]))
    train_set["images"] = train_set["images"][indices]
    train_set["labels"] = train_set["labels"][indices]

    train_total_classes = sorted(list(set(train_set["labels"])))
    n_validation_per_cls = validation_count // len(train_total_classes) #50 validation per class, total 5000 for 100 CIFAR100 classes
    train_images = []
    train_labels = []
    validation_images = []
    validation_labels = []
    
    for c in train_total_classes:
        cls_mask = (train_set["labels"]==c)
        c_images = train_set["images"][cls_mask]
        c_labels = train_set["labels"][cls_mask]
        train_images += [c_images[n_validation_per_cls:]]
        train_labels += [c_labels[n_validation_per_cls:]]
        validation_images += [c_images[:n_validation_per_cls]]
        validation_labels += [c_labels[:n_validation_per_cls]]
    
    train_images = np.concatenate(train_images, 0)
    train_labels = np.concatenate(train_labels, 0)
    validation_images = np.concatenate(validation_images, 0)
    validation_labels = np.concatenate(validation_labels, 0)
    
 
    #Add val set class filtering for CIFAR100ContaminationSetting, so that val set contains only classes 0-49
    validation_cls_mask = [True if element in np.arange(0, 50) else False for element in validation_labels]
    validation_images = validation_images[validation_cls_mask]
    validation_labels = validation_labels[validation_cls_mask]
    
    validation_set = {"images": validation_images, "labels": validation_labels}
    train_set = {"images": train_images, "labels": train_labels}
    
    
    # split training set into labeled data and unlabeled data
    l_train_set, u_train_set = split_l_u(train_set, args.nlabels, args.CIFAR100ContaminationLevel)

    #Design properties:
    #1. under the same seed, across different nlabels, the labeled train set of smaller nlabels is strictly a subset of labeled train set of larger nlabels; unlabeled train set of smaller nlabels is strictly a subset of unlabeled train set of larger nlabels (under the same CIFAR100ContaminationLevel)
    #2. under the same seed, for same nlabels, different CIFAR100ContaminationLevel use exactly the same labeled train set
    if not os.path.exists(os.path.join(args.data_rootdir, 'CIFAR100Contamination_ssl', args.normalization, args.dataset, 'data_seed{}'.format(args.seed), 'nlabels{}'.format(args.nlabels), 'CIFAR100ContaminationLevel{}'.format(str(args.CIFAR100ContaminationLevel)))):
        os.makedirs(os.path.join(args.data_rootdir, 'CIFAR100Contamination_ssl', args.normalization, args.dataset, 'data_seed{}'.format(args.seed), 'nlabels{}'.format(args.nlabels), 'CIFAR100ContaminationLevel{}'.format(str(args.CIFAR100ContaminationLevel))))

    np.save(os.path.join(args.data_rootdir, 'CIFAR100Contamination_ssl', args.normalization, args.dataset, 'data_seed{}'.format(args.seed), 'nlabels{}'.format(args.nlabels), 'CIFAR100ContaminationLevel{}'.format(str(args.CIFAR100ContaminationLevel)), "l_train"), l_train_set)
    np.save(os.path.join(args.data_rootdir, 'CIFAR100Contamination_ssl', args.normalization, args.dataset, 'data_seed{}'.format(args.seed), 'nlabels{}'.format(args.nlabels), 'CIFAR100ContaminationLevel{}'.format(str(args.CIFAR100ContaminationLevel)), "u_train"), u_train_set)
    np.save(os.path.join(args.data_rootdir, 'CIFAR100Contamination_ssl', args.normalization, args.dataset, 'data_seed{}'.format(args.seed), 'nlabels{}'.format(args.nlabels), 'CIFAR100ContaminationLevel{}'.format(str(args.CIFAR100ContaminationLevel)), "val"), validation_set)
    np.save(os.path.join(args.data_rootdir, 'CIFAR100Contamination_ssl', args.normalization, args.dataset, 'data_seed{}'.format(args.seed), 'nlabels{}'.format(args.nlabels), 'CIFAR100ContaminationLevel{}'.format(str(args.CIFAR100ContaminationLevel)), "test"), test_set)
    
    
    
    




