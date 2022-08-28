'''
This script is only for CIFAR10
0% contamination: --labeled_classes_filter=2,3,4,5,6,7 --unlabeled_classes_filter=4,5,6,7
25% contamination: --labeled_classes_filter=2,3,4,5,6,7 --unlabeled_classes_filter=0,5,6,7
50% contamination: --labeled_classes_filter=2,3,4,5,6,7 --unlabeled_classes_filter=0,1,6,7
75% contamination: --labeled_classes_filter=2,3,4,5,6,7 --unlabeled_classes_filter=0,1,8,7
100% contamination: --labeled_classes_filter=2,3,4,5,6,7 --unlabeled_classes_filter=0,1,8,9 
'''

from torchvision import datasets
import argparse, os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--data_rootdir", default='', type=str, help="folder to store the data")
parser.add_argument("--seed", "-s", default=1, type=int, help="random seed")
parser.add_argument("--dataset", "-d", default="cifar10", type=str, help="dataset name : [cifar10]")
parser.add_argument("--nlabels", "-n", default=1000, type=int, help="the number of labeled data")
parser.add_argument("--CIFAR10ContaminationLevel", default=0, type=int, help="[0,25,50,75,100]")
parser.add_argument("--normalization", default='unnormalized_HWC', type=str, help="[gcn_zca_normalized_CHW, unnormalized_HWC]")

COUNTS = {
    "cifar10": {"train": 50000, "test": 10000, "valid": 5000, "extra": 0},
}



def split_l_u(train_set, n_labels, CIFAR10ContaminationLevel):
    # NOTE: this function assume that the train_set is shuffled
    images = train_set["images"]
    labels = train_set["labels"]
    classes = np.unique(labels)
    n_labels_per_cls = n_labels // len(classes)
    l_images = []
    l_labels = []
    u_images = []
    u_labels = []

    labeled_classes = [2,3,4,5,6,7]
    
    if CIFAR10ContaminationLevel == 0:
        unlabeled_classes = [4,5,6,7]
    elif CIFAR10ContaminationLevel == 25:
        unlabeled_classes = [0,5,6,7]
    elif CIFAR10ContaminationLevel == 50:
        unlabeled_classes = [0,1,6,7]
    elif CIFAR10ContaminationLevel == 75:
        unlabeled_classes = [0,1,8,7]
    elif CIFAR10ContaminationLevel == 100:
        unlabeled_classes = [0,1,8,9]
    else:
        raise NameError('no such CIFAR10ContaminationLevel')
    
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
    #perform labels mapping for labeled set, validation set and test set, in order to train 6-class classifier
    l_labels = l_labels[labeled_cls_mask]
    l_labels = l_labels - 2
    
    u_images = u_images[unlabeled_cls_mask]
    u_labels_true = u_labels[unlabeled_cls_mask] #for debugging
    u_labels = np.zeros_like(u_labels_true)-1

    l_train_set = {"images": l_images, "labels": l_labels}
    u_train_set = {"images": u_images, "labels": u_labels_true}
    
    return l_train_set, u_train_set
    



#confirmed with pytorch.org and several other repos: this is the correct way to download with torchvision dataset
def _load_cifar10(data_rootdir): 
    splits = {}
    for train in [True, False]:
        tv_data = datasets.CIFAR10(data_rootdir, train, download=True)
        data = {}
        data["images"] = tv_data.data
        data["labels"] = np.array(tv_data.targets)        
        splits["train" if train else "test"] = data
    return splits.values()





if __name__=='__main__':
    args = parser.parse_args()
    
    #random seed for separating train and val set
    rng = np.random.RandomState(args.seed)

    validation_count = COUNTS[args.dataset]["valid"]

    
    train_set, test_set = _load_cifar10(args.data_rootdir)
    
    test_cls_mask = [True if element in [2,3,4,5,6,7] else False for element in test_set["labels"]]
    test_set["images"] = test_set["images"][test_cls_mask]
    test_set["labels"] = test_set["labels"][test_cls_mask]
    #perform labels mapping for labeled set, validation set and test set, in order to train 6-class classifier
    test_set["labels"] = test_set["labels"] - 2
    
    
    # permute index of training set
    indices = rng.permutation(len(train_set["images"]))
    train_set["images"] = train_set["images"][indices]
    train_set["labels"] = train_set["labels"][indices]

    train_total_classes = sorted(list(set(train_set["labels"])))
    n_validation_per_cls = validation_count // len(train_total_classes) #500 validation per class, total 5000 for 10 classes
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
    
 
    validation_cls_mask = [True if element in [2,3,4,5,6,7] else False for element in validation_labels]
    validation_images = validation_images[validation_cls_mask]
    validation_labels = validation_labels[validation_cls_mask]
    #perform labels mapping for labeled set, validation set and test set, in order to train 6-class classifier
    validation_labels = validation_labels - 2
    
    validation_set = {"images": validation_images, "labels": validation_labels}
    train_set = {"images": train_images, "labels": train_labels}
    
    
    # split training set into labeled data and unlabeled data
    l_train_set, u_train_set = split_l_u(train_set, args.nlabels, args.CIFAR10ContaminationLevel)

    
    if not os.path.exists(os.path.join(args.data_rootdir, 'CIFAR10Contamination_ssl', args.normalization, args.dataset, 'data_seed{}'.format(args.seed), 'nlabels{}'.format(args.nlabels), 'CIFAR10ContaminationLevel{}'.format(str(args.CIFAR10ContaminationLevel)))):
        os.makedirs(os.path.join(args.data_rootdir, 'CIFAR10Contamination_ssl', args.normalization, args.dataset, 'data_seed{}'.format(args.seed), 'nlabels{}'.format(args.nlabels), 'CIFAR10ContaminationLevel{}'.format(str(args.CIFAR10ContaminationLevel))))

        
    np.save(os.path.join(args.data_rootdir, 'CIFAR10Contamination_ssl', args.normalization, args.dataset, 'data_seed{}'.format(args.seed), 'nlabels{}'.format(args.nlabels), 'CIFAR10ContaminationLevel{}'.format(str(args.CIFAR10ContaminationLevel)), "l_train"), l_train_set)
    np.save(os.path.join(args.data_rootdir, 'CIFAR10Contamination_ssl', args.normalization, args.dataset, 'data_seed{}'.format(args.seed), 'nlabels{}'.format(args.nlabels), 'CIFAR10ContaminationLevel{}'.format(str(args.CIFAR10ContaminationLevel)), "u_train"), u_train_set)
    np.save(os.path.join(args.data_rootdir, 'CIFAR10Contamination_ssl', args.normalization, args.dataset, 'data_seed{}'.format(args.seed), 'nlabels{}'.format(args.nlabels), 'CIFAR10ContaminationLevel{}'.format(str(args.CIFAR10ContaminationLevel)), "val"), validation_set)
    np.save(os.path.join(args.data_rootdir, 'CIFAR10Contamination_ssl', args.normalization, args.dataset, 'data_seed{}'.format(args.seed), 'nlabels{}'.format(args.nlabels), 'CIFAR10ContaminationLevel{}'.format(str(args.CIFAR10ContaminationLevel)), "test"), test_set)
    
    
    
    




