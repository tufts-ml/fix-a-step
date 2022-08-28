import argparse
import logging
import os
import random


import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from torch.utils.tensorboard import SummaryWriter
# from tqdm import tqdm

# from dataset.cifar import DATASET_GETTERS
from libml.Echo_data import Echo_LabeledDataset, 

from libml.utils import save_pickle
from libml.utils import  eval_model 
from libml.models.ema import ModelEMA

logger = logging.getLogger(__name__)
best_acc = 0

parser = argparse.ArgumentParser(description='PyTorch FixMatch Training')


#paths
parser.add_argument('--dataset', default='echo', type=str,
                    choices=['echo'],
                    help='dataset name')
parser.add_argument('--image_path', default="", type=str)
parser.add_argument('--label_path', default="", type=str)

parser.add_argument('--train_dir', default='',help='directory to output the result')


parser.add_argument('--resume_checkpoint_fullpath', default='', type=str,
                    help='fullpath of the checkpoint to resume from(default: none)')

parser.add_argument('--seed', default=0, type=int,
                    help="random seed")
parser.add_argument('--arch', default='wideresnet_scale4', type=str,
                    help="model")
parser.add_argument('--dropout_rate', default=0, type=int,
                    help="model")
parser.add_argument('--class_weights', default="1,1,1,1", type=str,
                    help="model")
parser.add_argument('--ema_decay', default=0.999, type=float,
                    help='EMA decay rate')

    

#checked
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    




def create_model(args):
    if args.arch == 'wideresnet':
        import libml.models.wideresnet as models
        model = models.build_wideresnet(depth=args.model_depth,
                                        widen_factor=args.model_width,
                                        dropout=args.dropout_rate,
                                        num_classes=args.num_classes)
    elif args.arch=='wideresnet_scale4':
        import libml.models.wideresnet_ModifiedToBeSameAsTF as models
        model = models.build_wideresnet(depth=args.model_depth,
                                        widen_factor=args.model_width,
                                        dropout=args.dropout_rate,
                                        num_classes=args.num_classes)
        
    elif args.arch == 'resnext':
        import libml.models.resnext as models
        model = models.build_resnext(cardinality=args.model_cardinality,
                                     depth=args.model_depth,
                                     width=args.model_width,
                                     num_classes=args.num_classes)
    logger.info("Total params: {:.2f}M".format(
        sum(p.numel() for p in model.parameters())/1e6))
    return model




def main(args):
    
    

    image_path = args.image_path
    label_path = args.label_path
    args.train_epoch = 150
    train_epoch= 150
    transform_eval = transforms.Compose([transforms.ToTensor()]) 

    
    dataset = Echo_LabeledDataset(image_path, label_path, transform_eval)
    loader = DataLoader(dataset, shuffle=False)


    # These weights don't matter, they're just here because the eval function 
    # needs them
    args.PLAX_PSAX_upweight_factor = 3
    weights = args.class_weights
    weights = [float(i) for i in weights.split(',')]
    
    weights = torch.Tensor(weights)
    weights = weights.to(args.device)

    model = create_model(args)
    model.to(args.device)
    
    ema_model = ModelEMA(args, model, args.ema_decay)
    
    if args.resume_checkpoint_fullpath is not None:
        try:
            os.path.isfile(args.resume_checkpoint_fullpath)
            logger.info("==> Resuming from checkpoint..")
            checkpoint = torch.load(args.resume_checkpoint_fullpath)
            model.load_state_dict(checkpoint['state_dict'])
            ema_model.ema.load_state_dict(checkpoint['ema_state_dict'])
            
        except Exception as e:
            print(e)
    
    

    predictions_save_dict = dict()

    _, raw_acc, ema_acc, true_labels, raw_predictions, ema_predictions = eval_model(args, weights, loader, model, ema_model.ema, train_epoch, criterion='balanced_accuracy')
    
    predictions_save_dict['raw_accuracy'] = raw_acc
    predictions_save_dict['ema_accuracy'] = ema_acc
    predictions_save_dict['true_labels'] = true_labels
    predictions_save_dict['raw_predictions'] = raw_predictions
    predictions_save_dict['ema_predictions'] = ema_predictions

    save_pickle(os.path.join(args.experiment_dir, 'predictions'), f'predictions.pkl', predictions_save_dict)




if __name__ == '__main__':
    print('Fixed Train Labeled set DA!!!!!!!!!!!!!!!!!')
    
    args = parser.parse_args()

    cuda = torch.cuda.is_available()
    if cuda:
        print('cuda available')
        device = torch.device('cuda')
        args.device = device
    else:
        raise ValueError('Not Using GPU?')

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO)

  
    logger.info(dict(args._get_kwargs()))

    if args.seed is not None:
        print('setting seed{}'.format(args.seed), flush=True)
        set_seed(args)

    
    args.experiment_dir = args.train_dir
    
        
    os.makedirs(args.experiment_dir, exist_ok=True)
    args.writer = SummaryWriter(args.experiment_dir)

    if args.dataset == 'echo':
        args.num_classes=4
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 2
        elif args.arch == 'wideresnet_scale4':
            args.model_depth = 28
            args.model_width = 2
        elif args.arch == 'resnext':
            args.model_cardinality = 4
            args.model_depth = 28
            args.model_width = 4

    main(args)