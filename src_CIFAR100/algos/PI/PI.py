#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
naming convention:

[labeledtrain, labeledtrain_batchsize, labeledtrain_iter, labeledtrain_loader, labeledtrain_dataset]

[unlabeledtrain, unlabeledtrain_batchsize, unlabeledtrain_iter, unlabeledtrain_loader, unlabeledtrain_dataset]

'''
import argparse
import logging
import math
import os
import random
import shutil
import time
import json


import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchvision import transforms

from torch.utils.tensorboard import SummaryWriter

from libml.utils import save_pickle
from libml.utils import train_one_epoch, eval_model

import sys
ROOT_PATH = os.environ.get('ROOT_PATH')
sys.path.insert(0, os.path.join(ROOT_PATH, 'src_CIFAR100'))
from models.ema import ModelEMA
from load_data import CIFAR100 as dataset

from libml.utils.pimodel import PiModel


logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser()
#experiment setting
parser.add_argument('--dataset_name', default='cifar100', type=str, help='name of dataset')
parser.add_argument('--data_seed', default=0, type=int, help='random seed data partitioning procedure')
parser.add_argument('--training_seed', default=0, type=int, help='random seed for training procedure')
parser.add_argument("--nlabels", "-n", default=10000, type=int, help="the number of labeled data")
parser.add_argument("--CIFAR100ContaminationLevel", "-lvl", default=0, type=int, help="which level of UASD like Contamination setting: [0, 50, 100]")

parser.add_argument('--arch', default='wideresnet', type=str, help='backbone to use')
parser.add_argument('--train_epoch', default=300, type=int, help='total epochs to run')
parser.add_argument('--nimg_per_epoch', default=5000, type=int, help='how many images in the labeled train set')
parser.add_argument('--start_epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--eval_every_Xepoch', default=1, type=int, help='manual epoch number (useful on restarts)')


parser.add_argument('--resume', default='', type=str,
                    help='name of the checkpoint (default: none)')

parser.add_argument('--resume_checkpoint_fullpath', default='', type=str,
                    help='fullpath of the checkpoint to resume from(default: none)')

parser.add_argument('--train_dir', default='/$ROOT_PATH',
                    help='directory to output the result')


#data paths
parser.add_argument('--l_train_dataset_path', default='', type=str)
parser.add_argument('--u_train_dataset_path', default='', type=str)
parser.add_argument('--val_dataset_path', default='', type=str)
parser.add_argument('--test_dataset_path', default='', type=str)

#shared config
parser.add_argument('--flip', default='True', type=str)
parser.add_argument('--r_crop', default='True', type=str)
parser.add_argument('--g_noise', default='True', type=str)
parser.add_argument('--labeledtrain_batchsize', default=64, type=int)
parser.add_argument('--unlabeledtrain_batchsize', default=64, type=int)
parser.add_argument("--em", default=0, type=float, help="coefficient of entropy minimization. If you try VAT + EM, set 0.06")

#VAT config
parser.add_argument('--dropout_rate', default=0.0, type=float, help='dropout_rate')

parser.add_argument('--lr', default=3e-4, type=float, help='learning rate')

parser.add_argument('--wd', default=5e-4, type=float, help='weight decay')

parser.add_argument('--lambda_u_max', default=1, type=float, help='coefficient of unlabeled loss')

parser.add_argument('--lr_warmup_img', default=0, type=float,
                    help='warmup images for linear rate schedule') #following MixMatch and FixMatch repo

parser.add_argument('--optimizer_type', default='SGD', choices=['SGD', 'Adam'], type=str) 
parser.add_argument('--lr_schedule_type', default='CosineLR', choices=['CosineLR', 'FixedLR'], type=str) 
parser.add_argument('--lr_cycle_length', default='1048576', type=str) #following MixMatch and FixMatch repo


parser.add_argument('--unlabeledloss_warmup_iterations', default='16000', type=str, help='position at which unlabeled loss warmup ends') #following MixMatch and FixMatch repo
parser.add_argument('--unlabeledloss_warmup_schedule_type', default='GoogleFixmatchMixmatchRepo_Exact', choices=['GoogleFixmatchMixmatchRepo_Exact', 'GoogleFixmatchMixmatchRepo_Like', 'YU1utRepo_Exact', 'YU1utRepo_Like', 'PerryingRepo_Exact', 'PerryingRepo_Like'], type=str) 



#default hypers not to search for now
parser.add_argument('--nesterov', action='store_true', default=True,
                    help='use nesterov momentum')

parser.add_argument('--use_ema', action='store_true', default=True,
                    help='use EMA model')

parser.add_argument('--ema_decay', default=0.999, type=float,
                    help='EMA decay rate')

parser.add_argument('--num_classes', default=50, type=int)

def str2bool(s):
    if s == 'True':
        return True
    elif s == 'False':
        return False
    else:
        raise NameError('Bad string')
    

#checked
def save_checkpoint(state, is_best, checkpoint, filename='last_checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint,
                                               'model_best.pth.tar'))
        
#checked
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    
    
def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    lr_cycle_length, #total train iterations
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, float(lr_cycle_length) - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)    


def get_fixed_lr(optimizer,
                num_warmup_steps,
                lr_cycle_length, #total train iterations
                num_cycles=7./16.,
                last_epoch=-1):
    def _lr_lambda(current_step):
        
        return 1.0

    return LambdaLR(optimizer, _lr_lambda, last_epoch) 



def create_model(args):
    if args.arch == 'wideresnet':
        import models.wideresnet as models
        model_depth = 28
        model_width = 2
        model = models.build_wideresnet(depth=model_depth,
                                        widen_factor=model_width,
                                        dropout=args.dropout_rate,
                                        num_classes=args.num_classes)
    else:
        raise NameError('Note implemented yet')
    
    
#     trainable_paramters = sum([p.data.nelement() for p in model.parameters()])
# print("trainable parameters : {}".format(trainable_paramters))

    
    logger.info("Total params: {:.2f}M".format(
        sum(p.numel() for p in model.parameters())/1e6))
    return model




def main(args, brief_summary):
    ssl_obj = PiModel()
    
    #define transform for each part of the dataset
#     cifar10_mean = (0.4914, 0.4822, 0.4465)
#     cifar10_std = (0.2471, 0.2435, 0.2616)
    cifar100_mean = (0.5071, 0.4867, 0.4408)
    cifar100_std = (0.2675, 0.2565, 0.2761)
    
    #confirmed: first to tensor, then normalize
    transform_labeledtrain = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                             padding=int(32*0.125),
                             padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)
    ])
    
    transform_eval = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)
    ])
    
    
    class TransformTwice:
        def __init__(self, transform_fn):
            self.transform_fn = transform_fn
        
        def __call__(self, x):
            out1 = self.transform_fn(x)
            out2 = self.transform_fn(x)
        
            return out1, out2
    
    l_train_dataset = dataset(args.l_train_dataset_path, transform_fn=transform_labeledtrain)
    u_train_dataset = dataset(args.u_train_dataset_path, transform_fn=TransformTwice(transform_labeledtrain))
    val_dataset = dataset(args.val_dataset_path, transform_fn=transform_eval)
    test_dataset = dataset(args.test_dataset_path, transform_fn=transform_eval)
    

    print('Created dataset')
    print("labeled data : {}, unlabeled data : {}, training data : {}".format(
    len(l_train_dataset), len(u_train_dataset), len(l_train_dataset)+len(u_train_dataset)))
    print("validation data : {}, test data : {}".format(len(val_dataset), len(test_dataset)))
#     brief_summary["number_of_data"] = {
#     "labeled":len(l_train_dataset), "unlabeled":len(u_train_dataset),
#     "validation":len(val_dataset), "test":len(test_dataset)
# }

    
    l_loader = DataLoader(l_train_dataset, args.labeledtrain_batchsize, shuffle=True, drop_last=True)
    u_loader = DataLoader(u_train_dataset, args.unlabeledtrain_batchsize, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, 128, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, 128, shuffle=False, drop_last=False)

    
    #create model
    model = create_model(args)
    model.to(args.device)
    
    #optimizer_type choice
    if args.optimizer_type == 'SGD':
        no_decay = ['bias', 'bn']
        grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': args.wd},
            {'params': [p for n, p in model.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        
        optimizer = optim.SGD(grouped_parameters, lr=args.lr,
                              momentum=0.9, nesterov=args.nesterov)
    
    elif args.optimizer_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    else:
        raise NameError('Not supported optimizer setting')
        
    
    
    #lr_schedule_type choice
    if args.lr_schedule_type == 'CosineLR':
        scheduler = get_cosine_schedule_with_warmup(optimizer, args.lr_warmup_img//args.labeledtrain_batchsize, args.lr_cycle_length)
    
    elif args.lr_schedule_type == 'FixedLR':
        scheduler = get_fixed_lr(optimizer, args.lr_warmup_img//args.labeledtrain_batchsize, args.lr_cycle_length)
    
    else:
        raise NameError('Not supported lr scheduler setting')
        
    
    #instantiate the ema model object
    ema_model = ModelEMA(args, model, args.ema_decay)
    
    args.start_epoch = 0
    
    best_val_ema_acc = 0
    best_test_ema_acc_at_val = 0
    
    best_val_raw_acc = 0
    best_test_raw_acc_at_val = 0
    
    #if continued from a checkpoint, overwrite the best_val_ema_acc, best_test_ema_acc_at_val, 
    #                                              best_val_raw_acc, best_test_raw_acc_at_val,
    #                                              start_epoch,
    #                                              model weights, ema model weights
    #                                              optimizer state dict
    #                                              scheduler state dict
    if args.resume_checkpoint_fullpath is not None:
        try:
            os.path.isfile(args.resume_checkpoint_fullpath)
            logger.info("==> Resuming from checkpoint..")
            checkpoint = torch.load(args.resume_checkpoint_fullpath)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            ema_model.ema.load_state_dict(checkpoint['ema_state_dict'])

            best_val_ema_acc = checkpoint['best_val_ema_acc']
            best_test_ema_acc_at_val = checkpoint['best_test_ema_acc_at_val']
            
            best_val_raw_acc = checkpoint['best_val_raw_acc']
            best_test_raw_acc_at_val = checkpoint['best_test_raw_acc_at_val']
            
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
        except:
            print('!!!!Does not have checkpoint yet!!!!')
            
            
    logger.info("***** Running training *****")
    logger.info(f"  Task = {args.dataset_name}")
    logger.info(f"  Num Epochs = {args.train_epoch}")
    logger.info(f"  Batch size per GPU (labeled+unlabeled) = {args.labeledtrain_batchsize + args.unlabeledtrain_batchsize}")
    logger.info(f"  Total optimization steps = {args.train_iterations}")
            
    
    train_loss_dict = dict()
    train_loss_dict['train_total_loss'] = []
    train_loss_dict['labeled_loss'] = []
    train_loss_dict['unlabeled_loss_unscaled'] = []
    train_loss_dict['unlabeled_loss_scaled'] = []
    
    is_best = False
    for epoch in range(args.start_epoch, args.train_epoch):
        val_predictions_save_dict = dict()
        test_predictions_save_dict = dict()
        
        #train
        train_total_loss_list, train_labeled_loss_list, train_unlabeled_loss_unscaled_list, train_unlabeled_loss_scaled_list = train_one_epoch(args, ssl_obj, l_loader, u_loader, model, ema_model, optimizer, scheduler, epoch)
        
        train_loss_dict['train_total_loss'].extend(train_total_loss_list)
        train_loss_dict['labeled_loss'].extend(train_labeled_loss_list)
        train_loss_dict['unlabeled_loss_unscaled'].extend(train_unlabeled_loss_unscaled_list)
        train_loss_dict['unlabeled_loss_scaled'].extend(train_unlabeled_loss_scaled_list)
        
        save_pickle(os.path.join(args.experiment_dir, 'losses'), 'losses_dict.pkl', train_loss_dict)

        if epoch % args.eval_every_Xepoch == 0:
            #val
            val_loss, val_raw_acc, val_ema_acc, val_true_labels, val_raw_predictions, val_ema_predictions = eval_model(args, val_loader, model, ema_model.ema, epoch, evaluation_criterion='plain_accuracy')
            val_predictions_save_dict['raw_acc'] = val_raw_acc
            val_predictions_save_dict['ema_acc'] = val_ema_acc
            val_predictions_save_dict['true_labels'] = val_true_labels
            val_predictions_save_dict['raw_predictions'] = val_raw_predictions
            val_predictions_save_dict['ema_predictions'] = val_ema_predictions

            save_pickle(os.path.join(args.experiment_dir, 'predictions'), 'val_epoch_{}_predictions.pkl'.format(str(epoch)), val_predictions_save_dict)

            #test
            test_loss, test_raw_acc, test_ema_acc, test_true_labels, test_raw_predictions, test_ema_predictions = eval_model(args, test_loader, model, ema_model.ema, epoch, evaluation_criterion='plain_accuracy')

            test_predictions_save_dict['raw_acc'] = test_raw_acc
            test_predictions_save_dict['ema_acc'] = test_ema_acc
            test_predictions_save_dict['true_labels'] = test_true_labels
            test_predictions_save_dict['raw_predictions'] = test_raw_predictions
            test_predictions_save_dict['ema_predictions'] = test_ema_predictions

            save_pickle(os.path.join(args.experiment_dir, 'predictions'), 'test_epoch_{}_predictions.pkl'.format(str(epoch)), test_predictions_save_dict)
        
            if val_raw_acc > best_val_raw_acc:

                best_val_raw_acc = val_raw_acc
                best_test_raw_acc_at_val = test_raw_acc

                save_pickle(os.path.join(args.experiment_dir, 'best_predictions_at_raw_val'), 'val_predictions.pkl', val_predictions_save_dict)

                save_pickle(os.path.join(args.experiment_dir, 'best_predictions_at_raw_val'), 'test_predictions.pkl', test_predictions_save_dict)
            
        
            if val_ema_acc > best_val_ema_acc:
                is_best=True

                best_val_ema_acc = val_ema_acc
                best_test_ema_acc_at_val = test_ema_acc

                save_pickle(os.path.join(args.experiment_dir, 'best_predictions_at_ema_val'), 'val_predictions.pkl', val_predictions_save_dict)

                save_pickle(os.path.join(args.experiment_dir, 'best_predictions_at_ema_val'), 'test_predictions.pkl', test_predictions_save_dict)

            save_checkpoint(
                {
                'epoch': epoch+1,
                'state_dict': model.state_dict(),
                'ema_state_dict': ema_model.ema.state_dict(),
                'best_val_ema_acc': best_val_ema_acc,
                'best_val_raw_acc': best_val_raw_acc,
                'best_test_ema_acc_at_val': best_test_ema_acc_at_val,
                'best_test_raw_acc_at_val': best_test_raw_acc_at_val,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                }, is_best, args.experiment_dir)
        
            #return is_best to False
            is_best = False
        
            logger.info('RAW Best , validation/test %.2f %.2f ' % (best_val_raw_acc, best_test_raw_acc_at_val))

            logger.info('EMA Best, validation/test %.2f %.2f ' % (best_val_ema_acc, best_test_ema_acc_at_val))


            args.writer.add_scalar('train/1.total_loss', np.mean(train_total_loss_list), epoch)
            args.writer.add_scalar('train/2.labeled_loss', np.mean(train_labeled_loss_list), epoch)
            args.writer.add_scalar('train/3.unlabeled_loss_unscaled', np.mean(train_unlabeled_loss_unscaled_list), epoch)
            args.writer.add_scalar('train/4.unlabele_loss_scaled', np.mean(train_unlabeled_loss_scaled_list), epoch)


            args.writer.add_scalar('val/1.val_raw_acc', val_raw_acc, epoch)
            args.writer.add_scalar('val/2.val_ema_acc', val_ema_acc, epoch)
            args.writer.add_scalar('val/3.val_loss', val_loss, epoch)
            args.writer.add_scalar('test/1.test_raw_acc', test_raw_acc, epoch)
            args.writer.add_scalar('test/2.test_ema_acc', test_ema_acc, epoch)
            args.writer.add_scalar('test/3.test_loss', test_loss, epoch)

            brief_summary["number_of_data"] = {
        "labeled":len(l_train_dataset), "unlabeled":len(u_train_dataset),
        "validation":len(val_dataset), "test":len(test_dataset)
    }
            brief_summary['best_test_ema_acc_at_val'] = best_test_ema_acc_at_val 
            brief_summary['best_test_raw_acc_at_val'] = best_test_raw_acc_at_val
            brief_summary['best_val_ema_acc'] = best_val_ema_acc
            brief_summary['best_val_raw_acc'] = best_val_raw_acc
            with open(os.path.join(args.experiment_dir + "brief_summary.json"), "w") as f:
                json.dump(brief_summary, f)
            

    brief_summary["number_of_data"] = {
    "labeled":len(l_train_dataset), "unlabeled":len(u_train_dataset),
    "validation":len(val_dataset), "test":len(test_dataset)
}
    brief_summary['best_test_ema_acc_at_val'] = best_test_ema_acc_at_val 
    brief_summary['best_test_raw_acc_at_val'] = best_test_raw_acc_at_val
    brief_summary['best_val_ema_acc'] = best_val_ema_acc
    brief_summary['best_val_raw_acc'] = best_val_raw_acc

        
    args.writer.close()

    with open(os.path.join(args.experiment_dir + "brief_summary.json"), "w") as f:
        json.dump(brief_summary, f)

if __name__ == '__main__':
    args = parser.parse_args()
    
    cuda = torch.cuda.is_available()
    
    if cuda:
        print('cuda available')
        device = torch.device('cuda')
        args.device = device
        torch.backends.cudnn.benchmark = True
    else:
        raise ValueError('Not Using GPU')
    #     device = "cpu"
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO)

    logger.info(dict(args._get_kwargs()))

    
    if args.training_seed is not None:
        print('setting training seed{}'.format(args.training_seed), flush=True)
        set_seed(args.training_seed)
    
    args.train_iterations = args.train_epoch*args.nimg_per_epoch//args.labeledtrain_batchsize
    print('designated train iterations: {}'.format(args.train_iterations))
    
#     experiment_name = "dropout{}_lr{}_wd{}_lambda_u_max{}_unlabeledloss_warmup_iterations{}_lr_warmup_img{}_em{}".format(args.dropout_rate, args.lr, args.wd, args.lambda_u_max, args.unlabeledloss_warmup_iterations, args.lr_warmup_img, args.em)
    
    experiment_name = "Optimizer-{}_LrSchedule-{}_LrCycleLength-{}_UnlabeledlossWarmupSchedule-{}_UnlabeledlossWarmupIteations-{}_LambdaUMax-{}_lr-{}_wd-{}_em-{}".format(args.optimizer_type, args.lr_schedule_type, args.lr_cycle_length, args.unlabeledloss_warmup_schedule_type, args.unlabeledloss_warmup_iterations, args.lambda_u_max, args.lr, args.wd, args.em)

    
    args.experiment_dir = os.path.join(args.train_dir, experiment_name)
    
    if args.resume != 'None':
        args.resume_checkpoint_fullpath = os.path.join(args.experiment_dir, args.resume)
        print('args.resume_checkpoint_fullpath: {}'.format(args.resume_checkpoint_fullpath))
    else:
        args.resume_checkpoint_fullpath = None
        
    os.makedirs(args.experiment_dir, exist_ok=True)
    args.writer = SummaryWriter(args.experiment_dir)
    
    #brief summary:
    brief_summary = {}
    brief_summary['dataset_name'] = args.dataset_name
    brief_summary['algorithm'] = 'PI'
    brief_summary['hyperparameters'] = {
        'optimizer': args.optimizer_type,
        'lr_schedule_type': args.lr_schedule_type,
        'lr_cycle_length': args.lr_cycle_length,
        'unlabeledloss_warmup_schedule_type':args.unlabeledloss_warmup_schedule_type,
        'unlabeledloss_warmup_iterations': args.unlabeledloss_warmup_iterations,
        'dropout_rate':args.dropout_rate,
        'lr': args.lr,
        'wd': args.wd,
        'lambda_u_max': args.lambda_u_max,
        'lr_warmup_img': args.lr_warmup_img,
    }
    

    main(args, brief_summary)
    
    
    
    


    
