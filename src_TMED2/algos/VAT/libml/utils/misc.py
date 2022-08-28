import time
from tqdm import tqdm
import torch.nn.functional as F

import logging
from sklearn.metrics import confusion_matrix as sklearn_cm
import numpy as np
import os
import pickle

import math
import torch


logger = logging.getLogger(__name__)

__all__ = ['get_mean_and_std', 'AverageMeter', 'train_one_epoch', 'eval_model', 'save_pickle', 'calculate_plain_accuracy', 'calculate_balanced_accuracy', 'get_current_global_iteration']

def get_current_global_iteration(args, current_epoch, current_batch_idx):
    
    current_global_iteration = current_epoch * (args.nimg_per_epoch//args.labeledtrain_batchsize) + (current_batch_idx + 1)
    
    return current_global_iteration
    
                                 
    
def train_one_epoch(args, weights, ssl_obj, labeledtrain_loader, unlabeledtrain_loader, model, ema_model, optimizer, scheduler, epoch):
    
    '''
    this implementation follow: https://github.com/perrying/realistic-ssl-evaluation-pytorch/blob/master/lib/algs/pseudo_label.py
    #same as Oliver et al 2018, which use vanilla logits when unlabeled samples has maximum probability below threshold
    '''
    
    TotalLoss_this_epoch, LabeledLoss_this_epoch, UnlabeledLossUnscaled_this_epoch, UnlabeledLossScaled_this_epoch = [], [], [], []
    
    end_time = time.time()
    
    labeledtrain_iter = iter(labeledtrain_loader)
    unlabeledtrain_iter = iter(unlabeledtrain_loader)
    
    model.train()
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    total_loss = AverageMeter()
    labeled_loss = AverageMeter()
    unlabeled_loss_unscaled = AverageMeter()
    unlabeled_loss_scaled = AverageMeter()
    
    n_steps_per_epoch = args.nimg_per_epoch//args.labeledtrain_batchsize
    
    p_bar = tqdm(range(n_steps_per_epoch), disable=False)
    
    for batch_idx in range(n_steps_per_epoch):
 
        try:
            l_input, l_labels = labeledtrain_iter.next()
        except:
            labeledtrain_iter = iter(labeledtrain_loader)
            l_input, l_labels = labeledtrain_iter.next()
        
        try:
            (inputs_u, inputs_u2), u_labels = unlabeledtrain_iter.next()
        except:
            unlabeledtrain_iter = iter(unlabeledtrain_loader)
            (inputs_u, inputs_u2), u_labels = unlabeledtrain_iter.next()
        
        
        data_time.update(time.time() - end_time)
        
        ##############################################################################################################
        #For VAT
        #reference: https://github.com/perrying/realistic-ssl-evaluation-pytorch/blob/master/train.py
        
         
        #put data to device
        l_input, l_labels = l_input.to(args.device).float(), l_labels.to(args.device).long()
        inputs_u = inputs_u.to(args.device).float()
        inputs_u2 = inputs_u2.to(args.device).float()
        
        dummy_u_labels = torch.zeros_like(u_labels) - 1
        dummy_u_labels = dummy_u_labels.to(args.device).long()
        
        combined_labels = torch.cat([l_labels, dummy_u_labels], 0)
        unlabeled_mask = (combined_labels == -1).float()
        
        combined_input = torch.cat([l_input, inputs_u], 0)
        combined_outputs = model(combined_input) #outputs from model is pre-softmax
        
#         print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!Combined input pass through model!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
#         print('bn1.bias requires_grad: {}, is {}'.format(model.bn1.bias.requires_grad, model.bn1.bias))
#         print('bn1.running_mean requires_grad: {}, is {}'.format(model.bn1.running_mean.requires_grad, model.bn1.running_mean))
#         print('bn1.running_var requires_grad: {}, is {}'.format(model.bn1.running_var.requires_grad, model.bn1.running_var))
#         print('bn1.num_batches_tracked requires_grad: {}, is {}'.format(model.bn1.num_batches_tracked.requires_grad, model.bn1.num_batches_tracked))
#         print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')    
        
        labeledtrain_loss = F.cross_entropy(combined_outputs, combined_labels, weights, reduction='mean', ignore_index=-1)
        
#         unlabeledtrain_loss = ssl_obj(combined_input, combined_outputs.detach(), model, unlabeled_mask)
        unlabeledtrain_loss = ssl_obj(combined_input, combined_outputs.detach(), model, unlabeled_mask)
       
        current_global_iteration = get_current_global_iteration(args, epoch, batch_idx)
#         print('current_global_iteration is {}'.format(current_global_iteration))

        args.writer.add_scalar('train/lr', scheduler.get_last_lr()[0], current_global_iteration)
        
#         #https://discuss.pytorch.org/t/get-current-lr-of-optimizer-with-adaptive-lr/24851
#         args.writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], current_global_iteration)
        
        #unlabeledloss warmup schedule choice
        if args.unlabeledloss_warmup_schedule_type == 'GoogleFixmatchMixmatchRepo_Exact':
            current_lambda_u = args.lambda_u_max * np.clip(current_global_iteration/419430, 0, 1)
        
        elif args.unlabeledloss_warmup_schedule_type == 'GoogleFixmatchMixmatchRepo_Like':
            current_lambda_u = args.lambda_u_max * np.clip(current_global_iteration/max(1, float(args.unlabeledloss_warmup_iterations)), 0, 1)
        
        elif args.unlabeledloss_warmup_schedule_type == 'YU1utRepo_Exact':
            current_lambda_u = args.lambda_u_max * np.clip((current_global_iteration//1024)/1024, 0, 1)
        
        elif args.unlabeledloss_warmup_schedule_type == 'YU1utRepo_Like':
            current_lambda_u = args.lambda_u_max * np.clip(epoch/args.train_epoch, 0, 1)
            
        elif args.unlabeledloss_warmup_schedule_type == 'PerryingRepo_Exact':
            current_lambda_u = args.lambda_u_max * math.exp(-5 * (1 - min(current_global_iteration/200000, 1))**2)
        
        elif args.unlabeledloss_warmup_schedule_type == 'PerryingRepo_Like':
            current_lambda_u = args.lambda_u_max * math.exp(-5 * (1 - min(current_global_iteration/float(args.unlabeledloss_warmup_iterations), 1))**2)
        
        else:
            raise NameError('Not supported unlabeledloss warmup schedule')
            
        
        args.writer.add_scalar('train/lambda_u', current_lambda_u, current_global_iteration)

        loss = labeledtrain_loss + current_lambda_u * unlabeledtrain_loss
        
        if args.em > 0:
            loss -= args.em * ((combined_outputs.softmax(1) * F.log_softmax(combined_outputs, 1)).sum(1) * unlabeled_mask).mean()
        
        ###############################################################################################################
        
        loss.backward()
        
        total_loss.update(loss.item())
        labeled_loss.update(labeledtrain_loss.item())
        unlabeled_loss_unscaled.update(unlabeledtrain_loss.item())
        unlabeled_loss_scaled.update(unlabeledtrain_loss.item() * current_lambda_u)

        TotalLoss_this_epoch.append(loss.item())
        LabeledLoss_this_epoch.append(labeledtrain_loss.item())
        UnlabeledLossUnscaled_this_epoch.append(unlabeledtrain_loss.item())
        UnlabeledLossScaled_this_epoch.append(unlabeledtrain_loss.item() * current_lambda_u)
        
        optimizer.step()
        scheduler.step() #FixMatch repo's vat implementation didn't use lr schedule
        
        #update ema model
        ema_model.update(model)
        
        model.zero_grad()
        
        
        batch_time.update(time.time() - end_time)
        
        #update end time
        end_time = time.time()


        #tqdm display for each minibatch update
        p_bar.set_description("Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.4f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {total_loss:.4f}. Loss_x: {labeled_loss:.4f}. Loss_u: {unlabeled_loss_unscaled:.4f}. ".format(
                epoch=epoch + 1,
                epochs=args.train_epoch,
                batch=batch_idx + 1,
                iter=n_steps_per_epoch,
                lr=scheduler.get_last_lr()[0],
                data=data_time.avg,
                bt=batch_time.avg,
                total_loss=total_loss.avg,
                labeled_loss=labeled_loss.avg,
                unlabeled_loss_unscaled=unlabeled_loss_unscaled.avg,))
        p_bar.update()
        
        
        
##for debugging
    print('fc.weight: {}'.format(model.fc.weight.cpu().detach().numpy()))
#     print('output.bias: {}'.format(model.output.bias.cpu().detach().numpy()))
        
    print('ema fc.weight: {}'.format(ema_model.ema.fc.weight.cpu().detach().numpy()))
#     print('ema output.bias: {}'.format(ema_model.ema.output.bias.cpu().detach().numpy()))
        
    p_bar.close()
        
    
    return TotalLoss_this_epoch, LabeledLoss_this_epoch, UnlabeledLossUnscaled_this_epoch, UnlabeledLossScaled_this_epoch
        
   
    
    
    


#shared helper fct across different algos
def eval_model(args, weights, data_loader, raw_model, ema_model, epoch, criterion='balanced_accuracy'):
    
    if criterion == 'plain_accuracy':
        evaluation_method = calculate_plain_accuracy
    elif criterion == 'balanced_accuracy':
        evaluation_method = calculate_balanced_accuracy
    else:
        raise NameError('not supported criterion')
    
    raw_model.eval()
    ema_model.eval()

    end_time = time.time()
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    
    data_loader = tqdm(data_loader, disable=False)
    
    with torch.no_grad():
        total_targets = []
        total_raw_outputs = []
        total_ema_outputs = []
        
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            data_time.update(time.time() - end_time)
            
            inputs = inputs.to(args.device).float()
            targets = targets.to(args.device).long()
            raw_outputs = raw_model(inputs)
            ema_outputs = ema_model(inputs)
            
            total_targets.append(targets.detach().cpu())
            total_raw_outputs.append(raw_outputs.detach().cpu())
            total_ema_outputs.append(ema_outputs.detach().cpu())
            
            if weights is not None:
                loss = F.cross_entropy(raw_outputs, targets, weights)
            else:
                loss = F.cross_entropy(raw_outputs, targets)
            
            losses.update(loss.item(), inputs.shape[0])
            batch_time.update(time.time() - end_time)
            
            #update end time
            end_time = time.time()
            
            
        total_targets = np.concatenate(total_targets, axis=0)
        total_raw_outputs = np.concatenate(total_raw_outputs, axis=0)
        total_ema_outputs = np.concatenate(total_ema_outputs, axis=0)
        
        raw_performance = evaluation_method(total_raw_outputs, total_targets)
        ema_performance = evaluation_method(total_ema_outputs, total_targets)

        print('raw {} this evaluation step: {}'.format(criterion, raw_performance), flush=True)
        print('ema {} this evaluation step: {}'.format(criterion, ema_performance), flush=True)
        
        data_loader.close()
        
        
    return losses.avg, raw_performance, ema_performance, total_targets, total_raw_outputs, total_ema_outputs
    
    

def calculate_plain_accuracy(prediction, true_target):
    
    accuracy = (prediction.argmax(1) == true_target).mean()*100
    
    return accuracy


def calculate_balanced_accuracy(prediction, true_target, return_type = 'only balanced_accuracy'):
    
    confusion_matrix = sklearn_cm(true_target, prediction.argmax(1))
    n_class = confusion_matrix.shape[0]
    print('Inside calculate_balanced_accuracy, {} classes passed in'.format(n_class), flush=True)

    assert n_class==4
    
    recalls = []
    for i in range(n_class): 
        recall = confusion_matrix[i,i]/np.sum(confusion_matrix[i])
        recalls.append(recall)
        print('class{} recall: {}'.format(i, recall), flush=True)
        
    balanced_accuracy = np.mean(np.array(recalls))
    

    if return_type == 'all':
#         return balanced_accuracy * 100, class0_recall * 100, class1_recall * 100, class2_recall * 100
        return balanced_accuracy * 100, recalls

    elif return_type == 'only balanced_accuracy':
        return balanced_accuracy * 100
    else:
        raise NameError('Unsupported return_type in this calculate_balanced_accuracy fn')
        
        
#shared helper fct across different algos
def save_pickle(save_dir, save_file_name, data):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    data_save_fullpath = os.path.join(save_dir, save_file_name)
    with open(data_save_fullpath, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
#shared helper fct across different algos
def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=4)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    logger.info('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


#shared helper fct across different algos
class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
