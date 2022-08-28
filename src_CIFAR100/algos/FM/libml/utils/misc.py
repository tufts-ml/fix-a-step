import time
from tqdm import tqdm
import torch.nn.functional as F

import logging
import numpy as np
import os
import pickle

import torch

logger = logging.getLogger(__name__)

__all__ = ['get_mean_and_std', 'AverageMeter', 'train_one_epoch', 'eval_model', 'save_pickle', 'calculate_plain_accuracy', 'get_current_global_iteration']

def get_current_global_iteration(args, current_epoch, current_batch_idx):
    
    current_global_iteration = current_epoch * (args.nimg_per_epoch//args.labeledtrain_batchsize) + (current_batch_idx + 1)
    
    return current_global_iteration
    

def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


    
def train_one_epoch(args, labeledtrain_loader, unlabeledtrain_loader, model, ema_model, optimizer, scheduler, epoch, weights=None):
    
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
    mask_probs = AverageMeter() #how frequently the unlabeled samples' confidence score greater than pre-defined threshold
    
    n_steps_per_epoch = args.nimg_per_epoch//args.labeledtrain_batchsize
    
    p_bar = tqdm(range(n_steps_per_epoch), disable=False)
    
    for batch_idx in range(n_steps_per_epoch):
 
        try:
            l_input, l_labels = labeledtrain_iter.next()
        except:
            labeledtrain_iter = iter(labeledtrain_loader)
            l_input, l_labels = labeledtrain_iter.next()
        
        try:
            (u_input_weak, u_input_strong), u_labels = unlabeledtrain_iter.next()
        except:
            unlabeledtrain_iter = iter(unlabeledtrain_loader)
            (u_input_weak, u_input_strong), u_labels = unlabeledtrain_iter.next()
        
        
        data_time.update(time.time() - end_time)
        
        ##############################################################################################################
        #For FM:
        #reference: https://github.com/kekmodel/FixMatch-pytorch/blob/master/train.py
        inputs = interleave(torch.cat((l_input, u_input_weak, u_input_strong)), 2*args.mu+1).to(args.device)
        l_labels = l_labels.to(args.device).long()
        
        logits = model(inputs)
        logits = de_interleave(logits, 2*args.mu+1)
        logits_x = logits[:args.labeledtrain_batchsize]
        logits_u_w, logits_u_s = logits[args.labeledtrain_batchsize:].chunk(2)
        
        del logits
        
        labeledtrain_loss = F.cross_entropy(logits_x, l_labels, reduction='mean')
        
        #label guessing
        pseudo_label = torch.softmax(logits_u_w.detach()/args.temperature, dim=-1)
        max_probs, targets_u = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(args.threshold).float()
        
        unlabeledtrain_loss = (F.cross_entropy(logits_u_s, targets_u, reduction='none') * mask).mean()
        
        
        #from FixMatch and MixMatch: warmup = tf.clip_by_value(tf.to_float(self.step) / (warmup_pos * (FLAGS.train_kimg << 10)), 0, 1)
        current_global_iteration = get_current_global_iteration(args, epoch, batch_idx)
#         print('current_global_iteration is {}'.format(current_global_iteration))

        args.writer.add_scalar('train/lr', scheduler.get_last_lr()[0], current_global_iteration)

        
#         current_lambda_u = args.lambda_u_max * np.clip(current_global_iteration/(args.unlabeledloss_warmup_pos * args.train_iterations), 0, 1)
        current_lambda_u = args.lambda_u_max * 1 #FixMatch algo did not use unlabeled loss rampup schedule
        
        args.writer.add_scalar('train/lambda_u', current_lambda_u, current_global_iteration)

        loss = labeledtrain_loss + current_lambda_u * unlabeledtrain_loss
        
#         print('mask is {}'.format(mask))
        args.writer.add_scalar('train/lambda_u', current_lambda_u, current_global_iteration)
        args.writer.add_scalar('train/gt_mask', mask.mean(), current_global_iteration)
        
        ###############################################################################################################
        
        loss.backward()
        
        total_loss.update(loss.item())
        labeled_loss.update(labeledtrain_loss.item())
        unlabeled_loss_unscaled.update(unlabeledtrain_loss.item())
        unlabeled_loss_scaled.update(unlabeledtrain_loss.item() * current_lambda_u)
        mask_probs.update(mask.mean().item())

        TotalLoss_this_epoch.append(loss.item())
        LabeledLoss_this_epoch.append(labeledtrain_loss.item())
        UnlabeledLossUnscaled_this_epoch.append(unlabeledtrain_loss.item())
        UnlabeledLossScaled_this_epoch.append(unlabeledtrain_loss.item() * current_lambda_u)
        
        optimizer.step()
        scheduler.step()
        
        #update ema model
        ema_model.update(model)
        
        model.zero_grad()
        
        
        batch_time.update(time.time() - end_time)
        
        #update end time
        end_time = time.time()


        #tqdm display for each minibatch update
        p_bar.set_description("Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.4f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {total_loss:.4f}. Loss_x: {labeled_loss:.4f}. Loss_u: {unlabeled_loss_unscaled:.4f}. Mask: {mask:.2f}. ".format(
                epoch=epoch + 1,
                epochs=args.train_epoch,
                batch=batch_idx + 1,
                iter=n_steps_per_epoch,
                lr=scheduler.get_last_lr()[0],
                data=data_time.avg,
                bt=batch_time.avg,
                total_loss=total_loss.avg,
                labeled_loss=labeled_loss.avg,
                unlabeled_loss_unscaled=unlabeled_loss_unscaled.avg,
                mask=mask_probs.avg))
        p_bar.update()
        
        
        
##for debugging
#     print('fc.weight: {}'.format(model.fc.weight.cpu().detach().numpy()))
#     print('output.bias: {}'.format(model.output.bias.cpu().detach().numpy()))
        
#     print('ema fc.weight: {}'.format(ema_model.ema.fc.weight.cpu().detach().numpy()))
#     print('ema output.bias: {}'.format(ema_model.ema.output.bias.cpu().detach().numpy()))
        
    p_bar.close()
        
    
    return TotalLoss_this_epoch, LabeledLoss_this_epoch, UnlabeledLossUnscaled_this_epoch, UnlabeledLossScaled_this_epoch
        
   
    
    
    


#shared helper fct across different algos
def eval_model(args, data_loader, raw_model, ema_model, epoch, evaluation_criterion, weights=None):
    
    if evaluation_criterion == 'plain_accuracy':
        evaluation_method = calculate_plain_accuracy
    else:
        raise NameError('not supported yet')
    
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
                print('calculating weighted loss inside eval')
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

        print('raw {} this evaluation step: {}'.format(evaluation_criterion, raw_performance), flush=True)
        print('ema {} this evaluation step: {}'.format(evaluation_criterion, ema_performance), flush=True)
        
        data_loader.close()
        
        
    return losses.avg, raw_performance, ema_performance, total_targets, total_raw_outputs, total_ema_outputs
    

#shared helper fct across different algos
def calculate_plain_accuracy(output, target):
    
    accuracy = (output.argmax(1) == target).mean()*100
    
    return accuracy


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
