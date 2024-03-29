import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PL(nn.Module):
    def __init__(self, threshold, num_classes):
        super().__init__()
        self.th = threshold
        self.num_classes = num_classes

    def forward(self, x, y, model, mask):
        y_probs = y.softmax(1)
#         print('y_probs {} requires_grad {} is {}'.format(y_probs.shape, y_probs.requires_grad, y_probs))
#         print('y_probs.max(1)[1] {} requires_grad {} is {}'.format(y_probs.max(1)[1].shape, y_probs.max(1)[1].requires_grad, y_probs.max(1)[1]))
        
        onehot_label = self.__make_one_hot(y_probs.max(1)[1]).float()
#         print('onehot_label {} requires_grad {} is {}'.format(onehot_label.shape, onehot_label.requires_grad, onehot_label))
        
        gt_mask = (y_probs > self.th).float()
#         print('gt_mask {} requires_grad {} is {}'.format(gt_mask.shape, gt_mask.requires_grad, gt_mask))
        
        gt_mask = gt_mask.max(1)[0] # reduce_any
#         print('gt_mask {} requires_grad {} is {}'.format(gt_mask.shape, gt_mask.requires_grad, gt_mask))
        
        lt_mask = 1 - gt_mask # logical not
#         print('lt_mask {} requires_grad {} is {}'.format(lt_mask.shape, lt_mask.requires_grad, lt_mask))
        
#         p_target = gt_mask[:,None] * 10 * onehot_label + lt_mask[:,None] * y_probs
        p_target = gt_mask[:,None] * self.num_classes * onehot_label + lt_mask[:,None] * y_probs
#         print('p_target {} requires_grad {} is {}'.format(p_target.shape, p_target.requires_grad, p_target))

        p_target = p_target.detach()
#         print('p_target {} requires_grad {} is {}'.format(p_target.shape, p_target.requires_grad, p_target))
        
#         model.update_batch_stats(False)
        
#         output = model(x)
        
#         loss = (-(p_target.detach() * F.log_softmax(output, 1)).sum(1)*mask).mean()
        unlabeled_loss = (-(p_target * F.log_softmax(y, 1)).sum(1)*mask).mean()
        #https://discuss.pytorch.org/t/get-the-gradient-of-the-network-parameters/50575
        unlabeled_loss.backward(retain_graph=True)
       
        unlabeled_grads = []
        for name, param in model.named_parameters():
            try:
                unlabeled_grads.append(param.grad.view(-1))
            except:
#                 print('{} no grad'.format(name))
                continue
        unlabeled_grads = torch.cat(unlabeled_grads)
#         print('unlabeled_grads shape : {}'.format(unlabeled_grads.shape))
        
        #zero out the gradients
        model.zero_grad()
        
        return unlabeled_loss, gt_mask, unlabeled_grads

    def __make_one_hot(self, y):
        return torch.eye(self.num_classes)[y].to(y.device)
