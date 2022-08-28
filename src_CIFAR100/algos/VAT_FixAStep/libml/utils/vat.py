import torch
import torch.nn as nn
import torch.nn.functional as F

#https://github.com/YU1ut/MixMatch-pytorch/issues/43
def freeze_bn(m):
    if isinstance(m, nn.BatchNorm2d):
        m.track_running_stats=False

def activate_bn(m):
    if isinstance(m, nn.BatchNorm2d):
        m.track_running_stats=True
        
        
class VAT(nn.Module):
    def __init__(self, eps=1.0, xi=1e-6, n_iteration=1):
        super().__init__()
        self.eps = eps
        self.xi = xi
        self.n_iteration = n_iteration

    def kld(self, q_logit, p_logit):
        q = q_logit.softmax(1)
        qlogp = (q * self.__logsoftmax(p_logit)).sum(1)
        qlogq = (q * self.__logsoftmax(q_logit)).sum(1)
        return qlogq - qlogp

    def normalize(self, v):
#         print('v.shape: {}'.format(v.shape))
#         print('range(1, len(v.shape)): {}'.format(range(1, len(v.shape))))
#         print('self.__reduce_max(v.abs(), range(1, len(v.shape))) is {}'.format(self.__reduce_max(v.abs(), range(1, len(v.shape)))))
        v = v / (1e-12 + self.__reduce_max(v.abs(), range(1, len(v.shape))))
#         print('v.pow(2).sum((1,2,3),keepdim=True) is {}'.format(v.pow(2).sum((1,2,3),keepdim=True)))
        v = v / (1e-6 + v.pow(2).sum((1,2,3),keepdim=True)).sqrt()
        return v

    def forward(self, x, y, model, batch_size):
#         print('lvl1: y.requires_grad: {}'.format(y.requires_grad))
#         model.update_batch_stats(False)
        model.apply(freeze_bn)
        d = torch.randn_like(x)
#         print('d=torch.randn_like(x) shape {}, is {}'.format(d.shape, d))
        d = self.normalize(d)
#         print('d=self.normalize(d) shape {} is {}'.format(d.shape, d))
        for _ in range(self.n_iteration):
            d.requires_grad = True
            x_hat = x + self.xi * d
            y_hat = model(x_hat)
#             kld = self.kld(y.detach(), y_hat).mean()
            kld = self.kld(y.detach(), y_hat).mean()
#             print('lvl2: y.requires_grad: {}'.format(y.requires_grad))
            d = torch.autograd.grad(kld, d)[0]
#             print('d=torch.autograd.grad(kld, d)[0] shape {} is {}'.format(d.shape, d))            
            d = self.normalize(d).detach()
#             kld.backward()
#             a = d.grad
#             print('a is {}'.format(a))
        
        x_hat = x + self.eps * d
        y_hat = model(x_hat)
        # NOTE:
        # see issue https://github.com/brain-research/realistic-ssl-evaluation/issues/27
        unlabeled_loss = self.kld(y_hat[batch_size:,:], y.detach()[batch_size:,:]).mean()
#         print('lvl3: y.requires_grad: {}'.format(y.requires_grad))
#         model.update_batch_stats(True)
        model.apply(activate_bn)
    
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

        return unlabeled_loss, unlabeled_grads

    def __reduce_max(self, v, idx_list):
        for i in idx_list:
            v = v.max(i, keepdim=True)[0]
#             print('inside __reduce_max, i{}, v: {}'.format(i, v))
        return v

    def __logsoftmax(self,x):
        xdev = x - x.max(1, keepdim=True)[0]
        lsm = xdev - xdev.exp().sum(1, keepdim=True).log()
        return lsm