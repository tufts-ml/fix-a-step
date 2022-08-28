import torch
import torch.nn as nn
import torch.nn.functional as F

class MT(nn.Module):
    def __init__(self):
        super().__init__()
       

    def forward(self, x, y, model, ema_model, batch_size):
        
#         print('y requires_grad: {}'.format(y.requires_grad))
        y_hat = ema_model(x)
#         print('y_hat requires_grad: {}'.format(y_hat.requires_grad))
        
        masked_y_predictions = y.softmax(1)[batch_size:, :]
        
        unlabeled_loss = F.mse_loss(masked_y_predictions, y_hat.softmax(1).detach(), reduction="none").mean(1).mean()
        
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
