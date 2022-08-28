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
        
class PiModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y, model, mask):
        # NOTE:
        # stochastic transformation is embeded in forward function
        # so, pi-model is just to calculate consistency between two outputs

#         print('y.softmax(1) {} requires_grad {} is {}'.format(y.softmax(1).shape, y.softmax(1).requires_grad, y.softmax(1)))
#         print('mask {} requires_grad {} is {}'.format(mask.shape, mask.requires_grad, mask))
        
        masked_y_predictions = y.softmax(1)[mask]
#         masked_y_predictions = y.softmax(1)[-50:,:]
#         masked_y_predictions = y.softmax(1) * mask.unsqueeze(1)
#         masked_y_predictions = torch.masked_select(y.softmax(1), mask.unsqueeze(1))
#         print('masked_y_predictions {} requires_grad {} is {}'.format(masked_y_predictions.shape, masked_y_predictions.requires_grad, masked_y_predictions))
#         model.update_batch_stats(False) # Take only first call to update batch norm. as in MixMatch repo
        model.apply(freeze_bn) # Take only first call to update batch norm. as in MixMatch repo
        y_hat = model(x)
        model.apply(activate_bn)
#         model.update_batch_stats(True)
        
        return F.mse_loss(y_hat.softmax(1), masked_y_predictions, reduction="none").mean(1).mean()

#         return (F.mse_loss(y_hat.softmax(1), masked_y_predictions, reduction="none").mean(1) * mask).mean()
