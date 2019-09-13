import torch 
from torch.nn import functional as F
from torch.autograd import Variable 

from third_party.mean_teacher import losses as mt_losses


def softmax_mse_loss(input_logits, target_logits):
    return mt_losses.softmax_mse_loss(input_logits, target_logits)


def softmax_kl_loss(input_logits, target_logits):
    return mt_losses.softmax_kl_loss(input_logits, target_logits)


def symmetric_mse_loss(input1, input2):
    return mt_losses.symmetric_mse_loss(input1, input2)
    