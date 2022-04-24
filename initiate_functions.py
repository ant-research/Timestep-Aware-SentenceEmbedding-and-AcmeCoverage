import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *
from config import _cuda
import numpy as np

def init_rnn_wt(rnn_model):
    for names in rnn_model._all_weights:
        for name in names:
            if name.startswith('weight_'):
                wt = getattr(rnn_model, name)
                wt.data.uniform_(-float(args['rand_unif_init_mag']), float(args['rand_unif_init_mag']))
            elif name.startswith('bias_'):
                # set forget bias to 1
                bias = getattr(rnn_model, name)
                n = bias.size(0)
                start, end = n // 3, int(n // 1.5)
                bias.data.fill_(0.)
                bias.data[start:end].fill_(1.)

def init_linear_wt(linear):
    linear.weight.data.normal_(std=float(args['trunc_norm_init_std']))
    if linear.bias is not None:
        linear.bias.data.normal_(std=float(args['trunc_norm_init_std']))

def init_wt_normal(wt):
    wt.data.normal_(std=float(args['trunc_norm_init_std']))

def init_wt_unif(wt):
    wt.data.uniform_(-float(args['rand_unif_init_mag']), float(args['rand_unif_init_mag']))

