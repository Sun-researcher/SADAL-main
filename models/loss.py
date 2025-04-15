import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
import random
import numpy as np
import logging
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))



def adv(features, ad_net): #域判别损失
    ad_out = ad_net(features)
    batch_size = ad_out.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().to(features.device)
    return torch.nn.BCELoss()(ad_out, dc_target)


