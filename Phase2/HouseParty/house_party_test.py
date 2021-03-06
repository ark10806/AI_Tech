import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
from tqdm import tqdm
import os

from EnModels import MaskModel, GenderModel, AgeModel
import data
from options import Options
import pandas as pd

'''
##
# To-do
1. Custom dataset -> 
'''
opt = Options().parse()

### Load Data
# mask_loader = data.MaskLoader(dataroot=opt.dataroot, isTrin=False, batch_size=opt.batchsize)
# gend_loader = data.GenderLoader(dataroot=opt.dataroot, isTrin=False, batch_size=opt.batchsize)
# age_loader = data.AgeLoader(dataroot=opt.dataroot, isTrin=False, batch_size=opt.batchsize)

### Load Model
Mask = MaskModel(opt)
Gender = GenderModel(opt)
Age = AgeModel(opt)

m_weight = "MASK1_11_97.pth"
g_weight = "MASK1_10_'84.pth"
a_weight = "AgeModel1_6_85.pth"



### Test Model
mas_cls = Mask.test(m_weight)
# del mask_loader

gen_cls = Gender.test(g_weight, mas_cls)
# del gend_loader

# gen_cls = None

pred_dict = Age.test(a_weight, gen_cls)
# del age_loader




sheet = pd.read_csv('/opt/ml/input/data/eval/info.csv')
tmparr = []
for k in sheet['ImageID']:
    tmparr.append(pred_dict[k])
sheet['ans'] = tmparr
sheet.to_csv('./sub.csv', index=False)


        # submission.to_csv(f'./submission.csv', index=False)



### Get Prediction
# output = mas_cls*6 + gen_cls*3 + age_cls
# output = Enigma(output)
# write_submission(output)