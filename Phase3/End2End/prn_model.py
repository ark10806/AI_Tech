import data
import timm
from tqdm import tqdm
import torch.nn as nn
import torch
import numpy as np
import os

batch_size = 128
n_classes = 18
n_epochs = 30
lr = 1e-4
model = timm.create_model("efficientnet_b4", pretrained=True)
for param in model.parameters():
    param.requires_grad = True
outputs_attrs = n_classes
num_inputs = model.classifier.in_features
last_layer = nn.Linear(num_inputs, outputs_attrs)
model.classifier = last_layer
print(model)