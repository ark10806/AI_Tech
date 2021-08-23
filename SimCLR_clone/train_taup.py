import os
import time
from datetime import datetime
from tqdm import tqdm
import numpy as np
import torch
import torch.optim as optim
from torchvision.models import resnet18
from tensorboardX import SummaryWriter


from lars import LARS
from logger import Logger
from loss import ContrastiveLoss
from lr_scheduler import LR_Scheduler
from dataset_loader import gpu_transformer, get_train_mem_test_dataloaders
from model import ContrastiveModel, get_backbone
from knn_monitor import knn_monitor as accuracy_monitor
from opt import Options

if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
    device = torch.device('cuda')
    print(torch.cuda.get_device_name(0))
else:
    dtype = torch.FloatTensor
    device = torch.device('cpu')

opt = Options().parse()


logger = Logger(log_dir=opt.logdir, tensorboard=True, matplotlib=True)

backbone = get_backbone(resnet18(pretrained=False))
model = ContrastiveModel(backbone).to(opt.device)
loss_func = ContrastiveLoss().to(opt.device)

#hyperparams
features = 128
batch_size = batch = 2048
epochs = 25 #use num_epochs if you have time and resources to train. Else, for POC, 25 epochs should yield a decreasing loss. 
lr = 1e-4
device_id = 0
weight_decay  = 1.e-6

image_size = (32,32)
momentum = 0.9

warmup_epochs =  10
warmup_lr  =     0
base_lr =    0.3
final_lr =   0
num_epochs =     800 # this parameter influence the lr decay
stop_at_epoch =  100 # has to be smaller than num_epochs
batch_size =     256
knn_monitor =    False # knn monitor will take more time
knn_interval =   5
knn_k =      200

min_loss = np.inf
acc = 0

train_loader, mem_loader, test_loader = get_train_mem_test_dataloaders(
    dataset = opt.dataset,
    data_dir = opt.dataroot,
    batch_size = opt.batchsize,
    num_workers=0,
    download=True
)

train_transform, test_transform = gpu_transformer(opt.isize)

optm = LARS(model.named_modules(), lr=opt.lr*opt.batchsize/256, momentum=opt.momentum, weight_decay=opt.decay)

scheduler = LR_Scheduler(
    optim, opt.warmup_epoch, opt.warmup_lr*opt.batchsize/256,
    opt.epoch, opt.base_lr*opt.batchsize/256, opt.final_lr*opt.batchsize/256,
    len(train_loader),
    constant_predictor_lr=True
)

global_progress = tqdm(range(0, opt.epochs), desc='Training')
data_dict = {'loss': 100}
for ep in global_progress:
    model.train()
    local_progress = tqdm(train_loader, desc=f'Epoch {ep} / {opt.epoch}')

    for idx, (image, label) in enumerate(local_progress):
        img = image.to(opt.device)
        aug_img = train_transform(img)
        model.zero_grad()
        z = model.forward(img.to(opt.device, non_blocking=True))
        z_ = model.forward(aug_img.to(opt.device, non_blocking=True))
        loss = loss_func(z, z_)
        data_dict['loss'] = loss.item()
        loss.backward()
        optim.step()
        scheduler.step()
        data_dict.update({'lr': scheduler.get_last_lr()})
        local_progress.set_postfix(data_dict)
        logger.update_scalers(data_dict)

    current_loss = data_dict['loss']

    if ep % knn_interval == 0:
        acc = accuracy_monitor(model.backbone, mem_loader, test_loader, '-', hide_progress=True)
        data_dict['acc'] = acc
    
    global_progress.set_postfix(data_dict)
    logger.update_scalers(data_dict)

    pth_tmp = os.path.join(opt.ckpt_path, f'{opt.pth_name_tmp}.pth')

    if min_loss > current_loss:
        min_loss = current_loss

        torch.save({
            'epoch': ep + 1,
            'state_dict': model.state_dict()}, 
            pth_tmp
        )