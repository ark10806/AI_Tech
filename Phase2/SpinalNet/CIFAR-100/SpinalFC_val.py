import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import dataset
from options import Options
import random
from tqdm import tqdm
import os
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_epochs = 160
learning_rate = 0.001

torch.manual_seed(0)
random.seed(0)

first_HL = 512

opt = Options().parse()
dataloader = dataset.load_data(opt)
# train_loader = dataloader['train']
test_loader = dataloader['val']
# test_loader = dataset.load_test_data(opt)


def conv3x3(in_channels, out_channels, stride=1):
    """3x3 kernel size with padding convolutional layer in ResNet BasicBlock."""
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)

class BasicBlock(nn.Module):
    """Basic Block of ReseNet."""

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        """Basic Block of ReseNet Builder."""
        super(BasicBlock, self).__init__()

        # First conv3x3 layer
        self.conv1 = conv3x3(in_channels, out_channels, stride)

        #  Batch Normalization
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)

        # ReLU Activation Function
        self.relu = nn.ReLU(inplace=True)

        # Second conv3x3 layer
        self.conv2 = conv3x3(out_channels, out_channels)

        #  Batch Normalization
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)

        # downsample for `residual`
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """Forward Pass of Basic Block."""
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        return out

class SpinalResNet(nn.Module):
    """Residual Neural Network."""

    def __init__(self, block, duplicates, num_classes=100):
        """Residual Neural Network Builder."""
        super(SpinalResNet, self).__init__()

        self.in_channels = 32
        self.conv1 = conv3x3(in_channels=3, out_channels=32)
        self.bn = nn.BatchNorm2d(num_features=32)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=0.02)

        # block of Basic Blocks
        self.conv2_x = self._make_block(block, duplicates[0], out_channels=32)
        self.conv3_x = self._make_block(block, duplicates[1], out_channels=64, stride=2)
        self.conv4_x = self._make_block(block, duplicates[2], out_channels=128, stride=2)
        self.conv5_x = self._make_block(block, duplicates[3], out_channels=256, stride=2)

        self.maxpool = nn.MaxPool2d(kernel_size=4, stride=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        #self.fc_layer = nn.Linear(256, num_classes)
        
        self.fc1 = nn.Linear(256, first_HL) #changed from 16 to 8
        self.fc1_1 = nn.Linear(256 + first_HL, first_HL) #added
        self.fc1_2 = nn.Linear(256 + first_HL, first_HL) #added
        self.fc1_3 = nn.Linear(256 + first_HL, first_HL) #added
        
        self.fc_layer = nn.Linear(first_HL*4, num_classes)

        # initialize weights
        # self.apply(initialize_weights)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight.data, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_block(self, block, duplicates, out_channels, stride=1):
        """
        Create Block in ResNet.

        Args:
            block: BasicBlock
            duplicates: number of BasicBlock
            out_channels: out channels of the block

        Returns:
            nn.Sequential(*layers)
        """
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(num_features=out_channels)
            )

        layers = []
        layers.append(
            block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, duplicates):
            layers.append(block(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass of ResNet."""
        out = self.conv1(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)

        # Stacked Basic Blocks
        out = self.conv2_x(out)
        out = self.conv3_x(out)
        out = self.conv4_x(out)
        out = self.conv5_x(out)
        
        
        out1 = self.maxpool2(out)
        #print('out1',out1.shape)
        out2 = out1[:,:,0,0]
        #print('out2',out2.shape)
        out2 = out2.view(out2.size(0),-1)
        #print('out2',out2.shape)
        
        x1 = out1[:,:,0,0]
        x1 = self.relu(self.fc1(x1))
        x2= torch.cat([ out1[:,:,0,1], x1], dim=1)
        x2 = self.relu(self.fc1_1(x2))
        x3= torch.cat([ out1[:,:,1,0], x2], dim=1)
        x3 = self.relu(self.fc1_2(x3))
        x4= torch.cat([ out1[:,:,1,1], x3], dim=1)
        x4 = self.relu(self.fc1_3(x4))
        
        x = torch.cat([x1, x2], dim=1)
        x = torch.cat([x, x3], dim=1)
        out = torch.cat([x, x4], dim=1)
        
        out = self.fc_layer(out)

        return out



def SpinalResNet18():
    return SpinalResNet(BasicBlock, [2,2,2,2]).to(device)

def SpinalResNet34():
    return SpinalResNet(BasicBlock, [3, 4, 6, 3], num_classes=opt.n_class).to(device)

# For updating learning rate
def update_lr(optimizer, lr):    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Train the model
curr_lr2 = learning_rate

model2 = SpinalResNet34().to(device)

ckpt_path = os.path.join(opt.ckpt_path, opt.ckpt_test)
ckpt = torch.load(ckpt_path + '.pth')
model2.load_state_dict(ckpt['state_dict'])

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer2 = torch.optim.Adam(model2.parameters(), lr=learning_rate) 

# Train the model
best_accuracy2 =0

pred_dict = {}

model2.eval()

# 제출할 파일을 저장합니다.

# def write_submission(pred_dict):
#     origin = '/opt/ml/input/data/eval/info.csv'
#     subm = pd.read_csv(origin)
#     subm_dict = {}
#     for k, v in zip(subm['ImageID'], subm['ans']):
#         subm_dict[k] = v

#     for k in subm_dict.keys():
#         # print("@"*30)
#         # print(f'subm_dict = {subm_dict}')
#         # print("@"*30)
#         subm_dict[k] = pred_dict[k]
    
#     sav = pd.DataFrame(subm_dict.items(), columns=['ImageID', 'ans'])
#     sav.to_csv('./submission.csv', index=False)

def write_val(pred_dict):
    sav = pd.DataFrame(pred_dict.items(), columns=['ImageID', 'ans'])
    sav.to_csv('./val.csv', index=False)

sval = dataset.load_val(opt)

with torch.no_grad():
    for idx, (images, labels) in enumerate(tqdm(test_loader)):
        fname, _ = sval.samples[idx] 
        # fname, _ = test_loader.dataset.samples[idx] 

        ## To do: no shuffling on Test dataset
        ## To do: extentions on jupyter (O)
        # fname = fname.split('/')[-1]


        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model2(images)
        _, predicted = torch.max(outputs.data, 1)
        pp = int(predicted)
        # print("#"*30)
        # print(fname)
        # print("#"*30)
        pred_dict[fname] = pp
write_val(pred_dict)
