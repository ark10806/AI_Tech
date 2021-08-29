from torch.cuda import is_available
import torchvision
import torch
import numpy as np
import torchvision.transforms as T
from torch import nn

import data
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.Mask = torchvision.models.resnet18(pretrained=True)
        self.Mask.fc = nn.Sequential(
            nn.Linear(in_features=256, out_features=128),
            nn.Linear(in_features=128, out_features=64),
            nn.Linear(in_features=32, out_features=16)
        )

        self.Gend = torchvision.models.resnet18(pretrained=True)
        self.Gend.fc = nn.Sequential(
            nn.Linear(in_features=512, out_features=256),
            nn.Linear(in_features=256, out_features=64),
            nn.Linear(in_features=64, out_features=32)
        )

        self.Age = torchvision.models.resnet18(pretrained=True)
        self.Age.fc = nn.Sequential(
            nn.Linear(in_features=1024, out_features=512),
            nn.Linear(in_features=512, out_features=256),
            nn.Linear(in_features=128, out_features=64),
            nn.Linear(in_features=64, out_features=32)
        )

        self.fc1 = nn.Linear(in_features=64, out_features=32)
        self.fc2 = nn.Linear(in_features=48, out_features=32)
        self.fc3 = nn.Linear(in_features=48, out_features=18)    
        
    def forward(self, x):
        feat = self.fc1(torch.cat(self.Gend(x), self.Age(x)))
        output = self.fc2(torch.cat(x, self.Mask(x)))


        return output

classifier = MyModel()
dataloader = data.VanilaLoader(128)
loss_fn = torch.nn.CrossEntropyLoss()
optim = torch.optim.SGD(classifier.parameters, lr=1e-4, momentum=0.99)

epoch = 100
best_acc = 0.
best_loss = np.inf
for ep in range(epoch):
    for phase in ['train', 'val']:
        tmp_loss = 0
        tmp_acc = 0
    
        if phase == 'train':
            classifier.train()
        elif phase == 'val':
            classifier.eval()
        
        for cnt, (images, labels) in enumerate(tqdm(dataloader[phase])):
            images = images.to(device)
            labels = labels.to(device)

            optim.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                logits = classifier(images)
                _, preds = torch.max(logits, 1)
                loss = loss_fn(logits, labels)

                if phase == 'train':
                    loss.backward()
                    optim.step()
            
            tmp_loss += loss.item() * images.size(0)
            tmp_acc += torch.sum(preds == labels.data)
        loss = tmp_loss / len(dataloader[phase].dataset)
        acc = tmp_acc / len(dataloader[phase].dataset)

        if (phase == 'val') and best_acc < acc:
            best_acc = acc
        if (phase == 'val') and best_acc > acc:
            best_loss = loss
        

