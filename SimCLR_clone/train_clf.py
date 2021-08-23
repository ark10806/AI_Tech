import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
from tqdm import tqdm
import os

from model import get_backbone, FineTunedModel, ContrastiveModel
from dataset_loader import get_clf_train_test_transform
from dataset_loader import get_clf_train_test_dataloaders
from logger import Logger
from opt import Options

opt = Options().parse()

backbone = get_backbone(resnet18(pretrained=False))
model = ContrastiveModel(backbone).to(opt.device)
obj_pth = os.path.join(opt.ckpt_path, opt.pth_name)
obj = torch.load(obj_pth)
model.load_state_dict(obj['state_dict'])

encoder = model.backbone
last_layers = torch.nn.Sequential(*(list(model.projectionhead.children())[0:2])) # check.

enoder = nn.Sequential(
    encoder,
    last_layers
)

new_model = FineTunedModel(encoder, model.output_dim).to(opt.device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(new_model.parameters(), lr=1e-4,
    momentum=0.99, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
best_acc = 0.0

train_loader, test_loader = get_clf_train_test_dataloaders(percent_train_sample=opt.label_portion)
train_transform, test_transform = get_clf_train_test_transform(opt.imsize)

def train_clf(ep, epoch):
    new_model.train()
    train_loss = 0
    correct = 0
    total = 0

    local_progress = tqdm(train_loader, desc=f'Epoch {ep}/{epoch}')
    for idx, (img, labels) in enumerate(local_progress):
        img, labels = img.to(opt.device), labels.to(opt.device)
        aug_img = train_transform(img)

        optimizer.zero_grad()
        outputs = new_model(aug_img)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)

        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        data_dict = {'loss': train_loss, 'acc': 100.*correct / total}
        local_progress.set_postfix(data_dict)

    return data_dict

def test_clf(ep, epoch):
    global best_acc

    new_model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        local_progress = tqdm(test_loader, desc=f'Epoch {ep}/{epoch}')
        for idx, (img, label) in enumerate(local_progress):
            img, label = img.to(opt.device), label.to(opt.device)
            img = test_transform(img)
            outputs = new_model(img)
            loss = criterion(outputs, label)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += label.size(0)
            correct += predicted.eq(label).sum().item()
            data_dict = {'test_loss': test_loss, 'test_acc': 100.*correct/total}
            local_progress.set_postfix(data_dict)

    acc = 100. * correct / total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': new_model.state_dict(),
            'acc': acc,
            'epoch': ep
        }

        pth_fine = opt.path.join(opt.ckpt_path, f'{opt.pth_name_fine}.pth')
        torch.save({
            'epoch': ep+1,
            'state_dict': new_model.state_dict()}, pth_fine)
        print(f'Fine saved at: {pth_fine}')
        best_acc = acc

    return data_dict

logger = Logger(log_dir=opt.logdir, tensorboard=True, matplotlib=True)

for ep in range(opt.epoch):
    data_dict = train_clf(ep, opt.epoch)
    logger.update_scalers(data_dict)
    data_dict = test_clf(ep, opt.epoch)
    logger.update_scalers(data_dict)
    scheduler.step()