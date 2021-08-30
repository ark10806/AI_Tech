import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score

import Data
import Losses
import Logger
import Networks

'''
multi-label classification
lr_scheduler
'''

batch_size = 128
n_epochs = 30
lr = 1e-4
loss = 5
f1 = 0.1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f'[{torch.cuda.get_device_name()}]')

model = Networks.Efficientnet_b2()
# normal_data = Data.NormalLoader(isTrain=True, batch_size=batch_size)
normal_data = Data.load_data(True, batch_size, name=None, expand=True)
# normal_data = Data.NormalDataset(isTrain=True).get_loader(batch_size=batch_size)
# normal_data = data.ProjectedLoader(name='age', isTrain=True, batch_size=batch_size)

optim = torch.optim.Adam(model.parameters(), lr=lr)
criterion = Losses.FocalLoss()


print(f"[{'Epoch':<5} {'LR':<6} {'F1': <5} {'Loss':<6} {'Acc':<5}]")
best_accuracy = 10
for epoch in range(n_epochs):
    model.to(device)
    for images, labels in tqdm(normal_data['train'],bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', desc=f'[{epoch:^5} {lr*1000::^1.4f} {f1*100:>.2f} {loss:^5.4f} {best_accuracy*100:3.2f}%]'):
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        optim.zero_grad()
        loss.backward()
        optim.step()

    model.eval()
    with torch.no_grad():
        correct, total = 0, 0
        for images, labels in normal_data['val']:
            images = images.to(device)
            
            outputs = model(images).detach().cpu()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            f1 = f1_score(labels, predicted, average='macro')
        
        if best_accuracy> correct / total:
            pass

        curr_acc = correct / total
        if best_accuracy < curr_acc:
            best_accuracy = curr_acc
            if best_accuracy * 100 > 95:
                Logger.save_weights(state_dict=model.state_dict(), epoch=epoch, f1=f1, acc=best_accuracy)

    Logger.write_logs(epoch, lr, f1, loss, curr_acc)
    model.train()