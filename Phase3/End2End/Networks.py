import data as data
import timm
from tqdm import tqdm
import torch.nn as nn
import torch
import numpy as np
import os

ex_name = 'uniform'

batch_size = 128
n_classes = 18
n_epochs = 30
lr = 1e-4
model = timm.create_model("efficientnet_b2", pretrained=True)
for param in model.parameters():
    param.requires_grad = True
model.classifier.out_features=n_classes
stdv = 1 / model.classifier.in_features ** 0.5
model.classifier.bias.data.uniform_(-stdv, stdv)

normal_data = data.NormalLoader(isTrain=True, batch_size=batch_size)
# normal_data = data.ProjectedLoader(name='age', isTrain=True, batch_size=batch_size)

optim = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

if not os.path.isdir('./weight'):
    os.makedirs('./weight')
if not os.path.isdir('./logs'):
    os.makedirs('./logs')



def logger(fname:str, msg: str):
    if os.path.isfile(f'./log_{fname}'):
        with open(f'./logs/{fname}.txt', 'w') as f:
            f.write('Epoch\tAcc\tLoss')
    else:
        with open(f'./logs/{fname}.txt', 'a') as f:
            f.write(msg)

best_accuracy = 0
for epoch in range(n_epochs):
    model.to(device)
    # model.train()
    for images, labels in tqdm(normal_data['train'], desc=f'[Epoch: {epoch:>3}] {best_accuracy*100:.2f}%'):
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        optim.zero_grad()

        # Backward and optimize
        loss.backward()
        optim.step()

    model.eval()
    with torch.no_grad():
        print(f'\t[LOSS]: {loss}')
        correct = 0
        total = 0
        for images, labels in normal_data['val']:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        if best_accuracy> correct / total:
            # curr_lr = opt.learning_rate*np.asscalar(pow(np.random.rand(1),5))
            # update_lr(curr_lr)
            print('Epoch :{} Accuracy SpinalNet: ({:.2f}%), Maximum Accuracy: {:.2f}%'.format(epoch, 
                                            100 * correct / total, 100*best_accuracy))
        else:
            best_accuracy = correct / total
            print('Test Accuracy of SpinalNet: {} % (improvement)'.format(100 * correct / total))

            if best_accuracy * 100 > 90:
                ckpt_path = os.path.join('./weight', f'ex{ex_name}_{epoch+1}_{int(best_accuracy*100)}%.pth')
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict()},
                    ckpt_path)
                print(f'\t{ckpt_path.split("/")[-1]} saved!')
            
        model.train()
    logger(ex_name, f'{epoch+1:<3}\t{correct/total * 100: .2f}%\t{loss:.6f}\n')

