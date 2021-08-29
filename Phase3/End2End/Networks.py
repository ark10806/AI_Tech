import data_without_album as data
import timm
from tqdm import tqdm
import torch.nn as nn
import torch
import numpy as np
import os

ex_num = 2

batch_size = 64
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
# print(model)

normal_data = data.NormalLoader(isTrain=True, batch_size=batch_size)
# normal_data = data.ProjectedLoader(name='age', isTrain=True, batch_size=batch_size)

optim = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

if not os.path.isdir('./weight'):
    os.makedirs('./weight')



# def logger(msg: str):
#     with open('./log{ex_num}', 'w') as f:
#         f.wrtite(msg)

best_accuracy = 0
for epoch in tqdm(range(n_epochs)):
    model.to(device)
    # model.train()
    for images, labels in tqdm(normal_data['train']):
        images = images.to(device)
        labels = labels.to(device)
        # print(labels)
        
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

            if best_accuracy * 100 > 70:
                ckpt_path = os.path.join('./weight', f'{ex_num}{epoch}{int(best_accuracy*100)}.pth')
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict()},
                    ckpt_path)
                print(f'\t{ckpt_path.split("/")[-1]} saved!')
            
        model.train()
    # loggeer(f'[Epoch: {epoch:<3}]\t {correct/total * 100: .2f}  {loss:.6f}\n')

