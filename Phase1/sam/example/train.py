import argparse
import torch
import os
import numpy as np

from model.wide_res_net import WideResNet
from model.smooth_cross_entropy import smooth_crossentropy
# from data.cifar import Cifar
from utility.log import Log
from utility.initialize import initialize
from utility.step_lr import StepLR
import sys; sys.path.append("..")
from sam import SAM
from utility.options import Options
from utility import dataset
##
'''

'''
min_loss = np.inf
max_acc = 0

if __name__ == "__main__":
    opt = Options().parse()

    # initialize(opt, seed=42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # dataset = Cifar(opt.batch_size, opt.threads)
    dataset = dataset.load_data(opt)
    print(len(dataset['train']))
    print(len(dataset['val']))
    log = Log(log_each=10)
    model = WideResNet(opt.fs, opt.depth, opt.width_factor, opt.dropout, in_channels=3, labels=opt.n_class).to(device)
    ckpt_path = os.path.join(opt.ckpt_path, opt.ckpt_test)
    print(ckpt_path)
    ckpt = torch.load(ckpt_path + '.pth')
    model.load_state_dict(ckpt['state_dict'])

    base_optimizer = torch.optim.SGD
    optimizer = SAM(model.parameters(), base_optimizer, rho=opt.rho, adaptive=opt.adaptive, lr=opt.learning_rate, momentum=opt.momentum, weight_decay=opt.weight_decay)
    scheduler = StepLR(optimizer, opt.learning_rate, opt.epochs)

    for epoch in range(opt.epochs):
        model.train()
        log.train(len_dataset=len(dataset['train']))

        for batch in dataset['train']:
            inputs, targets = (b.to(device) for b in batch)
            # print(targets)

            # first forward-backward step
            predictions = model(inputs)
            loss = smooth_crossentropy(predictions, targets)
            loss.mean().backward()
            optimizer.first_step(zero_grad=True)

            # second forward-backward step
            pred_train = model(inputs)
            pred_train = torch.argmax(pred_train, 1)
            # for idx, tg in enumerate(targets):
            #     print(f'\t{tg:^4},{pred_train[idx]:^4}\t|', end='')
            # print()
            smooth_crossentropy(model(inputs), targets).mean().backward()
            optimizer.second_step(zero_grad=True)

            with torch.no_grad():
                correct = torch.argmax(predictions.data, 1) == targets
                log(model, loss.cpu(), correct.cpu(), scheduler.lr())
                scheduler(epoch)
                # print(correct.shape)
                # print(curr_acc)

        model.eval()
        log.eval(len_dataset=len(dataset['val']))

        with torch.no_grad():
            for batch in dataset['val']:
                inputs, targets = (b.to(device) for b in batch)

                predictions = model(inputs)
                loss = smooth_crossentropy(predictions, targets)
                current_loss = loss.mean()
                correct = torch.argmax(predictions, 1) == targets
                log(model, loss.cpu(), correct.cpu())
                curr_acc = (correct.cpu().sum().item() / opt.batch_size) * 100
        
        if (epoch>5 and min_loss>current_loss and max_acc+5<curr_acc) or \
            (epoch>10 and min_loss>current_loss and max_acc+0.1<curr_acc):
            min_loss = current_loss
            max_acc = curr_acc

            pth_tmp = os.path.join(opt.ckpt_path, f"{opt.ckpt_name + '_'+ str(epoch)+'_'}{int(curr_acc*100)}.pth")
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict()}, 
                pth_tmp
                )

    log.flush()
