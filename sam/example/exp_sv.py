import argparse
import torch
import os
import numpy as np

from model.wide_res_net import WideResNet
from model.smooth_cross_entropy import smooth_crossentropy
from data.cifar import Cifar
from utility.log import Log
from utility.initialize import initialize
from utility.step_lr import StepLR
import sys; sys.path.append("..")
from sam import SAM
from utility.options import Options
from utility import dataset

import csv

csv_path = '/opt/ml/input/new_test/info.csv'
csv_path = './test.csv'
##

if __name__ == "__main__":
    opt = Options().parse()

    # initialize(opt, seed=42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # dataset = Cifar(opt.batch_size, opt.threads)
    dataset = dataset.load_test_data(opt)
    log = Log(log_each=10)
    model = WideResNet(opt.fs, opt.depth, opt.width_factor, opt.dropout, in_channels=3, labels=opt.n_class).to(device)
    ckpt_path = os.path.join(opt.ckpt_path, opt.ckpt_test)
    print(ckpt_path)
    ckpt = torch.load(ckpt_path + '.pth')
    model.load_state_dict(ckpt['state_dict'])

    base_optimizer = torch.optim.SGD
    optimizer = SAM(model.parameters(), base_optimizer, rho=opt.rho, adaptive=opt.adaptive, lr=opt.learning_rate, momentum=opt.momentum, weight_decay=opt.weight_decay)
    scheduler = StepLR(optimizer, opt.learning_rate, opt.epochs)

    # for epoch in range(opt.epochs):

    model.eval()
    # log.eval(len_dataset=len(dataset))


    with open(csv_path, 'w', newline='\n') as csvfile:
        wr = csv.writer(csvfile, delimiter=',')
        with torch.no_grad():
            for i, batch in enumerate(dataset):
                inputs, targets = (b.to(device) for b in batch)
                # print(f'target: {targets}')
                fname, _ = dataset.dataset.samples[i]
                fname = fname.split("/")[-1]

                predictions = model(inputs)
                pp = int(torch.argmax(predictions,1))
                # print(targets.shape, pp.shape, end='')
                print(f'[i:{i}]\t{fname.split("/")[-1]},{int(pp):>3}', end='\t')
                print(f'\t{int(targets):^3},{pp:^3}\t')
                print(f'[{(i+1)/12600*100: .2f}{i+1 :>7} / 12600]')
                for k in range(len(targets)):
                    wr.writerow([fname, int(pp)])
                loss = smooth_crossentropy(predictions, targets)
                correct = torch.argmax(predictions, 1) == targets
                # print(torch.argmax(predictions, 1))
                # log(model, loss.cpu(), correct.cpu())

# log.flush()
