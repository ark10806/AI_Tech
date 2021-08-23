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
##

if __name__ == "__main__":
    opt = Options().parse()

    initialize(opt, seed=42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # dataset = Cifar(opt.batch_size, opt.threads)
    dataset = dataset.load_test_data(opt)
    log = Log(log_each=10)
    model = WideResNet(opt.fs, opt.depth, opt.width_factor, opt.dropout, in_channels=3, labels=opt.n_class).to(device)
    ckpt_path = os.path.join(opt.ckpt_path, opt.ckpt_test)
    ckpt = torch.load(ckpt_path)
    model.state_dict(ckpt['state_dict'])

    base_optimizer = torch.optim.SGD
    optimizer = SAM(model.parameters(), base_optimizer, rho=opt.rho, adaptive=opt.adaptive, lr=opt.learning_rate, momentum=opt.momentum, weight_decay=opt.weight_decay)
    scheduler = StepLR(optimizer, opt.learning_rate, opt.epochs)

    for epoch in range(opt.epochs):
        # model.train()
        # log.train(len_dataset=len(dataset))

        # for batch in dataset:
        #     inputs, targets = (b.to(device) for b in batch)

        #     # first forward-backward step
        #     predictions = model(inputs)
        #     loss = smooth_crossentropy(predictions, targets)
        #     current_loss = loss.mean()
        #     loss.mean().backward()
        #     optimizer.first_step(zero_grad=True)

        #     # second forward-backward step
        #     smooth_crossentropy(model(inputs), targets).mean().backward()
        #     optimizer.second_step(zero_grad=True)

        #     with torch.no_grad():
        #         correct = torch.argmax(predictions.data, 1) == targets
        #         log(model, loss.cpu(), correct.cpu(), scheduler.lr())
        #         scheduler(epoch)
            

        #     if (epoch > 0) and (min_loss > current_loss):
        #         min_loss = current_loss

        #         pth_tmp = os.path.join(opt.ckpt_path, f'{opt.ckpt_name}.pth')
        #         torch.save({
        #             'epoch': epoch + 1,
        #             'state_dict': model.state_dict()}, 
        #             pth_tmp
        #         )

        model.eval()
        log.eval(len_dataset=len(dataset))

        with torch.no_grad():
            for batch in dataset:
                inputs, targets = (b.to(device) for b in batch)
                print(f'target: {targets}')

                predictions = model(inputs)
                print(predictions)
                loss = smooth_crossentropy(predictions, targets)
                correct = torch.argmax(predictions, 1) == targets
                log(model, loss.cpu(), correct.cpu())

    log.flush()
