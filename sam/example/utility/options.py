import argparse
import torch
import os

class Options():
    """Options class

    Returns:
        [argparse]: argparse containing train and test options
    """

    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        # Base
        self.parser.add_argument('--dataset', default='new', help='MNIST | new')
        self.parser.add_argument('--dataroot', default='/opt/ml/input/new')
        self.parser.add_argument('--testroot', default='/opt/ml/input/new_test')
        # self.parser.add_argument('--batchsize', type=int, default=64, help='input batch size.')
        self.parser.add_argument('--droplast', action='store_true', default=True, help='Drop last batch size.')
        self.parser.add_argument('--shuffle', action='store_true', default=True, help='data shuffling')
        self.parser.add_argument('--isize', type=int, default=32, help='input image size.')
        self.parser.add_argument('--crsize', type=int, default=300, help='Center crop size.')
        self.parser.add_argument('--device', type=str, default='cpu', help='Device: gpu | cpu')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0 0,1,2 0,2. use -1 for CPU')
        self.parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
        self.parser.add_argument('--workers', type=int, default=0, help='num of workers')
        self.parser.add_argument('--manualseed', default=-1, type=int, help='manual seed')


        # Train
        self.parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        self.parser.add_argument('--ckpt_path', type=str, default=os.path.join(os.getcwd(), 'weights'), help='path to save checkpoints')
        self.parser.add_argument('--ckpt_name', type=str, default='2', help='checkpoint name')
        self.parser.add_argument('--ckpt_test', type=str, default='1', help='ckpt for test')
        # self.parser.add_argument('--epoch', type=int, default=500, help='number of epochs to train for')
        # self.parser.add_argument('--momentum', type=float, default=0.5, help='momentum of SGD')
        # self.parser.add_argument('--lr', type=float, default=1e-2, help='learning rate for SGD')

        self.parser.add_argument("--adaptive", default=True, type=bool, help="True if you want to use the Adaptive SAM.")
        self.parser.add_argument("--batch_size", default=32, type=int, help="Batch size used in the training and validation loop.")
        self.parser.add_argument("--n_class", default=18, type=int, help="number of classes")
        self.parser.add_argument('--fs', type=int, default=16, help='filter size to downsample')
        self.parser.add_argument("--depth", default=16, type=int, help="Number of layers.")
        self.parser.add_argument("--dropout", default=0.0, type=float, help="Dropout rate.")
        self.parser.add_argument("--epochs", default=200, type=int, help="Total number of epochs.")
        self.parser.add_argument("--label_smoothing", default=0.1, type=float, help="Use 0.0 for no label smoothing.")
        self.parser.add_argument("--learning_rate", default=0.1, type=float, help="Base learning rate at the start of the training.")
        self.parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum.")
        self.parser.add_argument("--threads", default=2, type=int, help="Number of CPU threads for dataloaders.")
        self.parser.add_argument("--rho", default=0.5, type=int, help="Rho parameter for SAM.")
        self.parser.add_argument("--weight_decay", default=0.0005, type=float, help="L2 weight decay.")
        self.parser.add_argument("--width_factor", default=16, type=int, help="How many times wider compared to normal ResNet.")

        self.opt = None


    def parse(self):
        """ Parse Arguments.
        """

        self.opt = self.parser.parse_args()
        
        self.opt.gpu_ids = [int(id) for id in self.opt.gpu_ids.split(',')]
        if self.opt.device == 'gpu':
            torch.cuda.set_device(self.opt.gpu_ids[0])
            print(torch.cuda.get_device_name(0))

        if not os.path.isdir(self.opt.ckpt_path):
            os.makedirs(self.opt.ckpt_path)

        return self.opt