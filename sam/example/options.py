import argparse
import torch

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
        self.parser.add_argument('--batchsize', type=int, default=64, help='input batch size.')
        self.parser.add_argument('--droplast', action='store_true', default=True, help='Drop last batch size.')
        self.parser.add_argument('--shuffle', action='store_true', default=True, help='data shuffling')
        self.parser.add_argument('--isize', type=int, default=28, help='input image size.')
        self.parser.add_argument('--device', type=str, default='cpu', help='Device: gpu | cpu')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0 0,1,2 0,2. use -1 for CPU')
        self.parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
        self.parser.add_argument('--workers', type=int, default=0, help='num of workers')
        self.parser.add_argument('--manualseed', default=-1, type=int, help='manual seed')


        # Train
        self.parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        self.parser.add_argument('--epoch', type=int, default=500, help='number of epochs to train for')
        self.parser.add_argument('--momentum', type=float, default=0.5, help='momentum of SGD')
        self.parser.add_argument('--lr', type=float, default=1e-2, help='learning rate for SGD')
        self.opt = None


    def parse(self):
        """ Parse Arguments.
        """

        self.opt = self.parser.parse_args()
        
        self.opt.gpu_ids = [int(id) for id in self.opt.gpu_ids.split(',')]
        if self.opt.device == 'gpu':
            torch.cuda.set_device(self.opt.gpu_ids[0])
            print(torch.cuda.get_device_name(0))

        return self.opt