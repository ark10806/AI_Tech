import argparse
import os
# from torch.cuda import get_device
from torch.cuda import set_device, get_device_name

class Options():
    """ Options class

    Returns:
        [argparse]: argparse containing model options
    """

    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        #
        self.parser.add_argument('--dataset', default='cifar10', help='cifar10 |')
        self.parser.add_argument('--dataroot', default='./data/')
        self.parser.add_argument('--batchsize', type=int, default=4, help='batch size')
        self.parser.add_argument('--droplast', action='store_true', default=True, help='True | False')
        self.parser.add_argument('--isize', type=int, default=32, help='28 for MNIST, 32 for cifar')
        self.parser.add_argument('--device', type=str, default='cpu', help='gpu | cpu')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='0 | 0,1,2 | 0,2. use -1 for cpu')
        self.parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
        
        # Train
        self.parser.add_argument('--exname', type=str, default='SimCLR', help='name of the experiment')
        self.parser.add_argument('--outf', type=str, default='output', help='name of dir to save outputs')
        self.parser.add_argument('--logdir', type=str, default='logs', help='name of dir to save logs')
        self.parser.add_argument('--pth_name_tmp', type=str, default='tmp', help='state_dict shared by Contra and FineTune')
        self.parser.add_argument('--pth_name_fine', type=str, default='fine', help='state_dict of FineTuned model')
        self.parser.add_argument('--print_freq', type=int, default=100, help='freq of showing results on CLI')
        self.parser.add_argument('--warmup_epoch', type=int, default=10, help='num of epochs to train for')
        self.parser.add_argument('--epoch', type=int, default=100, help='num of epochs to train for')
        self.parser.add_argument('--warmup_lr', type=float, default=0, help='warm-up lr for LARS')
        self.parser.add_argument('--base_lr', type=float, default=0.3, help='base lr for LARS')
        self.parser.add_argument('--final_lr', type=float, default=0, help='final lr for LARS')
        self.parser.add_argument('--lr', type=float, default=1e-4, help='lr for LARS')
        self.parser.add_argument('--momentum', type=float, default=0.5, help='momentum for LARS')
        self.parser.add_argument('--decay', type=float, default=1.e-6, help='weight decay')

        self.parser.add_argument('--label_portion', type=float, default=15, help='percentage of albeled data used for supervised training')

        self.opt = None

    def parse(self):
        """ Parse args
        """

        self.opt = self.parser.parse_args()

        self.opt.gpu_ids = sorted([int(id) if int(id)>0 else 0 for id in set(self.opt.gpu_ids.split(','))])
        if self.opt.device == 'gpu':
            set_device(self.opt.gpu_ids[0])
            print(get_device_name(0))

        expr_path = os.path.join(self.opt.outf, self.opt.exname)
        log_path = os.path.join(expr_path, self.opt.logdir)
        train_path = os.path.join(expr_path, 'train')
        test_path = os.path.join(expr_path, 'test')
        self.ckpt_path = os.path.join(train_path, 'ckpts')
        if not os.path.isdir(self.opt.outf):
            os.makedirs(self.opt.outf)
        if not os.path.isdir(expr_path):
            os.makedirs(expr_path)
        if not os.path.isdir(log_path):
            os.makedirs(log_path)
        if not os.path.isdir(train_path):
            os.makedirs(train_path)
        if not os.path.isdir(self.ckpt_path):
            os.makedirs(self.ckpt_path)
        if not os.path.isdir(test_path):
            os.makedirs(test_path)

        file_name = os.path.join(expr_path, 'opt.txt')
        with open(file_name, 'wt') as f:
            f.write('------------ Options -------------\n')
            for k, v in sorted(vars(self.opt).items()):
                f.write(f'{str(k): <20}: {str(v)}\n')
            f.write('-------------- End ---------------\n')
        return self.opt


if __name__ == '__main__':
    opts = Options()
    opts.parse()