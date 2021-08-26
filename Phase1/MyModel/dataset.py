import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from options import Options


def load_data(opt):
    """ Load Data

    Args:
        opt ([type]): Argument Parser
    Raises:
        IOError: Cannot load dataset
    Returns:
        [type]: DataLoader
    """

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(opt.isize),
        # transforms.Normalize
    ])

    dataset = ImageFolder(opt.dataroot, transform)
    dataloader = DataLoader(
        dataset = dataset,
        batch_size = opt.batchsize,
        shuffle = opt.shuffle,
        num_workers = int(opt.workers),
        drop_last = opt.droplast,
        worker_init_fn=(None if opt.manualseed == -1
            else lambda x: np.random.seed(opt.manualseed))
    )

    return dataloader


if __name__ == '__main__':
    opt = Options().parse()
    dataloader = load_data(opt)
    print(dataloader)