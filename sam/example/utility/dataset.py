import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
# from options import Options


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
        transforms.CenterCrop((opt.crsize, opt.crsize)),
        transforms.Resize((opt.isize, opt.isize)),
        transforms.ToTensor(),
        # transforms.Normalize
    ])

    dataset = ImageFolder(opt.dataroot, transform)
    dataloader = DataLoader(
        dataset = dataset,
        batch_size = opt.batch_size,
        shuffle = opt.shuffle,
        num_workers = int(opt.workers),
        drop_last = opt.droplast,
        worker_init_fn=(None if opt.manualseed == -1
            else lambda x: np.random.seed(opt.manualseed))
    )

    return dataloader

def load_test_data(opt):
    """ Load Test Data

    Args:
        opt ([type]): Argument Parser
    Raises:
        IOError: Cannot load dataset
    Returns:
        [type]: DataLoader
    """

    transform = transforms.Compose([
        transforms.CenterCrop((opt.crsize, opt.crsize)),
        transforms.Resize((opt.isize, opt.isize)),
        transforms.ToTensor(),
        # transforms.Normalize
    ])

    dataset = ImageFolder(opt.testroot, transform)
    dataloader = DataLoader(
        dataset = dataset, 
        batch_size = opt.batch_size,
        shuffle = False,
        num_workers = int(opt.workers),
        drop_last = False,
        worker_init_fn=(None if opt.manualseed == -1
            else lambda x: np.random.seed(opt.manualseed))
    )

    return dataloader



# if __name__ == '__main__':
    # opt = Options().parse()
    # dataloader = load_data(opt)
    # print(dataloader)

    # print(iter(dataloader))