import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split

# from options import Options

def train_val_dataset(dataset, val_split=0.05):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    return datasets

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
        # transforms.CenterCrop((opt.crsize, opt.crsize)),
        transforms.Resize((opt.isize, opt.isize)),
        transforms.ToTensor(),
        # transforms.Normalize
    ])
    dataset = ImageFolder(opt.dataroot, transform)
    dataset = train_val_dataset(dataset)

    dataloader = {x: DataLoader(
        dataset = dataset[x],
        batch_size = opt.batch_size,
        shuffle = opt.shuffle,
        num_workers = int(opt.workers),
        drop_last = opt.droplast,
        worker_init_fn=(None if opt.manualseed == -1
            else lambda x: np.random.seed(opt.manualseed))
    ) for x in ['train', 'val']}

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


def load_test_data_humpyter():
    """ Load Test Data

    Args:
        opt ([type]): Argument Parser
    Raises:
        IOError: Cannot load dataset
    Returns:
        [type]: DataLoader
    """

    transform = transforms.Compose([
        transforms.CenterCrop((300,300)),
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        # transforms.Normalize
    ])

    dataset = ImageFolder('/opt/ml/input/new_test', transform)
    dataloader = DataLoader(
        dataset = dataset, 
        batch_size = 1,
        shuffle = False,
        num_workers = 0,
        drop_last = False,
        # worker_init_fn=(None if opt.manualseed == -1
        #     else lambda x: np.random.seed(opt.manualseed))
    )

    return dataloader

# if __name__ == '__main__':
#     opt = Options().parse()
#     dataloader = load_data(opt)
#     print(dataloader)

#     print(iter(dataloader))
#     x, y = next(iter(dataloader['train']))
#     x_, y_ = next(iter(dataloader['val']))

#     print(f'{x.shape} {y.shape}')
#     print(f'{x_.shape} {y_.shape}')