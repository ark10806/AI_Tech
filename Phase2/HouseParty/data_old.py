import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
import os
import pandas as pd
from PIL import Image
from torchvision.transforms.transforms import RandomAutocontrast, RandomHorizontalFlip, RandomRotation


'''
RGB Mean: [0.56019358 0.52410121 0.501457  ]
RGB Standard Deviation: [0.23318603 0.24300033 0.24567522]

'''
# from options import Options

def train_val_dataset(dataset, val_split=0.2):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split, stratify=dataset.targets)
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
        transforms.RandomRotation((0,15)),
        transforms.CenterCrop((opt.crsize, opt.crsize)),
        transforms.Resize((opt.isize, opt.isize)),
        transforms.RandomAutocontrast(0.5),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
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

def load_val(opt):
    transform = transforms.Compose([
        transforms.RandomRotation((0,15)),
        transforms.CenterCrop((opt.crsize, opt.crsize)),
        transforms.Resize((opt.isize, opt.isize)),
        transforms.RandomAutocontrast(0.5),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
    ])
    dataset = ImageFolder(opt.dataroot, transform)

    return dataset


# def load_test_data(opt):
#     """ Load Test Data

#     Args:
#         opt ([type]): Argument Parser
#     Raises:
#         IOError: Cannot load dataset
#     Returns:
#         [type]: DataLoader
#     """

#     transform = transforms.Compose([
#         transforms.CenterCrop((opt.crsize, opt.crsize)),
#         transforms.Resize((opt.isize, opt.isize)),
#         transforms.ToTensor(),
#         # transforms.Normalize
#     ])

#     dataset = ImageFolder(opt.testroot, transform)
#     dataloader = DataLoader(
#         dataset = dataset, 
#         batch_size = opt.batch_size,
#         shuffle = False,
#         num_workers = int(opt.workers),
#         drop_last = False,
#         worker_init_fn=(None if opt.manualseed == -1
#             else lambda x: np.random.seed(opt.manualseed))
#     )

#     return dataloader, dataset

class TestDataset(Dataset):
    def __init__(self, img_paths, transform):
        self.img_paths = img_paths
        self.transform = transform

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_paths)

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
        transforms.RandomRotation((0,15)),
        transforms.CenterCrop((opt.crsize, opt.crsize)),
        transforms.Resize((opt.isize, opt.isize)),
        transforms.RandomAutocontrast(0.5),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
    ])

    test_dir = '/opt/ml/input/data/eval'
    submission = pd.read_csv(os.path.join(test_dir, 'info.csv'))
    img_dir = os.path.join(test_dir, 'images')
    
    img_paths = [os.path.join(img_dir, img_id) for img_id in submission.ImageID]
    dataset = TestDataset(img_paths, transform)

    loader = DataLoader(
        dataset,
        shuffle=False
    )
    return loader