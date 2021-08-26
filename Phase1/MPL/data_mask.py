import logging
import math

import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from torchvision.datasets import ImageFolder

from augmentation import RandAugment

logger = logging.getLogger(__name__)

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)

transform = transforms.Compose([
    transforms.CenterCrop((300,300)),
    transforms.Resize((64,64)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

def x_u_split(args, labels):
    label_per_class = args.num_labeled // args.num_classes
    labels = np.array(labels)
    labeled_idx = []
    # unlabeled data: all training data
    unlabeled_idx = np.array(range(len(labels)))
    for i in range(args.num_classes):
        idx = np.where(labels == i)[0]
        idx = np.random.choice(idx, label_per_class, False)
        labeled_idx.extend(idx)
    labeled_idx = np.array(labeled_idx)
    assert len(labeled_idx) == args.num_labeled

    if args.expand_labels or args.num_labeled < args.batch_size:
        num_expand_x = math.ceil(
            args.batch_size * args.eval_step / args.num_labeled)
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
    np.random.shuffle(labeled_idx)
    return labeled_idx, unlabeled_idx


def train_val_dataset(dataset, val_split=0.15):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    return datasets

def get_new(args, opt):
    dataset = ImageFolder(opt.dataroot, transform)
    trainset = dataset['train']
    val_dataset = dataset['val']

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(args, trainset.targets)

    train_labeled_dataset = trainset[train_labeled_idxs]
    train_unlabeled_dataset = trainset[train_unlabeled_idxs]

    return train_labeled_dataset, train_unlabeled_dataset, val_dataset

DATASET_GETTERS = {'new': get_new}
