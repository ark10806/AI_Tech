import glob
import os
from tqdm import tqdm

train = '/opt/ml/input/purified/train/'

cls = []

for i in tqdm(range(17)):
    tmp = 0
    tmp += len(glob.glob( train + str(i) + '/' + '*.jpg', recursive=True))
    tmp += len(glob.glob( train + str(i) + '/' + '*.png', recursive=True))
    tmp += len(glob.glob( train + str(i) + '/' + '*.jpeg', recursive=True))
    cls.append(tmp)

print(cls)