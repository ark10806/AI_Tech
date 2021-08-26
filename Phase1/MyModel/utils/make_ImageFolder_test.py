import torch
import pandas as pd
import glob
import os
from PIL import Image
'''
input
    |
    -data
        |
        -eval
        -train
            |
            -train.csv
            -images
                |
                -006959_male_Asian_19
                    |
                    -normal
                    -incorrect_mask.jpg
                    -mask1.jpg, 2, 3, 4, 5
'''
base = '/opt/ml/input/'
src = os.path.join(base, 'data/eval')
dst = os.path.join(base, 'new_test')

if not os.path.isdir(dst):
    os.makedirs(dst)

tot = len(glob.glob(src + '/images/' + '**/*.*', recursive=True))
for idx, img in enumerate(glob.glob(src + '/images/' + '**/*.*', recursive=True)):
    dir_name, ext = os.path.splitext(img)
    if ext != '.jpg':
        print('not jpg!')

    dst_path = os.path.join(dst, dir_name.split('/')[-1])
    if os.path.isfile(dst_path + ext):
        print(f'{dst_path} exists!')
        exit()
    with Image.open(img, 'r') as im:
        im.save(dst_path + ext)

    if (idx+1) % 100 == 0:
        print(f'[{idx+1: ^7} / {tot}: {(idx+1) / tot * 100: .2f}%]\t {dst_path.split("/")[-1]}')

print(len(glob.glob(src + '/images/' + '**/*.*', recursive=True)))
print(len(glob.glob(dst + '**/*.*', recursive=True)))