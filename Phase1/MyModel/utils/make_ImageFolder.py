import pandas as pd
import glob
import os
from PIL import Image
from tqdm import tqdm
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


''' ImageFolder
train
    |
    -0 (dir)
    -1
    -2
    |- imgs
'''
base = '/opt/ml/input/'
src = os.path.join(base, 'data/train')
dst = os.path.join(base, 'purified/train')

if not os.path.isdir(dst):
    os.makedirs(dst)
for class_num in range(18):
    dir_name = os.path.join(dst, str(class_num))
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
        print(f'{dir_name} created!')

def magical_hat(fe: list)-> str:
    if fe[0] == 'm':    fe[0] = 0
    elif fe[0] == 'i':  fe[0] = 1
    elif fe[0] == 'n':  fe[0] = 2
    else: raise Exception('feature mask error')

    if fe[1] == 'male': fe[1] = 0
    elif fe[1] == 'female': fe[1] = 1
    else: raise Exception('feature gender error')

    if fe[2] < 30:  fe[2] = 0
    elif 30 <= fe[2] and fe[2] < 60: fe[2] = 1
    else:  fe[2] = 2

    griffindor = fe[0] * 6 + fe[1] * 3 + fe[2]
    assert 0 <= griffindor and griffindor <= 17, f'{fe}: {griffindor}'
    return str(griffindor)

tot = len(glob.glob(src + '/images/' + '**/*.*', recursive=True))
for img in tqdm(glob.glob(src + '/images/' + '**/*.*', recursive=True)):
    dir_name, ext = os.path.splitext(img)
    is_masked = dir_name.split('/')[-1]
    etc = dir_name.split('/')[-2].split('_')
    etc.pop(2)
    features = [is_masked[0], etc[1], int(etc[2])]

    dst_cls = magical_hat(features)
    dst_path = os.path.join(dst, dst_cls, '_'.join(etc) + '_' + is_masked)

    if os.path.isfile(dst_path + ext):
        print(f'{dst_path} exists!')
        exit()
    with Image.open(img, 'r') as im:
        im.save(dst_path + ext)
    

new = '/opt/ml/input/purified/train'

print(len(glob.glob(new + '**/*.jpg', recursive=True)))