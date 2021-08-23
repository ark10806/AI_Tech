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
# for class_num in range(18):
#     dir_name = os.path.join(dst, str(class_num))
#     if not os.path.isdir(dir_name):
#         os.makedirs(dir_name)
#         print(f'{dir_name} created!')

# def magical_hat(fe: list)-> str:
#     if fe[0] == 'm':    fe[0] = 0
#     elif fe[0] == 'i':  fe[0] = 1
#     elif fe[0] == 'n':  fe[0] = 2
#     else: raise Exception('feature mask error')

#     if fe[1] == 'male': fe[1] = 0
#     elif fe[1] == 'female': fe[1] = 1
#     else: raise Exception('feature gender error')

#     if fe[2] < 30:  fe[2] = 0
#     elif 30 <= fe[2] and fe[2] < 60: fe[2] = 1
#     else:  fe[2] = 2

#     griffindor = fe[0] * 6 + fe[1] * 3 + fe[2]
#     assert 0 <= griffindor and griffindor <= 17, f'{fe}: {griffindor}'
#     return str(griffindor)

tot = len(glob.glob(src + '/images/' + '**/*.*', recursive=True))
for idx, img in enumerate(glob.glob(src + '/images/' + '**/*.*', recursive=True)):
    dir_name, ext = os.path.splitext(img)

    dst_path = os.path.join(dst, dir_name.split('/')[-1])
    if os.path.isfile(dst_path + '.jpg'):
        print(f'{dst_path} exists!')
        exit()
    with Image.open(img, 'r') as im:
        im.save(dst_path + '.jpg')

    if (idx+1) % 100 == 0:
        print(f'[{idx+1: ^7} / {tot}: {(idx+1) / tot * 100: .2f}%]\t {dst_path.split("/")[-1]}')

print(len(glob.glob(src + '/images/' + '**/*.*', recursive=True)))
print(len(glob.glob(dst + '**/*.jpg', recursive=True)))