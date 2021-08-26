import pandas as pd

origin = '/opt/ml/input/data/eval/info.csv'
subm = pd.read_csv(origin)
subm_dict = {}
for k, v in zip(subm['ImageID'], subm['ans']):
    subm_dict[k] = v


middle = '/opt/ml/code/AI_Tech/sam/example/test.csv'
mid = pd.read_csv(middle)
mid_dict = {}
for k, v in zip(mid['ImageID'], mid['ans']):
    mid_dict[k] = v




for k in subm_dict.keys():
    subm_dict[k] = mid_dict[k]

sav = pd.DataFrame(subm_dict.items(), columns=['ImageID', 'ans'])
sav.to_csv('./heey.csv', index=False)
