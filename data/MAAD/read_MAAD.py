import pandas as pd
import numpy as np
from PIL import Image
from matplotlib import cm
# load dataframe
dataset = pd.read_pickle("MAAD_Face_1.0.pkl") 
# convert dataframe to numpy array if necessary
# dataset.to_numpy()

# print(dataset.shape)

# img = dataset[0]
# img = np.reshape(img, (7,7))
# img = Image.fromarray(np.uint8((cm.gist_earth(img)*255)))
# img.show()

print(dataset)