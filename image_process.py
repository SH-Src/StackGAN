import PIL
from PIL import Image
import pickle
import os
import numpy as np
with open('filenames.pickle', 'rb') as f:
    filenames = pickle.load(f)

data = []
for filename in filenames:
    imdir = os.listdir('D:/image_extraction/'+filename)
    img = Image.open('D:/image_extraction/'+filename+'/'+imdir[0]).convert('RGB')
    width, height = img.size
    img = img.crop([0, 0, min(width, height), min(width, height)])
    img = img.resize((64,64))
    data.append(np.array(img))

X = np.array(data)
print(X.shape)
with open('image_data.pickle', 'wb') as w:
    pickle.dump(X, w)






