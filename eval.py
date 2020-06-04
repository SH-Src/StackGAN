import pickle
from keras.models import load_model
import numpy as np
from matplotlib import pyplot as plt

def save_rgb_img(img, path):
    """
    Save an rgb image
    """
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(img)
    ax.axis("off")
    ax.set_title("Image")

    plt.savefig(path)
    plt.close()


if __name__ == '__main__':
    model = load_model('stage1_gen-100.h5')
    with open('caption_data.pickle', 'rb') as c:
        caption_data = pickle.load(c)

    z_noise2 = np.random.normal(0, 1, size=(1, 100))
    embedding = caption_data[7:8]
    image = model.predict_on_batch([embedding, z_noise2])
    save_rgb_img(image, 'image.png')