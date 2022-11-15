import cv2
import numpy as np
from matplotlib import pyplot as plt

def load_image_grayscale(path, size=(128,128)):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # dtype=np.uint8
    return cv2.resize(gray, size)
 
def create_images_array(paths, size=(128,128)):
    images = []
    for path in paths:
        images.append(load_image_grayscale(path,size))
    return images

def convert_to_1d_vector(mtx):
    return mtx.flatten()

def pixel_by_pixel_fe():
    ...

def lbp_fe(image, radio=4):
    LBP = np.zeros_like(image)
    for ih in range(0,image.shape[0] - radio):
        for iw in range(0,image.shape[1] - radio):
            img = image[ih:ih+radio,iw:iw+radio]
            center = img[1,1]
            slika = (img >= center)*1.0
            slika_vector = slika.flatten()
            slika_vector = np.delete(slika_vector,4)
            where_slika_vector = np.where(slika_vector)[0]
            if len(where_slika_vector) >= 1:
                num = np.sum(2**where_slika_vector)
            else:
                num = 0
            LBP[ih+1,iw+1] = num
    return(LBP)

def lbp_histograms(image):
    imgLBP = lbp_fe(image)
    vecimgLBP = imgLBP.flatten()
    fig = plt.figure(figsize=(20,8))
    ax = fig.add_subplot(1,3,1)
    ax.imshow(image,cmap="gray")
    ax.set_title("gray scale image")
    ax = fig.add_subplot(1,3,2)
    ax.imshow(imgLBP,cmap="gray")
    ax.set_title("LBP converted image")
    ax = fig.add_subplot(1,3,3)
    freq,lbp, _ = ax.hist(vecimgLBP,bins=2**8)
    ax.set_ylim(0,40000)
    lbp = lbp[:-1]
    largeTF = freq > 5000
    vector = []
    for x, fr in zip(lbp[largeTF],freq[largeTF]):
        ax.text(x,fr, "{:6.0f}".format(x),color="magenta")
        vector.append(x)
    ax.set_title("LBP histogram")
    return vector
