import os
from importlib import reload
import numpy as np
import matplotlib.pyplot as plt
import imageio
import time
from io import BytesIO
from PIL import Image


def read_image(image_name):
    img = imageio.imread(image_name)
    img = img / 255
    return img


def initialize_means(img, clusters):
    points = np.reshape(img, (img.shape[0] * img.shape[1],
                              img.shape[2]))
    m, n = points.shape
    means = np.zeros((clusters, n))
    for i in range(clusters):
        rand1 = int(np.random.random(1) * 10)
        rand2 = int(np.random.random(1) * 8)
        means[i, 0] = points[rand1, 0]
        means[i, 1] = points[rand2, 1]
        means[i, 2] = points[rand2, 2]
    return points, means


def distance(x1, y1, x2, y2):
    dist = np.square(x1 - x2) + np.square(y1 - y2)
    dist = np.sqrt(dist)

    return dist


def k_means(points, means, clusters):
    iterations = 10  # the number of iterations
    m, n = points.shape
    index = np.zeros(m)

    while (iterations > 0):

        for j in range(len(points)):
            minv = 1000
            temp = None
            for k in range(clusters):
                x1 = points[j, 0]
                y1 = points[j, 1]
                x2 = means[k, 0]
                y2 = means[k, 1]
                if (distance(x1, y1, x2, y2) < minv):
                    minv = distance(x1, y1, x2, y2)
                    temp = k
                    index[j] = k
        for k in range(clusters):
            sumx = 0
            sumy = 0
            count = 0
            for j in range(len(points)):
                if (index[j] == k):
                    sumx += points[j, 0]
                    sumy += points[j, 1]
                    count += 1
            if (count == 0):
                count = 1
            means[k, 0] = float(sumx / count)
            means[k, 1] = float(sumy / count)
        iterations -= 1
    return means, index


def plotImage(img_array, size):
    reload(plt)
    plt.imshow(np.array(img_array / 255).reshape(*size))
    plt.axis('off')
    return plt


def imageByteSize(img):
    img_file = BytesIO()
    image = Image.fromarray(np.uint8(img))
    image.save(img_file, 'png')
    return img_file.tell() / 1024


def compress_image(means, index, img):
    centroid = np.array(means)
    recovered = centroid[index.astype(int), :]

    recovered = np.reshape(recovered, (img.shape[0], img.shape[1],
                                       img.shape[2]))
    plt.imshow(recovered)
    #plt.show()
    #print("image Size: {:.3f} KB".format(imageByteSize(recovered)))
    # imageio.imwrite('sscom.png',(color.convert_colorspace(recovered, 'HSV', 'RGB')*255).astype(np.uint8))


if __name__ == '__main__':

    for i_para in range(4,12,2):
        # path of images folder
        path= '/Users/jc/Desktop/Github/image_compression/image'

        files = [os.path.join(path, file) for file in os.listdir(path)]
        start_time = time.time()
        num_image = 0

        # K
        clusters = i_para

        print(clusters)

        for image_name in files:
            img = read_image(image_name)
            num_image += 1
            points, means = initialize_means(img, clusters)
            means, index = k_means(points, means, clusters)
            compress_image(means, index, img)

        print(clusters)
        print(time.time() - start_time)
