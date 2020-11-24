import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time
from numpy.linalg import norm
from random import normalvariate
from math import sqrt
import glob


# decompose image to green, red and blue layers
def decompose_rgb(image_name):
    start_time = time.time()
    img = Image.open(image_name)
    a=np.array(img)

    i = 1
    u,sigma,v=np.linalg.svd(a[:,:,0])
    R=rebuild_img(u,sigma,v,i)
    plt.subplot(331)
    plt.title('R')
    plt.imshow(R, cmap ='Reds')

    u,sigma,v=np.linalg.svd(a[:,:,1])
    G=rebuild_img(u,sigma,v,i)
    plt.subplot(332)
    plt.title('G')
    plt.imshow(G, cmap ='Greens')

    u,sigma,v=np.linalg.svd(a[:,:,2])
    B=rebuild_img(u,sigma,v,i)
    plt.subplot(333)
    plt.title('B')
    plt.imshow(B,cmap ='Blues')
    I=np.stack((R,G,B),2)
    print(I)
    plt.subplot(334)
    plt.title("original")
    plt.imshow(I)

    plt.show()

# impact of rank on compress ratio
def rank_compress(image_name):
    img = Image.open(image_name)
    a=np.array(img)

    for i in np.arange(0.1,1,0.1):
  
        u,sigma,v=np.linalg.svd(a[:,:,0])
        R=rebuild_img(u,sigma,v,i)

        u,sigma,v=np.linalg.svd(a[:,:,1])
        G=rebuild_img(u,sigma,v,i)

        u,sigma,v=np.linalg.svd(a[:,:,2])
        B=rebuild_img(u,sigma,v,i)

        I=np.stack((R,G,B),2)
        plt.subplot(330 + 10 * i)
        plt.title(i)
        plt.imshow(I)

    plt.show()

# impact of running time over rank
def rank_over_compress():
    allfiles = glob.glob('image/*.jpg')
    ranks = [0.3, 0.6, 0.9]
    x = [i+1 for i in range(20)]
    times = [[],[],[]]
    #only test on the first 20 images
    for image_name in allfiles[:20]:
        img = Image.open(image_name)
        a=np.array(img)
        for index, rank in enumerate(ranks):
            start_time = time.time()
            u,sigma,v=svd(a[:,:,0])
            R=rebuild_img(u,sigma,v,rank)

            u,sigma,v=svd(a[:,:,1])
            G=rebuild_img(u,sigma,v,rank)

            u,sigma,v=svd(a[:,:,2])
            B=rebuild_img(u,sigma,v,rank)

            I=np.stack((R,G,B),2)
            end_time = time.time() - start_time
            if times[index]:
                end_time += times[index][-1]
            times[index].append(end_time)
    
    l1 = plt.plot(x, times[0], 'r--', label = 'rank = 0.3')
    l1 = plt.plot(x, times[1], 'g--', label = 'rank = 0.6')
    l1 = plt.plot(x, times[2], 'b--', label = 'rank = 0.9')
    plt.legend(loc='upper left')
    plt.title('impact of rank on running time')
    plt.ylabel('running time/s')
    plt.xlabel('image number')
    plt.show()

# run method on large dataset to measure running time
def runTime(i):
    allfiles = glob.glob('image/*.jpg')
    print(len(allfiles))
    print(allfiles)
    start_time = time.time()
    count = 1
    for image_name in allfiles:
        img = Image.open(image_name)
        a=np.array(img)

        u,sigma,v=svd(a[:,:,0])
        R=rebuild_img(u,sigma,v,i)

        u,sigma,v=svd(a[:,:,1])
        G=rebuild_img(u,sigma,v,i)

        u,sigma,v=svd(a[:,:,2])
        B=rebuild_img(u,sigma,v,i)

        I=np.stack((R,G,B),2)
        print(count)
        count += 1

    print(time.time() - start_time)

# convert input graph data set to edge pairs
def readIn(fileName):
    max_node = 0
    arr = []
    f = open(fileName, 'r')
    points = f.read().split('\n')
    for pair in points:
        if not pair or pair[0] == '#':
            continue
        point = pair.split('\t')
        arr.append((int(point[0]), int(point[1])))
        max_node = max(max_node, int(point[0]), int(point[1]))
    return arr, max_node

# generate a random unit vector
def random_unit_vector(n):
    un_normalized = [normalvariate(0, 1) for _ in range(n)]
    Norm = sqrt(sum(x * x for x in un_normalized))
    output = [x / Norm for x in un_normalized]
    return output

# svd for 1-D matrix
def svd_1D(A, epsilon=1e-10):
    n, m = A.shape
    x = random_unit_vector(min(n,m))
    last_V = None
    current_V = x

    if n > m:
        B = np.dot(A.T, A)
    else:
        B = np.dot(A, A.T)

    while True:
        last_V = current_V
        current_V = np.dot(B, last_V)
        current_V = current_V / norm(current_V)

        if abs(np.dot(current_V, last_V)) > 1 - epsilon:
            return current_V

# svd decompose on A
def svd(A, epsilon = 1e-10):
    A = np.array(A, dtype=float)
    n, m = A.shape
    current_svd = []
    k = min(n, m)

    for i in range(k):
        matrixFor1D = A.copy()

        for singularValue, u, v in current_svd[:i]:
            matrixFor1D -= singularValue * np.outer(u, v)

        if n > m:
            v = svd_1D(matrixFor1D, epsilon=epsilon)  
            u_unnormalized = np.dot(A, v)
            sigma = norm(u_unnormalized)  
            u = u_unnormalized / sigma
        else:
            u = svd_1D(matrixFor1D, epsilon=epsilon)  
            v_unnormalized = np.dot(A.T, u)
            sigma = norm(v_unnormalized)  
            v = v_unnormalized / sigma

        current_svd.append((sigma, u, v))

    singularValues, us, vs = [np.array(x) for x in zip(*current_svd)]
    return us.T, singularValues, vs


# build red, green and blue layers with edges
def build(arr, m):
    r = np.full((m,m), 255)
    g = np.full((m,m), 255)
    b = np.full((m,m), 255)
    for x, y in arr:
        r[x-1][y-1] = 0
        g[x-1][y-1] = 0
        b[x-1][y-1] = 0
    p = np.stack((r,g,b),2)
    plt.title('picture')
    plt.imshow(p)
    plt.show()

# plot the distribution of edges in the graph
def displayspots(arr):
    x1 = [x[0] for x in arr]
    y1 = [x[1] for x in arr]
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.scatter(x1, y1, s=1, c='b', marker="s", label='points')
    plt.legend(loc='upper left')
    plt.show()


# rebuild img with given percent of rank
def rebuild_img(u,sigma,v,p):
    m = len(u)
    n = len(v)
    a = np.zeros((m,n))

    count = sum(sigma)
    curSum = 0
    k = 0

    while curSum < count * p:
        uk = u[:,k].reshape(m,1)
        vk = v[k].reshape(1,n)
        a += sigma[k] * np.dot(uk,vk)
        curSum += sigma[k]
        k += 1

    a[a<0]=0
    a[a>255]=255

    return np.rint(a).astype(int)

def main():
    pass

    '''
    test for decompose picture
    '''
    # decompose_rgb('sample.jpg')


    '''
    test for the impact of rank
    '''
    # rank_compress('sample.jpg')

    '''
    test for numpy.stack
    '''
    # a = np.array([[1, 2], [3,4], [5,6]])
    # b = np.array([[-1, -2], [-3,-4], [-5,-6]])
    # print(np.stack((a,b), 0))
    # print(np.stack((a,b), 0).shape)
    # print('--------------------------------------')
    # print(np.stack((a,b), 1))
    # print(np.stack((a,b), 1).shape)
    # print('--------------------------------------')
    # print(np.stack((a,b), 2))
    # print(np.stack((a,b), 2).shape)
    # print('--------------------------------------')

    '''
    test for dataset CA-GrQc.txt
    '''
    # arr, m = readIn('CA-GrQc.txt')
    # print(arr)
    # print(m)
    # build(arr,m)
    # displayspots(arr)

    '''
    test for running time over rank on first 50 images of the dataset
    '''
    rank_over_compress()


    '''
    test for running time of SVD, which keep 70% rank
    '''
    # runTime(0.7)



if __name__ == '__main__':
    main()