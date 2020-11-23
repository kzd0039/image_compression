import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time

def decompose_rgb(image_name):
    start_time = time.time()
    img = Image.open(image_name)
    a=np.array(img)
    u,sigma,v=np.linalg.svd(a[:,:,0])
    # R = rebuild_img(u, sigma, v, 0.5)

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

def displayspots(arr):
    x1 = [x[0] for x in arr]
    y1 = [x[1] for x in arr]
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.scatter(x1, y1, s=1, c='b', marker="s", label='points')
    plt.legend(loc='upper left')
    plt.show()

def rebuild_img(u,sigma,v,p):
    m=len(u)
    n=len(v)
    a=np.zeros((m,n))

    count=(int)(sum(sigma))
    curSum=0
    k=0

    while curSum<=count*p:
        uk=u[:,k].reshape(m,1)
        vk=v[k].reshape(1,n)
        a+=sigma[k]*np.dot(uk,vk)
        curSum+=sigma[k]
        k+=1

    a[a<0]=0
    a[a>255]=255

    return np.rint(a).astype(int)

def main():
    '''
    test for decompose picture
    '''
    decompose_rgb('sample.jpg')

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
    # # print(arr)
    # # print(m)
    # # build(arr,m)
    # displayspots(arr)


    pass

if __name__ == '__main__':
    main()