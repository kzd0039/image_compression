import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time

def _SVD(image_name):
    start_time = time.time()
    img = Image.open(image_name)
    a=np.array(img)
    u,sigma,v=np.linalg.svd(a[:,:,0])
    R = rebuild_img(u, sigma, v, 0.5)

    for i in np.arange(0.1,1,0.1):
        u,sigma,v=np.linalg.svd(a[:,:,0])
        R=rebuild_img(u,sigma,v,i)
    
        u,sigma,v=np.linalg.svd(a[:,:,1])
        G=rebuild_img(u,sigma,v,i)
    
        u,sigma,v=np.linalg.svd(a[:,:,2])
        B=rebuild_img(u,sigma,v,i)
    
        I=np.stack((R,G,B),2)
        plt.subplot(330+i*10)
        plt.title(i)
        plt.imshow(I)
    print(time.time() - start_time)
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
    _SVD('sample.jpg')
    

if __name__ == '__main__':
    main()