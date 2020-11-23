import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time
from numpy.linalg import norm
from random import normalvariate
from math import sqrt


def compress(image_name):
    start_time = time.time()
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


def randomUnitVector(n):
    unnormalized = [normalvariate(0, 1) for _ in range(n)]
    theNorm = sqrt(sum(x * x for x in unnormalized))
    return [x / theNorm for x in unnormalized]


def svd_1d(A, epsilon=1e-10):
    ''' The one-dimensional SVD '''

    n, m = A.shape
    x = randomUnitVector(min(n,m))
    lastV = None
    currentV = x

    if n > m:
        B = np.dot(A.T, A)
    else:
        B = np.dot(A, A.T)

    iterations = 0
    while True:
        iterations += 1
        lastV = currentV
        currentV = np.dot(B, lastV)
        currentV = currentV / norm(currentV)

        if abs(np.dot(currentV, lastV)) > 1 - epsilon:
            print("converged in {} iterations!".format(iterations))
            return currentV


def svd(A, k=None, epsilon=1e-10):
    '''
        Compute the singular value decomposition of a matrix A
        using the power method. A is the input matrix, and k
        is the number of singular values you wish to compute.
        If k is None, this computes the full-rank decomposition.
    '''
    A = np.array(A, dtype=float)
    n, m = A.shape
    svdSoFar = []
    if k is None:
        k = min(n, m)

    for i in range(k):
        matrixFor1D = A.copy()

        for singularValue, u, v in svdSoFar[:i]:
            matrixFor1D -= singularValue * np.outer(u, v)

        if n > m:
            v = svd_1d(matrixFor1D, epsilon=epsilon)  # next singular vector
            u_unnormalized = np.dot(A, v)
            sigma = norm(u_unnormalized)  # next singular value
            u = u_unnormalized / sigma
        else:
            u = svd_1d(matrixFor1D, epsilon=epsilon)  # next singular vector
            v_unnormalized = np.dot(A.T, u)
            sigma = norm(v_unnormalized)  # next singular value
            v = v_unnormalized / sigma

        svdSoFar.append((sigma, u, v))

    singularValues, us, vs = [np.array(x) for x in zip(*svdSoFar)]
    return singularValues, us.T, vs


def main():
    compress('sample.jpg')
    

if __name__ == '__main__':
    main()