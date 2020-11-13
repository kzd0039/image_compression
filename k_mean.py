import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time
import random

def _k_mean(image_name):
    pass

def _k_ref(arr,k):
    new_center = random.sample(arr, k)
    center = []
    count = 0
    while not equals(center, new_center):
        output = [[], [], []]
        for point in arr:
            dis = float('inf')
            index = -1
            for i,c in enumerate(new_center):
                if distance(point,c) <= dis:
                    dis = distance(point,c)
                    index = i
            output[index].append(point)
        count += 1
        center = new_center
        new_center = [average(x) for x in output]
 
    return output, new_center, count

def equals(a,b):
    if len(a) != len(b):
        return False
    a.sort()
    b.sort()
    flag = True
    for i in range(len(a)):
        if abs(a[i][0] - b[i][0]) > 0.000001 or abs(a[i][1] - b[i][1]) > 0.000001:
            flag = False
    return flag

def distance(a,b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

def average(arr):
    x = sum(x[0] for x in arr) / len(arr)
    y = sum(x[1] for x in arr) / len(arr)
    return (x,y)   

def main():
    _k_mean('sample.jpg')

if __name__ == '__main__':
    main()