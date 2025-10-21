import random

import numpy as np
from main_sample import fps
import math
from matplotlib import pyplot as plt
import cv2
np.set_printoptions(suppress=True)


# image= cv2.imread('/Users/apple/Desktop/11.13/6/cool2.png')
# dp=np.load('/home/hexin/Desktop/11.13/6/dep1111.npy')
def randomcolor():

    colora=()

    for i in range(3):

        colora=colora+(random.randint(0,255),)

    return colora
def three2two(world):
    cx = 957.895
    cy = 546.481
    fx = 916.797
    fy = 916.603
    y=round(world[0]*fx/world[2]+cx)
    x=round(world[1]*fy/world[2]+cy)
    return [x,y]
def shortline(p,ps):
    dis=[]
    for i in ps:
        dis.append(math.sqrt((p[0]-i[0])**2+(p[1]-i[1])**2+(p[2]-i[2])**2))
    return dis.index(min(dis))
def segment(uu,dd,ll,rr,name,image):
    itritive=10
    shortest = np.zeros(shape=(dd-uu,rr-ll))
    slabel = np.zeros(shape=(dd-uu,rr-ll))
    n_superface = 300
    L_superface = [-1]*n_superface
    superface = [[] for row in range(n_superface)]
    def ini():
        threecenter,allpoint = fps(name, n_superface)
        twocenter = []
        for i in threecenter:
            twocenter.append(three2two(i))
        return twocenter,threecenter,allpoint
    tcenter,threecenter,allpoint=ini()
    for i in range(len(tcenter)):
        superface[i].append(threecenter[i])
        L_superface[i]=i
        slabel[tcenter[i][0]-uu][tcenter[i][1]-ll]=i
    for i in allpoint:
        t=three2two(i)
        stindex=shortline(i,threecenter)
        shortest[t[0]-uu][t[1]-ll]=stindex
        slabel[t[0]-uu][t[1]-ll]=stindex
        superface[stindex].append(i)
    while itritive>0:
        for i in range(len(superface)):
            k=np.asarray(superface[i])
            superface[i]=np.average(k,axis=0)
        newcenter=superface
        twonewcenter=[]
        for i in newcenter:
            twonewcenter.append(three2two(i))
        superface = [[] for row in range(n_superface)]
        shortest = np.zeros(shape=(dd - uu, rr - ll))
        slabel = np.zeros(shape=(dd - uu, rr - ll))
        for i in range(len(twonewcenter)):
            superface[i].append(newcenter[i])
            L_superface[i] = i
            slabel[twonewcenter[i][0] - uu][twonewcenter[i][1] - ll] = i
        for i in allpoint:
            t = three2two(i)
            stindex = shortline(i, newcenter)
            shortest[t[0] - uu][t[1] - ll] = stindex
            slabel[t[0] - uu][t[1] - ll] = stindex
            superface[stindex].append(i)
        itritive-=1
        print(itritive)
    for i in superface:
        q=randomcolor()
        for j in i:
            pt=three2two(j)
            image[pt[0]][pt[1]] = np.array([q[0], q[1], q[2]])











