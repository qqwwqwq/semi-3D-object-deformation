import math
import random
import tkinter

import numpy as np
import cv2
from matplotlib import pyplot as plt
import cv2
import numpy
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D
from sympy import *
from sympy.abc import x, y, z
def LLs(x, y, z, size):
    a = 0
    A = np.ones((size, 3))
    for i in range(0, size):
        A[(i, 0)] = x[a]
        A[(i, 1)] = y[a]
        a = a + 1

    b = np.zeros((size, 1))
    a = 0
    for i in range(0, size):
        b[(i, 0)] = z[a]
        a = a + 1

    A_T = A.T
    A1 = np.dot(A_T, A)
    A2 = np.linalg.inv(A1)
    A3 = np.dot(A2, A_T)
    X = np.dot(A3, b)
    for i in X:
        i[0] = float(i[0])

    return X
def growprocess(seed, imgd,uu,dd,rr,ll):
    img=imgd.copy()
    up = seed[0][0]
    down = seed[0][0]
    left = seed[0][1]
    right = seed[0][1]
    print('start')
    setx = []
    sety = []
    setz = []

    def ring(tw=None):
        cx = 957.895
        cy = 546.481
        fx = 916.797
        fy = 916.603
        x = tw[0]
        y = tw[1]
        z = imgd[x][y]
        xw1 = (y - cx) * z / fx
        yw1 = (x - cy) * z / fy
        (setx.append(xw1), sety.append(yw1), setz.append(z))
    def check2(a,b):
        o1=seed[0][0]
        o2=seed[0][1]
        cx = 957.895
        cy = 546.481
        fx = 916.797
        fy = 916.603
        x1=(b - cx) * imgd[a][b] / fx
        y1=(a - cy) * imgd[a][b]/ fy
        x2=(o2 - cx) * imgd[o1][o2] / fx
        y2=(o1 - cy) * imgd[o1][o2] / fy
        out=[(x1+x2)/2,(y1+y2)/2,(float(imgd[a][b])+float(imgd[o1][o2] ))/2]
        yy=int(out[0]*fx/out[2]+cx)
        xx=int(out[1]*fy/out[2]+cy)
        zz=imgd[xx][yy]
        xw=(yy - cx) * zz / fx
        yw=(xx-cy)*zz/fy
        error=sqrt(pow(out[0]-xw,2)+pow(out[1]-yw,2)+pow(float(out[2])-float(zz),2))
        return error



    dir = []
    distan = 0
    j = 1
    ring(seed[0])
    preve = []
    while j < 8:

        if j ==1:
            seed0 = []
            for i in range(down - 1, up + 2):
                if imgd[(i, left - 1)] != 0:
                    seed0.append([i, left - 1])
                if imgd[(i, right + 1)] != 0:
                    seed0.append([ i,right + 1])
            for i in range(left, right + 1):
                if imgd[(up + 1, i)] != 0:
                    seed0.append([up + 1,i])
                if imgd[(down - 1, i)] != 0:
                    seed0.append([ down - 1,i])
            for i in seed0:
                ring(i)


            preve = seed0[:]
            seed.extend(seed0)
            if len(setz) < 4:
                for i in seed:
                    imgd[i[0]][i[1]] = 0
                return np.array([0,0,0]),seed,imgd
            if up + 1 < dd:
                up += 1
            if down > uu:
                down -= 1
            if left > ll:
                left -= 1
            if right < rr:
                right += 1
            A = LLs(setx, sety, setz, len(setz))
            dir = [  A[0][0],   A[1][0],  -1]
            distan = A[2][0]
            j += 1
            continue
        seed1 = []
        ober=2

        # if j < 3072*4096/400:
        #     thre = 4* pow(1 - pow(numpy.e, -j), 2)
        # else:
        #     thre = 0.09 * pow(distan, 2) * 9 * pow(1 - pow(np.e, -j), 2)
        for i in range(down - 1, up + 2):
            if  imgd[(i, left - 1)] != 0 :
                if  [i,  left] in preve or [ i - 1, left] in preve or [i + 1,  left] in preve :
                    if check2(i,left-1)<ober:
                        seed1.append([ i, left - 1])
        for i in range(down - 1, up + 2):
            if imgd[(i, right + 1)] != 0:
                if [i, right] in preve or [i - 1, right] in preve or [i + 1, right] in preve :
                    if check2(i, right+1)<ober:
                        seed1.append([i,right + 1])
        for i in range(left, right + 1):
            if imgd[(up + 1, i)] != 0:
                if [up, i] in preve or [up,i - 1] in preve or [up,i + 1] in preve :
                    if check2(up+1, i)<ober:
                        seed1.append([up + 1, i])
        for i in range(left, right + 1):
            if imgd[(down - 1, i)] != 0:
                 if [down, i] in preve or [down, i - 1] in preve or [down, i + 1] in preve :
                    if check2(down-1, i)<ober:
                        seed1.append([down - 1, i])
        preve = seed1[:]
        for i in seed1:
            ring(i)
        seed.extend(seed1)
        if up + 1 < dd:
            up += 1
        if down > uu:
            down -= 1
        if left > ll:
            left -= 1
        if right < rr:
            right += 1
        A = LLs(setx, sety, setz, len(setz))
        dir = np.array([A[0][0],A[1][0],-1])
        distan = A[2][0]
        j += 1
    #
    dir1 = dir / np.linalg.norm(dir)
    for i in seed:
        imgd[i[0]][i[1]]=0
    # plt.imshow(imgd)
    # plt.show()
    return dir1,seed,imgd