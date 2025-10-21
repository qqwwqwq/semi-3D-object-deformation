import math

import cv2
import numpy as np
import matplotlib.pyplot as plt
from random import randint
from math import *
np.set_printoptions(suppress=True)

# In[18]:

# imagepath = 'scar.jpg'
SEGS = 4
NEIGHBORS = [(-1,0) , (1,0) , (0,-1) , (0,1),(1,1),(1,-1),(-1,1),(-1,-1)]
#(1,1),(1,-1),(-1,1),(-1,-1)
BETA =1
TEMPERATURE = 4000
ITERATIONS =0
COOLRATE = 0.95
zeta=1
# In[3]:

def isSafe(a,x,y):
    return x>=0 and x<a[0] and y>=0 and y<a[1]
def delta(i,l,a,b,k1s,k2s):
    if(i==l):
        return BETA*min(1,math.sqrt((k1s[a[0]][a[1]]-k1s[b[0]][b[1]])**2+(k2s[a[0]][a[1]]-k2s[b[0]][b[1]])**2))
    return 1*BETA
# In[4]:

def reconstruct(labs):
    labels = labs
    for i in range(len(labels)):
        for j in range(len(labels[0])):
            labels[i][j] = (labels[i][j]*255)/(SEGS-1)
    return labels


# In[5]:
def twotoone(img,two):
    x=two[0]
    y=two[1]
    if y==0:
        one=x
    else:
        one=x*img.shape[1]+y
    return one

def calculateEnergy(img,labels,k1s,k2s):
    energy = 0.0
    nos = [0.0] * SEGS
    tag=np.zeros((img.shape[0]*img.shape[1],img.shape[0]*img.shape[1]),dtype="uint8")
    for i in range(len(img)):
        for j in range(len(img[0])):

            l = labels[i][j]
            vr1=lb1(k1s[i][j],k2s[i][j])
            vr2 = lb2(k1s[i][j],k2s[i][j])
            vr3 = lb3(k1s[i][j], k2s[i][j])
            vr4 = lb4(k1s[i][j], k2s[i][j])
            p = np.array([vr1, vr2, vr3, vr4])
            t=np.random.choice([0,1,2,3],p=p.ravel())
            if max(vr1,vr2,vr3,vr4)==vr1:
                nos[0]+=1
                # labels[i][j]=0
                energy += (1-vr1)
                for (p,q) in NEIGHBORS:
                    if isSafe(img.shape,i+p,j+q):
                        one1=twotoone(img,[i,j])
                        one2=twotoone(img,[i+p,j+q])
                        if tag[one1, one2]==0  and tag[one2,one1]==0:
                            tag[one1, one2]=1
                            energy += delta(l,labels[i+p][j+q],[i,j],[i+p,j+q],k1s,k2s)
            elif  max(vr1,vr2,vr3,vr4)==vr2:
                nos[1] += 1
                # labels[i][j] = 1
                energy += (1 - vr2)
                for (p, q) in NEIGHBORS:
                    if isSafe(img.shape, i + p, j + q):
                        one1 = twotoone(img, [i, j])
                        one2 = twotoone(img, [i + p, j + q])
                        if tag[one1, one2] == 0 and tag[one2, one1] == 0:
                            tag[one1, one2] = 1
                            energy += delta(l, labels[i + p][j + q], [i, j], [i + p, j + q], k1s, k2s)
            elif max(vr1,vr2,vr3,vr4)==vr3:
                nos[2] += 1
                # labels[i][j] = 2
                energy += (1 -vr3)
                for (p, q) in NEIGHBORS:
                    if isSafe(img.shape, i + p, j + q):
                        one1 = twotoone(img, [i, j])
                        one2 = twotoone(img, [i + p, j + q])
                        if tag[one1, one2] == 0 and tag[one2, one1] == 0:
                            tag[one1, one2] = 1
                            energy += delta(l, labels[i + p][j + q], [i, j], [i + p, j + q], k1s, k2s)
            elif max(vr1,vr2,vr3,vr4)==vr4:
                nos[3] += 1
                labels[i][j] = 3
                energy += (1 - vr4)
                for (p, q) in NEIGHBORS:
                    if isSafe(img.shape, i + p, j + q):
                        energy += delta(l, labels[i + p][j + q], [i, j], [i + p, j + q], k1s, k2s)
    return energy,nos


# In[6]:

def variance(sums1,squares1,nos1):
    return squares1/nos1-(sums1/nos1)**2


# In[7]:
# def lb1(k1,k2):
#     rs=math.exp(-(k1**2)/(2*(zeta**2)))*math.exp(-(k2**2)/(2*(zeta**2)))
#     print(rs,"0")
#     return rs
# def lb2(k1,k2):
#     rs=(1-math.exp(-((-k1)**2)/(2*(zeta**2))))*math.exp(-(k2**2)/(2*(zeta**2)))
#     print(rs, "1")
#     return rs
# def lb3(k1,k2):
#     rs=(1-math.exp(-(k1**2)/(2*(zeta**2))))*(1-math.exp(-((-k2)**2)/(2*(zeta**2))))
#     print(rs, "2")
#     return rs
# def lb4(k1,k2):
#     rs=math.exp(-((-k1)**2)/(2*(zeta**2)))*(1-math.exp(-((-k2)**2)/(2*(zeta**2))))
#     print(rs, "3")
#     return rs
# def lb1(k1,k2):
#     rs=math.exp(-(k1**2)/(2*(zeta**2)))*math.exp(-(k2**2)/(2*(zeta**2)))
#     print(rs,"0",k1,k2)
#     return rs
# def lb2(k1,k2):
#     rs=math.exp(-((1-k1)**2)/(2*(zeta**2)))*math.exp(-(k2**2)/(2*(zeta**2)))
#     print(rs, "1",math.exp(-((1-k1)**2)/(2*(zeta**2))),math.exp(-(k2**2)/(2*(zeta**2))))
#     return rs
# def lb3(k1,k2):
#     rs=math.exp(-((1-k1)**2)/(2*(zeta**2)))*math.exp(-((1-k2)**2)/(2*(zeta**2)))
#     print(rs, "2",k1,k2)
#     return rs

def lb1(k1,k2):
    if (k1==0 and k2==0) :
        return 1
    else:
        return 0
def lb2(k1,k2):
    if k1*k2==0 and (k1+k2)!=0 :
        return 1
    else:
        return 0
def lb3(k1,k2):
    # if (k1*k2>0 or k1*k2<0 ):
    if k1*k2>0:
        return 1
    else:
        return 0
def lb4(k1,k2):
    if k1*k2<0:
        return 1
    else:
        return 0
def initialize(img,k1s,k2s):
    labels = np.zeros(shape=img.shape,dtype=np.uint8)
    print(labels.shape,1111)

    # squares = [0.0]*SEGS
    for i in range(len(img)):
        for j in range(len(img[0])):
            # l = randint(0,SEGS-1)
            # labels[i][j] = l
            k1=k1s[i][j]
            k2=k2s[i][j]
            vr1 = lb1(k1, k2)
            vr2 = lb2(k1, k2)
            vr3 = lb3(k1, k2)
            vr4=lb4(k1,k2)
            p=np.array([vr1,vr2,vr3,vr4])
            # labels[i][j]=np.random.choice([0,1,2,3],p=p.ravel())
            if   max(vr1,vr2,vr3,vr4)==vr1:
                labels[i][j] = 0
            if  max(vr1,vr2,vr3,vr4)==vr2:
                labels[i][j] = 1
            if  max(vr1,vr2,vr3,vr4)==vr3:
                labels[i][j] = 1
            if  max(vr1,vr2,vr3,vr4)==vr4:
                labels[i][j] = 1
            # if k1 * k2 < 0:
            #     labels[i][j] = 3
    return labels


# In[8]:
def MRF(img,k1s,k2s):
    # original = cv2.imread(image)
    # origflt = original.astype(float)
    # # In[9]:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # In[10]:
    labels = initialize(img,k1s,k2s)
    return labels
    # variances = [variance(sums[i], squares[i], nos[i]) for i in range(SEGS)]
    # In[11]:
    energy,nos = calculateEnergy(img, labels,k1s,k2s)
    # In[12]:
    # variances,sums,squares,nos
    # In[13]:
    temp = TEMPERATURE
    it = ITERATIONS
    while it > 0:
        (a, b) = img.shape
        change = False
        x = randint(0, a - 1)
        y = randint(0, b - 1)
        val = float(img[x][y])
        l = labels[x][y]
        newl = l
        while newl == l:
            newl = randint(0, SEGS - 1)

        val = float(val)
        newenergy = energy

        if l==0:
            newenergy -= (1-lb1(k1s[x][y],k2s[x][y]))
            if newl==1:
                newenergy += (1-lb2(k1s[x][y],k2s[x][y]))
            elif newl==2:
                newenergy += (1 - lb3(k1s[x][y], k2s[x][y]))
            for (p, q) in NEIGHBORS:
                if isSafe((a, b), x + p, y + q):
                    newenergy -= delta(l, labels[x + p][y + q], [x, y], [x+ p, y + q], k1s, k2s)
                    newenergy += delta(newl, labels[x + p][y + q], [x, y], [x+ p, y + q], k1s, k2s)
        elif l==1:
            newenergy -= (1 - lb2(k1s[x, y], k2s[x, y]))
            if newl == 0:
                newenergy += (1 - lb1(k1s[x, y], k2s[x, y]))
            elif newl == 2:
                newenergy += (1 - lb3(k1s[x, y], k2s[x, y]))

            for (p, q) in NEIGHBORS:
                if isSafe((a, b), x + p, y + q):
                    newenergy -= delta(l, labels[x + p][y + q], [x, y], [x+ p, y + q], k1s, k2s)
                    newenergy += delta(newl, labels[x + p][y + q], [x, y], [x+ p, y + q], k1s, k2s)
        elif l==2:
            newenergy -= (1 - lb3(k1s[x, y], k2s[x, y]))
            if newl == 1:
                newenergy += (1 - lb2(k1s[x, y], k2s[x, y]))
            elif newl == 0:
                newenergy += (1 - lb1(k1s[x, y], k2s[x, y]))

            for (p, q) in NEIGHBORS:
                if isSafe((a, b), x + p, y + q):
                    newenergy -= delta(l, labels[x + p][y + q], [x, y], [x+ p, y + q], k1s, k2s)
                    newenergy += delta(newl, labels[x + p][y + q], [x, y], [x+ p, y + q], k1s, k2s)
        elif l==3:
            newenergy -= (1 - lb4(k1s[x, y], k2s[x, y]))
            if newl == 1:
                newenergy += (1 - lb2(k1s[x, y], k2s[x, y]))
            elif newl == 2:
                newenergy += (1 - lb3(k1s[x, y], k2s[x, y]))
            elif newl == 0:
                newenergy += (1 - lb1(k1s[x, y], k2s[x, y]))
            for (p, q) in NEIGHBORS:
                if isSafe((a, b), x + p, y + q):
                    newenergy -= delta(l, labels[x + p][y + q], [x, y], [x+ p, y + q], k1s, k2s)
                    newenergy += delta(newl, labels[x + p][y + q], [x, y], [x+ p, y + q], k1s, k2s)
        if newenergy < energy:
            change = True
        else:
            prob = 0.1
            if temp != 0:
                prob = np.exp((energy - newenergy) / temp)
            if prob >= (randint(0, 1000) + 0.0) / 1000:
                change = True

        if change:
            print(temp,energy, "energy", it)
            labels[x][y] = newl
            energy = newenergy
            nos[l] -= 1
            nos[newl] += 1
        temp *= COOLRATE
        it -= 1
    # In[14]:
    print(nos)
    return labels
    # plt.imshow(reconstruct(labels), interpolation='nearest', cmap='gray')
    # plt.show()
    # plt.imshow(img, cmap='gray')
    # cv2.imwrite("segmented.jpg", labels)
#
# MRF(imagepath)