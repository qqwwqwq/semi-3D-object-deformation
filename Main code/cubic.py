import math

import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from numpy import matrix, average
import scipy.linalg
from mpl_toolkits.mplot3d import Axes3D
# Parameters
pointsInterpolation = False
curveInterpolation = True
'''
    numberOfInterpolation determines the precision of interpolation.
    bigger numberOfInterpolation, more smooth curve
'''
numberOfInterpolation = 100

j = 0
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


def cubicSplineInterpolate(x_axis, y_axis, z_axis):
    '''
        prepare right-side vector
    '''
    dx = []
    dy = []
    dz = []
    matrix = []
    n = 2
    while n < len(x_axis):
        dx.append(3 * (x_axis[n] - 2 * x_axis[n - 1] + x_axis[n - 2]))
        dy.append(3 * (y_axis[n] - 2 * y_axis[n - 1] + y_axis[n - 2]))
        dz.append(3 * (z_axis[n] - 2 * z_axis[n - 1] + z_axis[n - 2]))
        n = n + 1
    '''
        produce square matrix looks like :
        [[2.0, 0.5, 0.0, 0.0], [0.5, 2.0, 0.5, 0.0], [0.0, 0.5, 2.0, 0.5], [0.0, 0.0, 2.0, 0.5]]
        the classes of the matrix depends on the length of x_axis(number of nodes)
    '''
    matrix.append([float(2), float(0.5)])
    for m in range(len(x_axis) - 4):
        matrix[0].append(float(0))
    n = 2
    while n < len(x_axis) - 2:
        matrix.append([])
        for m in range(n - 2):
            matrix[n - 1].append(float(0))

        matrix[n - 1].append(float(0.5))
        matrix[n - 1].append(float(2))
        matrix[n - 1].append(float(0.5))

        for m in range(len(x_axis) - n - 3):
            matrix[n - 1].append(float(0))
        n = n + 1

    matrix.append([])
    for m in range(n - 2):
        matrix[n - 1].append(float(0))
    matrix[n - 1].append(float(0.5))
    matrix[n - 1].append(float(2))
    '''
        LU Factorization may not be optimal method to solve this regular matrix. 
        If you guys have better idea to solve the Equation, please contact me.
        As the LU Factorization algorithm cost 2*n^3/3 + O(n^2) (e.g. Doolittle algorithm, Crout algorithm, etc).
        (How about Rx = Q'y using matrix = QR (Schmidt orthogonalization)?)
        If your application field requires interpolating into constant number nodes, 
        It is highly recommended to cache the P,L,U and reuse them to get O(n^2) complexity.
    '''
    P, L, U = doLUFactorization(matrix)
    u = solveEquations(P, L, U, dx)
    v = solveEquations(P, L, U, dy)
    w = solveEquations(P, L, U, dz)

    '''
        define gradient of start/end point
    '''
    m = 0
    U = [0]
    V = [0]
    W = [0]
    while m < len(u):
        U.append(u[m])
        V.append(v[m])
        W.append(w[m])
        m = m + 1
    U.append(0)
    V.append(0)
    W.append(0)

    return plotCubicSpline(U, V, W, x_axis, y_axis, z_axis)


'''
    calculate each parameters of location.
'''


def func(x1, x2, t, v1, v2, t1, t2):
    ft = ((t2 - t) ** 3 * v1 + (t - t1) ** 3 * v2) / 6 + (t - t1) * (x2 - v2 / 6) + (t2 - t) * (x1 - v1 / 6)
    return ft


'''
    note: 
    too many interpolate points make your computer slack.
    To interpolate large amount of input parameters,
    please switch to ax.plot().
'''
def lerp(b,a,c):
    v1=(c*a[0])+((1-c)*b[0])
    v2=(c*a[1])+((1-c)*b[1])
    v3=(c*a[2])+((1-c)*b[2])
    return np.asarray([v1,v2,v3])
def dis(a,b):
    k=math.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2+(a[2]-b[2])**2)
    return k
def plotCubicSpline(U, V, W, x_axis, y_axis, z_axis):
    def curvelen(pts):
        cp_curve = pts.copy()
        cp_curve[0] = cp_curve[-1]
        cp_curve = np.roll(cp_curve, -1, axis=0)
        lenth = np.sqrt((pts[:, 0] - cp_curve[:, 0]) ** 2 + (pts[:, 1] - cp_curve[:, 1]) ** 2 + (pts[:, 2] - cp_curve[:, 2]) ** 2)
        return lenth
    m = 1
    xLinespace = []
    yLinespace = []
    zLinespace = []
    while m < len(x_axis):
        for t in np.arange(m - 1, m, 1 / float(numberOfInterpolation)):
            xLinespace.append(func(x_axis[m - 1], x_axis[m], t, U[m - 1], U[m], m - 1, m))
            yLinespace.append(func(y_axis[m - 1], y_axis[m], t, V[m - 1], V[m], m - 1, m))
            zLinespace.append(func(z_axis[m - 1], z_axis[m], t, W[m - 1], W[m], m - 1, m))
        m = m + 1
    if pointsInterpolation:
        ax.scatter(xLinespace, yLinespace, zLinespace, color="red", s=0.01)
    if curveInterpolation:
        ax.plot(xLinespace, yLinespace, zLinespace, color="red")
    '''
    matched group, annotate it if unnecessary
    '''
    ax.plot(x_axis, y_axis, z_axis, color="blue")
    xLinespace=np.asarray(xLinespace)
    yLinespace = np.asarray(yLinespace)
    zLinespace = np.asarray(zLinespace)
    result=np.asarray([xLinespace,yLinespace,zLinespace])
    result=result.T
    # delete small one
    res=[]
    res.append(result[0])
    i=0
    j=i+1
    Tvalue=0.3
    while i< len(result)-1 :
        while dis(result[i],result[j])<Tvalue and j<len(result)-1:
            j=j+1
        res.append(result[j])
        i=j
        j=i+1
    res=np.asarray(res)
    # extend the big one
    reset=[]
    reset.append(res[0])
    j = 1
    while j < len(res):
        lenth = dis(reset[-1], res[j])
        inter = int(lenth / Tvalue)
        for step in range(1, inter + 1):
            reset.append(lerp(reset[-1], res[j], step * Tvalue / lenth))
        j = j + 1
    reset=np.asarray(reset)
    xaxis=reset[:,0]
    yaxis=reset[:,1]
    zaxis=reset[:,2]
    ax.plot(xaxis, yaxis, zaxis, color="blue")
    # plt.show()
    return reset


def corss(lp, rp, stp, edp):
    v1 = lp - stp
    v2 = rp - stp
    v3 = edp - stp
    a1 = np.cross(v1[:2], v3[:2])
    a2 = np.cross(v2[:2], v3[:2])
    if a1 * a2 > 0:
        return False
    return True

# print(corss(np.asarray([1,1,0]),np.asarray([0,0.5,0]),np.asarray([0,0,0]),np.asarray([1,0,0])))
#
# ax.plot3D(np.asarray([1,0]),np.asarray([1,0.5]),np.asarray([0,0]), 'red')
# ax.plot3D(np.asarray([0,1]),np.asarray([0,0]),np.asarray([0,0]), 'red')
# # ax.plot3D(np.asarray([0,1,0]),np.asarray([0,2,0]), 'blue')
# plt.show()
'''
    P stands for the permutation Matrix
    L stands for the lower-triangle Matrix
    U stands for the upper-triangle Matrix
    matrix·x = y
    P·matrix = L·U
    P·matrix·x = L·U·x = P·y
    L·U·x = y1
    U·x = y2
    x = y3
'''


def solveEquations(P, L, U, y):
    y1 = np.dot(P, y)
    y2 = y1
    m = 0
    for m in range(0, len(y)):
        for n in range(0, m):
            y2[m] = y2[m] - y2[n] * L[m][n]
        y2[m] = y2[m] / L[m][m]
    y3 = y2
    for m in range(len(y) - 1, -1, -1):
        for n in range(len(y) - 1, m, -1):
            y3[m] = y3[m] - y3[n] * U[m][n]
        y3[m] = y3[m] / U[m][m]
    return y3


'''
    this is the Scipy tool with high complexity.
'''


def doLUFactorization(matrix):
    P, L, U = scipy.linalg.lu(matrix)
    return P, L, U


'''
    input parameters
    each vector contain at least 3 elements
'''

# x_axis = [1, 2, 3, 4]
# y_axis = [2, 3, 4, 5]
# z_axis = [3, 4, 7, 5]
#
# cubicSplineInterpolate(x_axis, y_axis, z_axis)
#
# plt.show()