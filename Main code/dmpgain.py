import math
import random

import numpy as np
import matplotlib.pyplot as plt
from pDMP_functions import pDMP
from scipy import signal
import pandas as pd
pd.set_option('display.float_format',lambda x : '%.3f' % x)
class gaindmp:
    def __init__(self,length,unit,period,demodata):
        self.length=length#length of demo
        self.unit=unit #length of sample
        self.period=period #period
        self.phi=0
        self.data=[]
        self.demodata=demodata
    def learn(self):
        myDMP = pDMP(1, 25, 8, 2, 0.995, self.unit)
        y_old = 0
        dy_old = 0
        samples = self.length / self.unit
        # helf=self.period/self.unit/2
        yset = np.zeros(int(samples) + 1)
        for i in range(int(samples) + 1):
            if i != 0:
                self.phi += 2 * np.pi * self.unit / self.period
            y = np.array([self.demodata[i]])
            print(self.phi / np.pi)
            dy = (y - y_old) / self.unit
            ddy = (dy - dy_old) / self.unit
            myDMP.set_phase(np.array([self.phi]))
            myDMP.set_period(np.array([self.period]))
            myDMP.learn(y, dy, ddy)
            myDMP.integration()
            # old values
            y_old = y
            dy_old = dy
            # store data for plotting
            x, dx, ph, ta = myDMP.get_state()
            yset[i] += x
            time = self.unit * i
            # self.data.append([time, self.phi, x[0], y[0]])
    def update(self,target):
        myDMP = pDMP(1, 25, 8, 2,0.98,self.unit)
        y_old = 0
        dy_old = 0
        samples = self.length / self.unit
        yset=np.zeros(int(samples)+1)
        for i in range(int(samples)+1):
            if i!=0:
                self.phi += 2 * np.pi * self.unit / self.period
            y = np.array([self.demodata[i]])
            # print(self.phi / np.pi)
            dy = (y - y_old) / self.unit
            ddy = (dy - dy_old) / self.unit
            gnew=(target-max(yset))+(target-abs(min(yset)))
            U = gnew * y
            myDMP.set_phase(np.array([self.phi]))
            myDMP.set_period(np.array([self.period]))
            # if i < int(samples/16) :
            #     myDMP.learn(y,dy,ddy)
            # else:
            myDMP.update(U)
            # else:
            #     myDMP.repeat()
            myDMP.integration()
            # old values
            y_old = y
            dy_old = dy
            # store data for plotting
            x, dx, ph, ta = myDMP.get_state()
            yset[i]+=x
            time = self.unit * i
            self.data.append([time, self.phi, x[0], y[0],gnew])
        self.data = np.array(self.data)
        num=signal.argrelextrema(self.data[:,2], np.greater)[0][-2]
        return self.data[:,0][int(num-self.period/self.unit/2):int(num+self.period/self.unit/2)],self.data[:,2][int(num-self.period/self.unit/2):int(num+self.period/self.unit/2)]
    def show(self):
        self.data=np.array(self.data)
        print(self.data.shape)
        # plt.subplot(2, 1, 1)
        # print(np.format_float_positional(self.data[:,4][-1], trim='-'),self.data[:,2][-1],11111111111111111111111111111)
        plt.plot(self.data[:,0],self.data[:,2],"r",label="adapted value")
        yvals=self.data[:,2]
        # plt.plot(self.data[:,0][signal.argrelextrema(yvals, np.greater)[0]], yvals[signal.argrelextrema(yvals, np.greater)], 'o',
        #          markersize=10)  # 极大值点
        # plt.plot(self.data[:,0][signal.argrelextrema(yvals, np.less)[0]], yvals[signal.argrelextrema(yvals, np.less)], '+',
        #          markersize=10)  # 极小值点
        # plt.subplot(2, 1, 2)
        plt.plot(self.data[:,0], self.data[:,4], "g",label="task related signal")
        # plt.plot(self.data[:, 0], [15]*len(self.data[:,0]), "y", label="target_width")
        plt.xlabel('time [s]', fontsize='12')
        plt.ylabel('Value of 1-DOF', fontsize='13')
        plt.plot(self.data[:, 0], self.demodata, "b",label="demo")
        # plt.plot(self.data[:, 0], self.gain, "y")
        plt.xlabel("length of demo",fontsize="17")
        plt.ylabel("value of 1 DOF",fontsize="17")
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        # plt.xlim(0,141.7)
        plt.legend(prop={'size': 15})
        plt.show()


