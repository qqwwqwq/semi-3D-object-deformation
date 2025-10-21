import numpy as np
import matplotlib.pyplot as plt
class dmtracject:
    def __init__(self,length,unit,period):
        self.length=length#length of demo
        self.unit=unit #length of sample
        self.period=period #period
        self.phi=0
        self.data=[]
    def generate(self):
        samples=self.length/self.unit
        for i in range(int(samples)+1):
            if i!=0:
                self.phi+=2*np.pi*self.unit/self.period
            # print(self.phi/np.pi)
            y=np.array([np.cos(self.phi)*10])
            leng=i*self.unit
            self.data.append([leng,y])
        self.data = np.array(self.data)
    def show(self):
        print(self.data[:,1][0],self.data[:,1][-1])
        fig=plt.figure(figsize=(16,9))
        plt.plot(self.data[:,0],self.data[:,1],"b")
        plt.scatter(np.array(self.data[:, 0]), np.array(self.data[:, 1]),c="r",s=25)
        # plt.xlabel("length",fontsize="30")
        # plt.ylabel("signal",fontsize="30")
        plt.xticks(fontsize="30")

        plt.yticks(fontsize="30")
        plt.savefig("high.png",dpi=900,format='png')
        plt.show()
