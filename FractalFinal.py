import numpy as np
import scipy.misc as msc
import scipy.ndimage as img
import scipy as sp
import math
from scipy.optimize import curve_fit
import matplotlib.pyplot as mp

def fit(x, b, D): # this is our 'straight line' y=f(x)
    return D*x + b

#Class to implement a deque
class Deque:
    def __init__(self):
        self.items = []
    def isEmpty(self):
        return self.items == []
    def addFront(self, item):
        self.items.append(item)
    def addRear(self, item):
        self.items.insert(0,item)
    def removeFront(self):
        return self.items.pop()
    def removeRear(self):
        return self.items.pop(0)
    def size(self):
        return len(self.items)


class FractalDimention:
        def __init__(self,f):
            self.OriginalImage = f
            self.BinaryImageArray = self.foreground2BinImg(f)
            self.CumSum = self.DPStoreSum()
            self.Queue = Deque() #Deque will be used for Breadth First Search of the box counting
            self.BoxLevel = []

        # Convert the image to binary format ( function is given in assignment )
        def foreground2BinImg(self,f):
            d = img.filters.gaussian_filter(f, sigma=0.50, mode='reflect') - img.filters.gaussian_filter(f, sigma=1.00,mode='reflect')
            d = np.abs(d)
            m = d.max()
            d[d < 0.1 * m] = 0
            d[d >= 0.1 * m] = 1
            return img.morphology.binary_closing(d)

        #This function stores the cumulative sum of the matix into another auxiliary matrix
        # This is an implementation of Dynamic Programming Paradigm
        def DPStoreSum(self):
            G = self.BinaryImageArray
            G = (1*G)
            SUM = sp.zeros((G.shape[0],G.shape[1]), dtype = np.int )
            for i in range(1, SUM.shape[0]):
                SUM[0][i] = (G[0][i]+SUM[0][i-1])
            for i in range(1, SUM.shape[1]):
                SUM[i][0] = (G[i][0]+SUM[i-1][0])
            for i in range(1,SUM.shape[0]):
                for j in range(1,SUM.shape[1]):
                    SUM[i][j] = ( G[i][j] + (SUM[i-1][j] + SUM[i][j-1] - SUM[i-1][j-1]) )
            return SUM

        #This function returns the total number of set pixels ( i.e 1s ) in a box which is upper bounded by Index1 and lower bounded by Index2
        #The operation done in this function is an O(1) function - Query the indices of an array and return the result
        def CountPixelsInBox(self,Index1,Index2):
            if Index1[0] > Index2[0] or Index1[1] > Index2[1]:
               return -1
            if Index1[0] == 0 and Index1[1] == 0:
               return self.CumSum[Index2[0]][Index2[1]]
            elif Index1[0] == 0:
                return self.CumSum[Index2[0]][Index2[1]] - self.CumSum[Index2[0]][Index1[1] - 1]
            elif Index1[1] == 0:
                return self.CumSum[Index2[0]][Index2[1]] - self.CumSum[Index1[0] - 1][Index2[1]]
            elif Index2[0]==0:
                 return self.CumSum[0][Index2[1]] - self.CumSum[0][Index1[1]-1]
            elif Index2[1] == 0:
                 return self.CumSum[Index2[0]][0] - self.CumSum[Index1[0]-1][0]
            else:
                 return self.CumSum[Index2[0]][Index2[1]] - \
                        self.CumSum[Index2[0]][Index1[1]-1] - \
                        self.CumSum[Index1[0]-1][Index2[1]] + \
                        self.CumSum[Index1[0]-1][Index1[1]-1]

        #Implement BoxCounting
        # It is assumed that G is a N x N square array containing the binary image, where N is a perfect power of 2
        # 1/Si is the input,
        # This function implements the BFS traversal
        def BoxCount(self,squareDimentions):
            self.Queue.addRear(((0,0),(squareDimentions-1,squareDimentions-1),0))
            self.BoxLevel = {0:1}
            while self.Queue.isEmpty() != True:
                  Element = self.Queue.removeFront()
                  NoOfPixels = self.CountPixelsInBox(Element[0],Element[1])
                  x1 = Element[0][0]
                  y1 = Element[0][1]
                  x2 = Element[1][0]
                  y2 = Element[1][1]
                  if NoOfPixels > 0:
                     if Element[2] in self.BoxLevel.keys():
                          self.BoxLevel[Element[2]] += 1
                     else:
                          self.BoxLevel[Element[2]] = 1
                     #if x1==x2 or y1==y2:
                     if Element[2] == math.log(squareDimentions,2):
                         continue
                     self.Queue.addRear( ((x1,y1)                             ,  ( int((x1+x2)/2)  , int((y1+y2)/2) ), Element[2]+1) )
                     self.Queue.addRear( ((x1,1+int((y1+y2)/2))               ,  ( int((x1+x2)/2),y2)                , Element[2]+1) )
                     self.Queue.addRear( ((1+int((x1+x2)/2),y1)               ,  ( x2, int((y1+y2)/2))               , Element[2]+1) )
                     self.Queue.addRear( ((1+int((x1+x2)/2),1+int((y1+y2)/2)) ,  ( x2,y2)                           ,  Element[2]+1) )
                  elif NoOfPixels <= 0:
                      continue


        #Plot the points
        def PlotPoints(self):
            Lx = []
            Ly = []
            for key in self.BoxLevel:
                #print(key," = ", self.BoxLevel[key])
                Lx.append( math.log(2**key,10))
                Ly.append( math.log(self.BoxLevel[key],10))
            mp.plot(Lx,Ly,'ro')
            mp.plot(Lx,Ly,'g')
            mp.xlabel(" log 1/si ---->")
            mp.ylabel(" log ni ---->")
            #mp.show()

            #Use curve_fit to find the estimates. It uses Least Square internally
            y0 = np.array(Ly)
            x0 = np.array(Lx)
            initial = [Lx[0],Ly[0]]
            parg, popt = curve_fit(fit,x0,y0,initial)
            print("\nParameters = ",parg)
            print("\nCovariance Matrix = ",popt)
            print("\n Fractal Dimention of Image (D)= ",parg[1])
            #Plot the best fit line
            mp.plot(x0, fit(x0, *parg), 'b-')
            mp.show()





imgName = 'tree-2'
f = msc.imread(imgName+'.png', flatten=True).astype(np.float)
#msc.imsave('bitmap.png',g,format=None)

FD = FractalDimention(f)
FD.BoxCount(f.shape[0]) # It is assumed to be a square image, so shape[0] = shape[1]
FD.PlotPoints()









