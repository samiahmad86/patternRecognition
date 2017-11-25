import numpy as np
import scipy.misc as msc
import scipy.ndimage as img
import scipy as sp
import math
from scipy.optimize import curve_fit
import matplotlib.pyplot as mp
import timeit
import os

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
        def __init__(self,f,imgName):
            self.imgName = imgName
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
        def PlotPoints(self,LogDir,starttime,endtime):
            fileio = open(LogDir + "LogFile.txt", "a+")
            Lx = []
            Ly = []
            for key in self.BoxLevel:
                Lx.append( math.log(2**key,10))
                Ly.append( math.log(self.BoxLevel[key],10))

            #Basic Plotting of the scattered points obtained from the calculations and the curve through them
            mp.plot(Lx,Ly,'ro',label='Points -> ( log(ni) , log(1/si) )')
            mp.plot(Lx,Ly,'y',label=' Curve through the points')

            #Use curve_fit to find the estimates. It uses Least Square internally
            y0 = np.array(Ly)
            x0 = np.array(Lx)
            initial = [Lx[0],Ly[0]]

            #parg will contain the set of equation parameters, including D which we want to finc from the calculation
            parg, popt = curve_fit(fit,x0,y0,initial)

            #Write to the log file
            fileio.write("\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            fileio.write(str("\n Calculations for Image = "+self.imgName+".png"))
            fileio.write(str("\nParameters = ")+str(parg))
            fileio.write(str("\nCovariance Matrix = ")+str(popt))
            fileio.write(str("\n Total Time elapsed :: ")+str(endtime-starttime)+" units")
            fileio.write(str("\n Fractal Dimention of Image (D)= "+str(parg[1])))
            fileio.write("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

            #Plot the best fit line
            mp.plot(x0, fit(x0, *parg), 'b-', label = 'Best Fit Line')

            #Add labels to the image
            mp.grid(True)
            mp.legend()
            mp.title(self.imgName+'.png')
            mp.xlabel(str(" log 1/si ----> ( Log Base 10 ) Fractal Dimention of Image = ")+str(parg[1]))
            mp.ylabel(" log ni ----> ( Log Base 10 )")
            mp.savefig(LogDir+"Plot-"+imgName+".pdf",facecolor='w', edgecolor='w',
                        papertype=None, format='pdf', transparent=False,
                        bbox_inches='tight', pad_inches=0.1)

            mp.show()
            fileio.close()


#Wrapper Function to implement Box Counting and generate the relevant files
def CalculateFractalDimention(CurrDir,imgName,LogDir):
    f = msc.imread(CurrDir+"//"+imgName+'.png', flatten=True).astype(np.float)


    starttime = timeit.default_timer()
    FD = FractalDimention(f,imgName)
    FD.BoxCount(f.shape[0]) # It is assumed to be a square image, so shape[0] = shape[1]
    endtime = timeit.default_timer()
    msc.imsave(LogDir + imgName + '-binary.png', FD.BinaryImageArray, format=None)
    FD.PlotPoints(LogDir,starttime,endtime)



ListofImages=['tree','lightning-3']
LogDir = os.path.dirname(os.path.realpath(__file__)) +"//Logs_FractalDimention//"
try:
  os.stat(LogDir)
except:
  os.mkdir(LogDir)
for imgName in ListofImages:
    CalculateFractalDimention(os.getcwd(),imgName,LogDir)






