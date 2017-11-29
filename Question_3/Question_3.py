import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as mp
import scipy as sh
import math as mth
import scipy.stats as sh


# Fit the parameters using Maximum Likelihood Parameters
# def WeibullFit(DataArray,iterations,stop):

def weibull(x, alpha, beta):
    return (beta / alpha) * ((x / alpha) ** (beta - 1)) * (mth.exp(- ((x / alpha) ** beta)))

def plotData2D(X1, X2, filename=None):
    # create a figure and its axes
    fig = mp.figure()
    axs = fig.add_subplot(111)

    # plot the data
    axs.plot(X1[0, :], X1[1, :], label='Google data')
    axs.plot(X2[0, :], X2[1, :], label='Weibull fit')

    # set x and y limits of the plotting area
    xmin = 0
    xmax = 500
    axs.set_xlim(0, 500)
    axs.set_ylim(0, 110)

    # set properties of the legend of the plot
    leg = axs.legend(loc='upper left', shadow=True, fancybox=True, numpoints=1)
    leg.get_frame().set_alpha(0.5)

    # either show figure on screen or write it to disk
    if filename == None:
        mp.show()
    else:
        mp.savefig(filename, facecolor='w', edgecolor='w',
                    papertype=None, format='pdf', transparent=False,
                    bbox_inches='tight', pad_inches=0.1)
    mp.close()


class FitWeibull:
    def __init__(self, FileName):
        dt = np.dtype([('dates', np.str), ('count', np.int)])
        data = np.loadtxt('../myspace.csv', dtype=dt, comments='#', delimiter=',')
        RawData = np.array([d[1] for d in data])
        frequency = []
        valueIndex = []
        for i in range(len(RawData)):
            valueIndex.append(i)
            frequency.append(RawData[i])
        # Observations data
        data = []
        for i in range(len(frequency)):
            for j in range(frequency[i]):
                if valueIndex[i] > 0:
                    data.append(valueIndex[i])

        self.h = data
        self.original = RawData

    def iterateNewtonMethod(self, s):
        a = 1.0
        k = 1.0
        N = len(s)
        for i in range(25):
            alphaMatrix = np.array([[k], [a]])
            Sum1 = 0.0
            # Sigma[ (d(i)/alpha)^k  * log(d(i)/alpha)
            Sum2 = 0.0
            # Sigma[ d(i)/alpha)^k]
            Sum3 = 0.0
            # Sigma[ (d(i)/alpha)^k * log(d(i)/alpha)^2
            Sum4 = 0.0
            for i in range(len(s)):
                Sum1 += mth.log(s[i])
                Sum2 += (s[i] / a) ** k * mth.log(s[i] / a)
                Sum3 += (s[i] / a) ** k
                Sum4 += (s[i] / a) ** k * mth.log(s[i] / a, 2) ** 2
            # d(L)/d(k)
            d1 = N / k - N * mth.log(a) + Sum1 - Sum2
            # d(L)/d(alpha)
            d2 = (k / a) * (Sum3 - N)
            # dd(L)/d(k)^2
            d3 = (-1 * N) / (k * k) - Sum4
            # dd(L)/d(alpha)^2
            d4 = (k / (a * a)) * (N - ((k + 1) * Sum3))
            # dd(L)/ d(alpha) d(k)
            d5 = (1.0 / a) * Sum3 + (k / a) * Sum2 - N / a

            hessianmatrix = np.matrix([[d3, d5], [d5, d4]])
            gradient = np.matrix([[-1.0 * d1], [-1.0 * d2]])
            product = inv(hessianmatrix) * gradient
            alphaMatrix = alphaMatrix + product
            k = alphaMatrix[0, 0]
            a = alphaMatrix[1, 0]
        return a, k

    def CreatePlotData(self,alpha,k):
        X = []
        Y = []
        for i in range(len(self.original)):
            X.append(i)
            Y.append(self.original[i])

        # Create the original Plot Data
        P1 = np.array([X, Y])

        # Creating Weibull data by calculated value of k and alpha
        x = np.arange(1.0, len(self.original))
        y =[]
        for i in x:
            y.append(weibull(i, alpha, k))

        scaleFactor =  100.0 / max(y)

        # Ploting Weibull fit

        P2 = np.array([x, np.array(y) * scaleFactor])
        plotData2D(P1, P2, 'WeibullDistributionFit.pdf')
        plotData2D(P1, P2)


plot = FitWeibull('../myspace.csv')
a, k = plot.iterateNewtonMethod(plot.h)

print("\n Obtained by Iterative procedure = ", a, k)
Weibull_parameters = sh.exponweib.fit(plot.h, floc=0, f0=1)
print("\n Obtained by using exponweib = ",Weibull_parameters[1],Weibull_parameters[3])

plot.CreatePlotData(a,k)


