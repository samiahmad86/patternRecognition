import pylab
import math
import numpy as np
import scipy as sp
import scipy.stats as stats
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid.axislines import SubplotZero
from numpy import inf



def plotData2D(X, filename=None):
    # create a figure and its axes
    fig = plt.figure()
    axs = fig.add_subplot(111)

    # see what happens, if you uncomment the next line
    # axs.set_aspect('equal')
    
    # plot the data 
    axs.plot(X[0,:], X[1,:], 'ro', label='data')

    # set x and y limits of the plotting area
    xmin = X[0,:].min()
    xmax = X[0,:].max()
    axs.set_xlim(xmin-10, xmax+10)
    axs.set_ylim(-2, X[1,:].max()+10)

    # set properties of the legend of the plot
    leg = axs.legend(loc='upper left', shadow=True, fancybox=True, numpoints=1)
    leg.get_frame().set_alpha(0.5)
    filename = None
    # either show figure on screen or write it to disk
    if filename == None:
        plt.show()
    else:
        plt.savefig(filename, facecolor='w', edgecolor='w',
                    papertype=None, format='pdf', transparent=False,
                    bbox_inches='tight', pad_inches=0.1)
    plt.close()
    




if __name__ == "__main__":
    #######################################################################
    # 1st alternative for reading multi-typed data from a text file
    #######################################################################
    # define type of data to be read and read data from file
    dt = np.dtype([('w', np.float), ('h', np.float), ('g', np.str_, 1)])
    data = np.loadtxt('whData.dat', dtype=dt, comments='#', delimiter=None)

    # read height, weight and gender information into 1D arrays
    ws = np.array([d[0] for d in data])
    hs = np.array([d[1] for d in data])
    gs = np.array([d[2] for d in data]) 


    ##########################################################################
    # 2nd alternative for reading multi-typed data from a text file
    ##########################################################################
    # read data as 2D array of data type 'object'
    data = np.loadtxt('whData.dat',dtype=np.object,comments='#',delimiter=None)

    # read height and weight data into 2D array (i.e. into a matrix)
    X = data[:,0:2].astype(np.float)


    # read gender data into 1D array (i.e. into a vector)
    y = data[:,2]

    # removing negative and zeros from both columns
    X = X[X[:, 1] > 0, :]
    X = X[X[:, 0] > 0, :]

    # body height only
    height = X[:,1]

    # let's transpose the data matrix 
    X = X.T

    # TODO Uncomment this before submission
    # now, plot weight vs. height using the function defined above
    # plotData2D(X, 'plotWH.pdf')

    # next, let's plot height vs. weight 
    # first, copy information rows of X into 1D arrays
    w = np.copy(X[0,:])
    h = np.copy(X[1,:])
    
    # second, create new data matrix Z by stacking h and w
    Z = np.vstack((h,w))

    # TODO Uncomment this
    # third, plot this new representation of the data
    # plotData2D(Z, 'plotHW.pdf')

    # Mean and standard deviation of height
    height.sort()
    hMean = np.mean(height)
    hStd = height.std()
    print(hMean)
    print(hStd)

    range = np.arange(140,210,0.005)
    pdf= stats.norm.pdf(range, hMean, hStd)
   # plt.plot(height, np.zeros_like(height) + 0, 'o', label="data")
    # plt.plot(range, pdf, color="yellow", linewidth=1, linestyle="-", label="normal")
  #  plt.legend(loc='upper right')
  #  plt.show()


    # Consider the Lp norm for p = 1/2 and plot the corresponding  unit circle
    def plotUnitCircle(p):
        """ plot some 2D vectors with p-norm < 1 """
        fig = plt.figure(1)
        ax = SubplotZero(fig, 111)
        fig.add_subplot(ax)

        for direction in ["xzero", "yzero"]:
            ax.axis[direction].set_axisline_style("-|>")
            ax.axis[direction].set_visible(True)

        for direction in ["left", "right", "bottom", "top"]:
            ax.axis[direction].set_visible(False)


        x = np.linspace(-1.0, 1.0, 1000)
        y = np.linspace(-1.0, 1.0, 1000)
        X, Y = np.meshgrid(x, y)
        F = (((abs(X)**p+abs(Y)**p)**(1.0/p)) - 1)
        plt.contour(X, Y, F, [0])
        plt.show()


    # plotUnitCircle(1)

    plotUnitCircle(0.5)