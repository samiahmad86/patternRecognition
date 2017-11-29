import pylab
import math
import numpy as np
import scipy as sp
import scipy.stats as stats
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid.axislines import SubplotZero
from numpy import inf

if __name__ == "__main__":

    # read data as 2D array of data type 'object'
    data = np.loadtxt('../whData.dat', dtype=np.object, comments='#', delimiter=None)

    # read height and weight data into 2D array (i.e. into a matrix)
    X = data[:, 0:2].astype(np.float)

    # removing negative and zeros from both columns
    X = X[X[:, 1] > 0, :]
    X = X[X[:, 0] > 0, :]

    # body height only
    height = X[:, 1]

    # Mean and standard deviation of height
    height.sort()
    hMean = np.mean(height)
    hStd = height.std()
    print("Mean of the sample:", hMean)
    print("Standard Deviation of the sample:",hStd)

    range = np.arange(140, 210, 0.001)
    pdf = stats.norm.pdf(range, hMean, hStd)

    plt.plot(height, np.zeros_like(height) + 0, 'o', label="data")
    plt.plot(range, pdf, color="yellow", linewidth=1, linestyle="-", label="normal")
    plt.legend(loc='upper right')
    plt.savefig('NormalDistribution.pdf', facecolor='w', edgecolor='w',
                papertype=None, format='pdf', transparent=False,
                bbox_inches='tight', pad_inches=0.1)
    plt.show()
