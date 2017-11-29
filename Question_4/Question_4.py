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
    axs.plot(X[0, :], X[1, :], 'ro', label='data')

    # set x and y limits of the plotting area
    xmin = X[0, :].min()
    xmax = X[0, :].max()
    axs.set_xlim(xmin - 10, xmax + 10)
    axs.set_ylim(-2, X[1, :].max() + 10)

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
        F = (((abs(X) ** p + abs(Y) ** p) ** (1.0 / p)) - 1)
        ax.contour(X, Y, F, [0])
        plt.savefig('UnitCircle.pdf', facecolor='w', edgecolor='w',
                    papertype=None, format='pdf', transparent=False,
                    bbox_inches='tight', pad_inches=0.1)
        plt.show()

    plotUnitCircle(0.5)