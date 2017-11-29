
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import math


# weibull probability density function , x is the set of data
def weibull(x, a, k):
    return (k / a) * (x / a) ** (k - 1) * np.exp(-(x / a) ** k)


def plotData2D(X1, X2, filename=None):
    # create a figure and its axes
    fig = plt.figure()
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
        plt.show()
    else:
        plt.savefig(filename, facecolor='w', edgecolor='w',
                    papertype=None, format='pdf', transparent=False,
                    bbox_inches='tight', pad_inches=0.1)
    plt.close()


if __name__ == "__main__":

    # read data
    dt = np.dtype([('D', np.dtype('a40')), ('V', np.int32)])
    data = np.loadtxt('myspace.csv', dtype=dt, comments='#', delimiter=',')
    Temp = np.array([d[1] for d in data])

    # wi[i] is the frequency of vi[i]
    wi = []
    vi = []
    for i in range(len(Temp)):
        vi.append(i)
        wi.append(Temp[i])
    print(wi)
    # Observations data
    data = []
    for i in range(len(wi)):
        for j in range(wi[i]):
            if vi[i] > 0:
                data.append(vi[i])


    print(data)
                # initial value of k and alpha
    k = 1.
    alpha = 1.
    N = len(data)

    # Run Newton's method for 25 iteration
    for L in range(25):

        # Sigma[ log(di) ]
        Sigma1 = 0.0

        # Sigma[ (d(i)/alpha)^k  * log(d(i)/alpha)
        Sigma2 = 0.0

        # Sigma[ d(i)/alpha)^k]
        Sigma3 = 0.0

        # Sigma[ (d(i)/alpha)^k * log(d(i)/alpha)^2
        Sigma4 = 0.0

        for i in range(len(data)):
            Sigma1 += math.log(data[i])
            Sigma2 += (data[i] / alpha) ** k * math.log(data[i] / alpha)
            Sigma3 += (data[i] / alpha) ** k
            Sigma4 += (data[i] / alpha) ** k * math.log(data[i] / alpha, 2) ** 2

        # d(L)/d(k)
        par1 = N / k - N * math.log(alpha) + Sigma1 - Sigma2

        # d(L)/d(alpha)
        par2 = (k / alpha) * (Sigma3 - N)

        # dd(L)/d(k)^2
        par3 = (-1 * N) / (k * k) - Sigma4

        # dd(L)/d(alpha)^2
        par4 = (k / (alpha * alpha)) * (N - ((k + 1) * Sigma3))

        # dd(L)/ d(alpha) d(k)
        par5 = (1.0 / alpha) * Sigma3 + (k / alpha) * Sigma2 - N / alpha

        # simultaneous equations
        M0 = np.matrix([[k], [alpha]])

        # Matrix inversion
        M1 = np.linalg.inv(np.matrix([[par3, par5], [par5, par4]]))
        M2 = np.matrix([[-1 * par1], [-1 * par2]])

        M0 = M0 + M1 * M2
        k = M0[0, 0]
        alpha = M0[1, 0]
        print( alpha,k )
        # Remove the comment to see the convergence of k and alpha
        # print("--------")
        # print("iteration= " + str(L) + "  K=" + str(k) + "   Alpha=" + str(alpha))

    X = []
    Y = []
    for i in range(len(Temp)):
        X.append(i)
        Y.append(Temp[i])

    # Plot Google Data
    P1 = np.array([X, Y])

    # Creating Weibull data by calculated value of k and alpha
    d = np.arange(1., len(Temp))
    scale = 100. / weibull(d, alpha, k).max()

    # Ploting Weibull fit
    P2 = np.array([d, weibull(d, alpha, k) * scale])

    plotData2D(P1, P2, 'Task03.pdf')
