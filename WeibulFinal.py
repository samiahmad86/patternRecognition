import numpy as np
import scipy.optimize as scp
import math as mth
from numpy.linalg import inv
import matplotlib.pyplot as mp
import scipy as sh


import math as mth
import scipy.stats as sh
import scipy



# Fit the parameters using Maximum Likelihood Parameters
#def WeibullFit(DataArray,iterations,stop):

def weibull(x,alpha,beta):
    return (beta/alpha)*( (x/alpha)**(beta-1) )* ( mth.exp ( - (( x/alpha )**beta) ))


#def LikelihoodFunc(a,k,x):
#    N = x.shape[0]
#    return N*(np.log(k)-k*np.log(a)) + (k-1)*np.sum(x) - np.sum( np.exp(x/a,k) )

def sum1(data,a,k):
    sum = 0.0
    for i in data:
        sum += ((i/a)**k) * mth.log(i/a)
    return sum

def sum2(data,a,k):
    sum = 0.0
    for i in data:
        sum += ((i/a)**k)
    return sum

def sum3(data,a,k):
    sum = 0.0
    for i in data:
        sum += ((i/a)**k) * mth.log(i/a,2) * mth.log(i/a,2)
    return sum

def sum4(data):
    sum = 0.0
    for i in data:
        sum += mth.log(i)
    return sum



def DerivativeLK(x, k, a):
    N = len(x)
    return (N/k) - (N*mth.log(a)) + sum4(x) - sum1(x,a,k)

def DerivativeLA(x, k, a):
    N = len(x)
    return (k/a)*( sum2(x,a,k) - N )

def SecondOrderDerivativeLK( x,k,a ):
    N = len(x)
    return -(N/(k*k)) - sum3(x,a,k)

def SecondOrderDerivativeLA( x,k,a ):
    N = len(x)
    return (k/(a*a))*( N - (k+1)*sum2(x,a,k) )

def SecondOrderLAK( x,k,a):
    N = len(x)
    return (1.0/a)*sum2(x,a,k) + (k/a)*sum1(x,a,k)  - (N/a)


class FitWeibull:
     def __init__(self, FileName):

        dt = np.dtype([('dates', np.str), ('count', np.int)])
        data = np.loadtxt('myspace.csv', dtype=dt, comments='#', delimiter=',')
        hs = np.array([d[1] for d in data])
        hs = np.trim_zeros(hs, 'f')

        self.h = hs
        #self.N = len(self.h)
        #x = np.arange(1, 1 + len(self.h), 1)
        #self.x = x.tolist()

        import collections as c
        dict = c.Counter(hs)
        self.d = []
        for i in range(0,len(hs)):
            hs[i] = dict[hs[i]]*hs[i]
           #self.d.append(key*dict[key])
           # print(key,dict[key])

        print(hs)
        self.d = hs
        x = np.arange(1, 1 + len(hs), 1)
        self.x = x.tolist()
        #print(self.d)


        '''
        data = np.loadtxt('myspace.csv', dtype=dt, comments='#', delimiter=',')
        hs = np.array([d[1] for d in data])
        hs = np.trim_zeros(hs, 'f')
        hs = hs.tolist()
        
        s = set(hs)
        print(s)
        hf = []
        for i in s:
            hf.append(hs.count(i))
        print(hf)
        self.h = hf


        '''
        #x = np.arange(1, 1 + len(hs), 1)
        #self.x = x.tolist()


     def iterateNewtonMethod(self,s):
       a = 1
       k = 1
       for i in range(20):
           #Construct hessian matrix
           v1 = SecondOrderDerivativeLK(s,k,a)
           v2 = SecondOrderLAK(s,k,a)
           v3 = SecondOrderDerivativeLA(s,k,a)
           #print(v1,v2,v3)

           hessianmatrix = np.array([[v1,v2],[v2,v3]])
           gradient = np.array([[ - (DerivativeLK(s,k,a)) , - (DerivativeLA(s,k,a)) ]] ).transpose()

           #hessianmatrix = nd.hessian(LikelihoodFunc)[k,a,self.h]

           alpha = np.array([[k,a]]).transpose()
           product = np.multiply(inv(hessianmatrix),gradient)
           alpha = np.add(alpha,product)
           k = alpha[0][0]
           a = alpha[1][0]
           #print(alpha)


       return a,k





p = FitWeibull('myspace.csv')
a,k = p.iterateNewtonMethod(p.d)
print("\n By out function = ",a,k)


Weibull_parameters = sh.exponweib.fit(p.d, floc=0, f0=1)
Beta, Alpha = Weibull_parameters[1], Weibull_parameters[3]
print("\n By exponweib = ",Weibull_parameters)

x0=[]
for i in range(0,len(p.d)):
    x0.append(i)

p0 = []
for i in (x0):
    p0.append(weibull(i,215.4,2.5))

mp.plot(x0,p.h)
#mp.plot(x0, p0,'r-')
mp.show()


#mp.plot(p.x,weibull(p.x,a,k))
#mp.show()


