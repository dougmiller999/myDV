# this file generates files of test data
# FRI January 31, 2025

import numpy as np
import sys

if len(sys.argv) > 1:
    filename = sys.argv[1]
else:
    filename = 'data.gnu'

def f(x):
    return x*x

def g(x):
    return x*x*x

def h(x):
    return np.sin(x)

funclist = [f,g,h]

A = [0.5, 1.0, 0.1]
y = []

for a,func in zip(A,funclist):
    x = np.linspace(0,4.0)
    y.append( func(x) + a*np.random.random(len(x)) )

f = open(filename, 'w')

f.write('# time  rabbits  foxes  bears\n')
for k in range(len(x)):
    string = "%f  %f  %f  %f" % (x[k], y[0][k], y[1][k], y[2][k])
    f.write(string+'\n')

f.close()        
