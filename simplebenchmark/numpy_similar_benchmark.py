data = range(1000000)
import numpy
import time
from numpy import exp as ef
d = numpy.array(data)
t0 = time.time()
f = 1.0/(1+ef(-d))
t1 = time.time()
delta = t1-t0
print "sigmoid on 1 million elements took %10.2f microseconds" %( delta*1000000.0)
