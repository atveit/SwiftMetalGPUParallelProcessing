import pandas
import numpy
import sys
import numpy as np
import matplotlib as mpl
#mpl.use('Agg') # alternative without display - http://stackoverflow.com/questions/4931376/generating-matplotlib-graphs-without-a-running-x-server
import matplotlib.pyplot as plt

filename = sys.argv[1] 

d = pandas.read_table(filename, sep="\t")

# descriptive statistics
print d.describe()

print d.keys()

print type(d['vectorsize as 2^x'])

#plt.scatter(d['vectorsize as 2^x'].values, )

print d['metal gpu time (microsec)']

import math
ts = pandas.Series(np.random.randn(1000), index=pandas.date_range('1/1/2000', periods=1000))
df = pandas.DataFrame(np.random.randn(1000, 4), index=ts.index, columns=list('ABCD'))
df = df.cumsum()
plt.figure(); df.plot();

df = pandas.Series(d['metal relative to accelerate'], index=d['vectorsize as 2^x'])

print df

plt.ioff() # interactive mode off
plt.figure()
plt.grid(True)
#plt.cumsum()
plt.plot(x=d['vectorsize as 2^x'], y=d['metal relative to accelerate'])
plt.show()
plt.savefig("yoda.png")
plt.close()
