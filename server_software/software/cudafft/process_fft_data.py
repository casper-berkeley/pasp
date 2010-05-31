#!/usr/bin/env python2.6

import numpy
import pylab
import matplotlib

barwidth=8096
data = numpy.genfromtxt("out.tsv")
#datat = data.transpose()
#pylab.bar(datat[0],datat[2])
#pylab.show()
matplotlib.pyplot.cla()

#pylab.xscale('log') 




for i in range(len(data)):
    if data[i][1]==1:
        patch=pylab.bar(data[i][0],data[i][3]+data[i][4]+data[i][5], width=barwidth)
        patch=pylab.bar(data[i][0],data[i][3]+data[i][4], width=barwidth, color='g')
        patch=pylab.bar(data[i][0],data[i][3], width=barwidth, color='r')
    
pylab.show()       


