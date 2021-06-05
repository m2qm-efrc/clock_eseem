#! /usr/bin/env python3
"""signal processing, phase shifter and mixer"""

import sys
import numpy
from scipy.fft import fft, ifft

time = []
signal = []

with open('20_tt.out', 'r') as f:
    lines = f.readlines()
for line in lines:
    tt = line.split()
    time.append(float(tt[0]))
    signal.append(float(tt[1]))

signal = numpy.array(signal)

#p_shift = -1.0 * signal
#
#mixer = []
#for i in range(len(signal)):
#    mixer.append( signal[i] * p_shift[i] )
#print(len(signal))
#print(len(p_shift))

y = numpy.fft.rfft(signal)

#for i in range(len(y)):
#    if i > 500:
#        y[i] = y[i]/(100000)
#
#x = numpy.fft.irfft(y)
for i in range(len(y)):
    print(i, numpy.abs(y[i]))
