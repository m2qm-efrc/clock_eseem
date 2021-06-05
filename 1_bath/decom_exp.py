#! /usr/bin/env python3
"""Suzuki Trotter dynamics of spin chain"""

import sys
import random
import numpy
import math
import scipy.linalg as LA
from cmath import sqrt
from itertools import product

N_bath = 1

N_Hil = 3 * 2**N_bath # spin one + nuclear clear spin 1/2
# estibalish basis 

# Pauli operators  ==================================
sgm_x = numpy.array([[0.0, 1.0], [1.0, 0.0]]) * 0.5
sgm_y = numpy.array([[0.0, -1j], [1j, 0.0]])  * 0.5
sgm_z = numpy.array([[1.0, 0.0], [0.0, -1.0]]) * 0.5
sgm_p = numpy.array([[0.0, 1.0], [0.0,  0.0]])
sgm_n = numpy.array([[0.0, 0.0], [1.0,  0.0]])
ident = numpy.array([[1.0, 0.0], [0.0,  1.0]])
#=====================================================

# Spin One operators =================================
S1x = numpy.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]) / sqrt(2.0)
S1y = numpy.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 1.0], [0.0, -1.0, 0.0]]) / (sqrt(2.0) * 1j)
S1z = numpy.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
S1p = numpy.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]) * sqrt(2.)
S1n = numpy.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]) * sqrt(2.)
#=====================================================

#set up for dynamics ===============================================
dt = 0.01
#A = 10. # intensity of pulse
J = 0.2 * 1.07
#Be = 50.0 #658.169 * Bp and Bp = 1; Be = 658.169 * Bp
Be = 20
Bp = Be / 658.1691772885284
D = -100.0 * 1.07
E = 10.0 * 1.07
a = 10.0
HF = 0.01 * 1.07 #hyperfine coupling
pHF = 0.01 * 1.07 # pesudo-hyperfine coupling
#===================================================================
# about unit of time 
# for proton H = omega_0 Sz
# for B = 1T, omega_0 = 42.578E6 s^-1
# Energy unit is 28.025 MHz
# so units for time is 3.568E-08s, 0.03568 microsecond
#====================================================================

# interaction =======================================================

ano_iso = (-1./3.* D + E) * S1x @ S1x + (-1./3.* D - E) * S1y @ S1y + 2./3.* D * S1z @ S1z -a * S1z 
w_e,v_e = LA.eigh(ano_iso + Be*S1z)

hyperfine = HF * numpy.kron(S1z, sgm_z) + pHF * (numpy.kron(S1z, sgm_x) + numpy.kron(S1z, sgm_y))
H1 = numpy.kron( ano_iso, numpy.eye(2)) + numpy.kron( Be*S1z, numpy.eye(2))  + numpy.kron( numpy.eye(3), Bp* sgm_z) + hyperfine
H2 = numpy.kron( ano_iso, Bp* sgm_z) + numpy.kron( Be*S1z, numpy.eye(2)) + hyperfine
print(numpy.allclose(H1,H2))

w,v = LA.eigh(H1)
print(w)
print(numpy.allclose(numpy.matrix(v).getH(), numpy.linalg.inv(v))) # orthogonal matrix, conjugate transpose equal inverse
print(numpy.allclose(H1, v @ numpy.diag(w)@ numpy.matrix(v).getH())) # eigenvalue decomposition
# wave-function initilization, tensor product of electronic spin and nuclear spins ===
wf_e = []
phase = 0.0
wf_e = v_e[:,0] + numpy.exp(-1j * phase * 2 * numpy.pi)* v_e[:,1]
norm = numpy.linalg.norm(wf_e)
wf_e = wf_e / norm
print(wf_e)
Mz0 = numpy.einsum('i, ij, j', wf_e, S1z, wf_e)
print('initial Mz {:10.5f}'.format(abs(Mz0)))

wf_n = []
for i in range(2):
    tt = random.uniform(-1, 1) + 1j * random.uniform(-1, 1)
    wf_n.append(tt)

norm = numpy.linalg.norm(wf_n)
wf_n = wf_n / norm

wf0 = numpy.kron(wf_e, wf_n)
#===========================================================================

# reduced density matrix ===================================================
def rdm(wf):
    """ reduced density matrix for spin-one from wave-function """
    dm = numpy.zeros((3,3), dtype=complex)
    sb = [0,1]
    for i in range(3):
        for j in range(3):
            left_ind = 2*i; right_ind = 2*j
            for q in (sb):
                left_ind += q
                right_ind += q
                dm[i,j] += numpy.conj(wf[left_ind]) * wf[right_ind]
    return dm
#===========================================================================

peroid = 2. * numpy.pi / (w_e[1] - w_e[0])
print('peroid {:10.5f}'.format(peroid))

# simulation ===============================================================
A = S1x @ S1y + S1y @ S1x
B = LA.expm(-1j * numpy.pi / 2. * A)
R = numpy.kron(B, numpy.eye(2))
T_tot = 400000.
T = peroid
exp_dt = numpy.array( v @ numpy.diag( numpy.exp(-1j * dt * w ) ) @ numpy.matrix(v).getH())

while T < T_tot:
    wf = numpy.copy(wf0)
    exp_H = numpy.array( v @ numpy.diag( numpy.exp(-1j * T * w) ) @ numpy.matrix(v).getH() )
    #exp_H = LA.expm( -1j * T * H)
    wf = exp_H @ wf
    wf = R @ wf
    wf = exp_H @ wf
    dm = rdm(wf)
    Mz = numpy.trace(dm@S1z).real
    ##print(numpy.linalg.norm(wf))
    s_t = 0.0
    sec_mag = []
    while s_t <= 1.2 * peroid:
        wf = exp_dt @ wf
        dm = rdm(wf)
        Mz = numpy.trace(dm@S1z).real
        sec_mag.append(Mz)
        s_t += dt
    ra = (max(sec_mag) - min(sec_mag)) / (2. * abs(Mz0))
    print('{:10.5f} {:10.8f}'.format(2*T, ra))
    #print('{:10.5f} {:10.8f}'.format(T, Mz))
    T += dt * 10 
