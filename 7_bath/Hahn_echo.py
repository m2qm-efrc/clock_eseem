#! /usr/bin/env python3
"""Suzuki Trotter dynamics of spin chain"""

import sys
import random
import numpy 
import math
import scipy.linalg as LA
from cmath import sqrt
from itertools import product
from numpy import kron, eye

N_bath = 7

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
#Be = 50.0 #658.169 * Bp and Bp = 1; Be = 658.169 * Bp
Be = 7
Bp = Be / 658.1691772885284
D = -100.0 * 1.07
E = 10.0 * 1.07
a = 10.0
HF = 0.01 * 1.07 #hyperfine coupling
pHF = 0.01 * 1.07 # pesudo-hyperfine coupling
J = 0.0005
#===================================================================
# about unit of time 
# for proton H = omega_0 Sz
# for B = 1T, omega_0 = 42.578E6 s^-1
# Energy unit is 28.025 MHz
# so units for time is 3.568E-08s, 0.03568 microsecond
#====================================================================

# interaction =======================================================
#Hei = J * numpy.kron(sgm_x, sgm_x) + J * numpy.kron(sgm_y, sgm_y) + J * numpy.kron(sgm_z, sgm_z)

ano_iso = (-1./3.* D + E) * S1x @ S1x + (-1./3.* D - E) * S1y @ S1y + 2./3.* D * S1z @ S1z -a * S1z 
w_e,v_e = LA.eigh(ano_iso + Be*S1z)
#print(w)
#print(v)

hyperfine = HF * kron(S1z, sgm_z) + pHF * (kron(S1z, sgm_x) + kron(S1z, sgm_y))
for i in range(N_bath -1):
    hyperfine = kron(hyperfine, eye(2))

hyperfine +=  1.3  * HF  * kron(kron(kron(kron(kron(kron(kron(S1z, eye(2)), sgm_z), eye(2)), eye(2)), eye(2)), eye(2)), eye(2))
hyperfine +=  1.3  * pHF * kron(kron(kron(kron(kron(kron(kron(S1z, eye(2)), sgm_x), eye(2)), eye(2)), eye(2)), eye(2)), eye(2))
hyperfine +=  1.3  * pHF * kron(kron(kron(kron(kron(kron(kron(S1z, eye(2)), sgm_y), eye(2)), eye(2)), eye(2)), eye(2)), eye(2))

hyperfine +=  1.2  * HF  * kron(kron(kron(kron(kron(kron(kron(S1z, eye(2)), eye(2)), sgm_z), eye(2)), eye(2)), eye(2)), eye(2))
hyperfine +=  1.2  * pHF * kron(kron(kron(kron(kron(kron(kron(S1z, eye(2)), eye(2)), sgm_x), eye(2)), eye(2)), eye(2)), eye(2))
hyperfine +=  1.2  * pHF * kron(kron(kron(kron(kron(kron(kron(S1z, eye(2)), eye(2)), sgm_y), eye(2)), eye(2)), eye(2)), eye(2))

hyperfine +=  1.1 * HF  * kron(kron(kron(kron(kron(kron(kron(S1z, eye(2)), eye(2)), eye(2)), sgm_z), eye(2)), eye(2)), eye(2))
hyperfine +=  1.1 * pHF * kron(kron(kron(kron(kron(kron(kron(S1z, eye(2)), eye(2)), eye(2)), sgm_x), eye(2)), eye(2)), eye(2))
hyperfine +=  1.1 * pHF * kron(kron(kron(kron(kron(kron(kron(S1z, eye(2)), eye(2)), eye(2)), sgm_y), eye(2)), eye(2)), eye(2))

hyperfine +=  0.9 * HF  * kron(kron(kron(kron(kron(kron(kron(S1z, eye(2)), eye(2)), eye(2)), eye(2)), sgm_z), eye(2)), eye(2))
hyperfine +=  0.9 * pHF * kron(kron(kron(kron(kron(kron(kron(S1z, eye(2)), eye(2)), eye(2)), eye(2)), sgm_x), eye(2)), eye(2))
hyperfine +=  0.9 * pHF * kron(kron(kron(kron(kron(kron(kron(S1z, eye(2)), eye(2)), eye(2)), eye(2)), sgm_y), eye(2)), eye(2))

hyperfine +=  0.8 * HF  * kron(kron(kron(kron(kron(kron(kron(S1z, eye(2)), eye(2)), eye(2)), eye(2)), eye(2)), sgm_z), eye(2))
hyperfine +=  0.8 * pHF * kron(kron(kron(kron(kron(kron(kron(S1z, eye(2)), eye(2)), eye(2)), eye(2)), eye(2)), sgm_x), eye(2))
hyperfine +=  0.8 * pHF * kron(kron(kron(kron(kron(kron(kron(S1z, eye(2)), eye(2)), eye(2)), eye(2)), eye(2)), sgm_y), eye(2))

hyperfine +=  0.7 * HF  * kron(kron(kron(kron(kron(kron(kron(S1z, eye(2)), eye(2)), eye(2)), eye(2)), eye(2)), eye(2)), sgm_z)
hyperfine +=  0.7 * pHF * kron(kron(kron(kron(kron(kron(kron(S1z, eye(2)), eye(2)), eye(2)), eye(2)), eye(2)), eye(2)), sgm_x)
hyperfine +=  0.7 * pHF * kron(kron(kron(kron(kron(kron(kron(S1z, eye(2)), eye(2)), eye(2)), eye(2)), eye(2)), eye(2)), sgm_y)

for i in range(N_bath):
    ano_iso = kron(ano_iso, eye(2)) 

zeeman  = Be * kron(kron(kron(kron(kron(kron(kron(S1z, eye(2)), eye(2)), eye(2)), eye(2)), eye(2)),  eye(2)), eye(2))
zeeman += Bp * kron(kron(kron(kron(kron(kron(kron(eye(3), sgm_z), eye(2)), eye(2)), eye(2)), eye(2)), eye(2)), eye(2))
zeeman += Bp * kron(kron(kron(kron(kron(kron(kron(eye(3), eye(2)), sgm_z), eye(2)), eye(2)), eye(2)), eye(2)), eye(2))
zeeman += Bp * kron(kron(kron(kron(kron(kron(kron(eye(3), eye(2)), eye(2)), sgm_z), eye(2)), eye(2)), eye(2)), eye(2))
zeeman += Bp * kron(kron(kron(kron(kron(kron(kron(eye(3), eye(2)), eye(2)), eye(2)), sgm_z), eye(2)), eye(2)), eye(2))
zeeman += Bp * kron(kron(kron(kron(kron(kron(kron(eye(3), eye(2)), eye(2)), eye(2)), eye(2)), sgm_z), eye(2)), eye(2))
zeeman += Bp * kron(kron(kron(kron(kron(kron(kron(eye(3), eye(2)), eye(2)), eye(2)), eye(2)), eye(2)), sgm_z), eye(2))
zeeman += Bp * kron(kron(kron(kron(kron(kron(kron(eye(3), eye(2)), eye(2)), eye(2)), eye(2)), eye(2)), eye(2)), sgm_z)
#Hei = numpy.kron(eye(3), Hei)

Hei = J * kron(kron(kron(kron(kron(kron(kron(eye(3), sgm_y), sgm_y), eye(2)), eye(2)), eye(2)),  eye(2)), eye(2))
Hei += J * kron(kron(kron(kron(kron(kron(kron(eye(3), eye(2)), sgm_y), sgm_y), eye(2)), eye(2)),  eye(2)), eye(2))
Hei += J * kron(kron(kron(kron(kron(kron(kron(eye(3), eye(2)), eye(2)), sgm_y), sgm_y), eye(2)),  eye(2)), eye(2))
Hei += J * kron(kron(kron(kron(kron(kron(kron(eye(3), eye(2)), eye(2)), eye(2)), sgm_y), sgm_y),  eye(2)), eye(2))
Hei += J * kron(kron(kron(kron(kron(kron(kron(eye(3), eye(2)), eye(2)), eye(2)), eye(2)), sgm_y), sgm_y), eye(2))
Hei += J * kron(kron(kron(kron(kron(kron(kron(eye(3), eye(2)), eye(2)), eye(2)), eye(2)), eye(2)), sgm_y), sgm_y)

Hei += J * kron(kron(kron(kron(kron(kron(kron(eye(3), sgm_z), sgm_z), eye(2)), eye(2)), eye(2)),  eye(2)), eye(2))
Hei += J * kron(kron(kron(kron(kron(kron(kron(eye(3), eye(2)), sgm_z), sgm_z), eye(2)), eye(2)),  eye(2)), eye(2))
Hei += J * kron(kron(kron(kron(kron(kron(kron(eye(3), eye(2)), eye(2)), sgm_z), sgm_z), eye(2)),  eye(2)), eye(2))
Hei += J * kron(kron(kron(kron(kron(kron(kron(eye(3), eye(2)), eye(2)), eye(2)), sgm_z), sgm_z),  eye(2)), eye(2))
Hei += J * kron(kron(kron(kron(kron(kron(kron(eye(3), eye(2)), eye(2)), eye(2)), eye(2)), sgm_z), sgm_z), eye(2))
Hei += J * kron(kron(kron(kron(kron(kron(kron(eye(3), eye(2)), eye(2)), eye(2)), eye(2)), eye(2)), sgm_z), sgm_z)

Hei += J * kron(kron(kron(kron(kron(kron(kron(eye(3), sgm_x), sgm_x), eye(2)), eye(2)), eye(2)),  eye(2)), eye(2))
Hei += J * kron(kron(kron(kron(kron(kron(kron(eye(3), eye(2)), sgm_x), sgm_x), eye(2)), eye(2)),  eye(2)), eye(2))
Hei += J * kron(kron(kron(kron(kron(kron(kron(eye(3), eye(2)), eye(2)), sgm_x), sgm_x), eye(2)),  eye(2)), eye(2))
Hei += J * kron(kron(kron(kron(kron(kron(kron(eye(3), eye(2)), eye(2)), eye(2)), sgm_x), sgm_x),  eye(2)), eye(2))
Hei += J * kron(kron(kron(kron(kron(kron(kron(eye(3), eye(2)), eye(2)), eye(2)), eye(2)), sgm_x), sgm_x), eye(2))
Hei += J * kron(kron(kron(kron(kron(kron(kron(eye(3), eye(2)), eye(2)), eye(2)), eye(2)), eye(2)), sgm_x), sgm_x)

H = hyperfine + ano_iso + zeeman + Hei

w,v = LA.eigh(H)
print(w[0])
#wave-function initilization, tensor product of electronic spin and nuclear spins ===
wf_e = []
phase = 0.0
wf_e = v_e[:,0] + numpy.exp(-1j * phase * 2 * numpy.pi)* v_e[:,1]
norm = numpy.linalg.norm(wf_e)
wf_e = wf_e / norm
print(wf_e)
Mz0 = numpy.einsum('i, ij, j', wf_e, S1z, wf_e)
print('initial Mz {:10.5f}'.format(abs(Mz0)))

wf_n = []
for i in range(2**(N_bath)):
    tt = random.uniform(-1, 1) + 1j * random.uniform(-1, 1)
    wf_n.append(tt)

norm = numpy.linalg.norm(wf_n)
wf_n = wf_n / norm

wf0 = numpy.kron(wf_e, wf_n)

#=====================================================================================
arrays = []
for i in range(N_bath):
    arrays.append((0, 1))
sb = list(product(*arrays))

# reduced density matrix ============================================================
def rdm(wf, sb):
    """ reduced density matrix for spin-one from wave-function """
    dm = numpy.zeros((3,3), dtype=complex)

    N_bath = len(sb[0])
    for i in range(3):
        for j in range(3):
            for k in range(len(sb)):
                left_ind = i * 2**(N_bath)
                right_ind = j * 2**(N_bath)
                for q in range(N_bath):
                    left_ind += sb[k][q] * 2**(N_bath-q-1)
                    right_ind += sb[k][q] * 2**(N_bath-q-1)
                dm[i,j] += numpy.conj(wf[left_ind]) * wf[right_ind]
    return dm
#======================================================================================

# some parameter for echo ================
peroid = 2. * numpy.pi / (w_e[1] - w_e[0])
print('peroid {:10.5f}'.format(peroid))
#N_p = math.ceil(period / dt)
#print('number of points in one period {}'.format(N_p) )

# simulation ========================================================================
A = S1x @ S1y + S1y @ S1x
B = LA.expm(-1j * numpy.pi / 2. * A)
R = kron(B, eye(2))
for i in range(N_bath -1):
    R = kron(R, eye(2))

T_tot = 40000.
T = peroid
exp_dt = numpy.array( v @ numpy.diag( numpy.exp(-1j * dt * w ) ) @ numpy.matrix(v).getH())

while T < T_tot:
    wf = numpy.copy(wf0)
    exp_H = numpy.array( v @ numpy.diag( numpy.exp(-1j * T * w) ) @ numpy.matrix(v).getH() )
    #exp_H = LA.expm( -1j * T * H)
    wf = exp_H @ wf
    wf = R @ wf
    wf = exp_H @ wf
    dm = rdm(wf, sb)
    Mz = numpy.trace(dm@S1z).real
    ##print(numpy.linalg.norm(wf))
    s_t = 0.0
    sec_mag = []
    while s_t <= 1.2 * peroid:
        wf = exp_dt @ wf
        dm = rdm(wf, sb)
        Mz = numpy.trace(dm@S1z).real
        sec_mag.append(Mz)
        s_t += dt
    ra = (max(sec_mag) - min(sec_mag)) / (2. * abs(Mz0))
    print('{:10.5f} {:10.8f}'.format(2*T, ra), flush=True)
    #print('{:10.5f} {:10.8f}'.format(T, Mz))
    T += dt * 10

