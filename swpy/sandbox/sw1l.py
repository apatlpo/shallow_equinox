"""

Demo program that solves the 1 layer shallow water equation in a doubly
periodic domain. The solution is initialized using a perturbed
vortex and evolved in time with a 4'th order Runge Kutta method.

Please note that this is not an optimized solver. For fast solvers, see
http://github.com/spectralDNS/spectralDNS

Future work:
- implement a n layer solver based on spectraldns software architecture
- use spectral space to compute derivates but time step in physical space

To run the code:
mpirun -n 4 python -u sw1l.py

https://mpi4py-fft.readthedocs.io/en/latest/io.html
"""
from time import time
import numpy as np
from mpi4py import MPI
from mpi4py_fft.mpifft import PFFT, Function

import matplotlib.pyplot as plt

# Set viscosity, end time and time step
#nu = 0.000625
day2sec = 86400.
T = 1. * day2sec
dt = 1.e0
g = 9.8
plot = False

# Set global size of the computational box
M = 10
N = [2**M, 2**M]
Ld = 100e3
L = np.array([2*np.pi*Ld, 2*np.pi*Ld], dtype=float)
# Needs to be (2*int)*pi in all directions (periodic) because of initialization

# Create instance of PFFT to perform parallel FFT + an instance to do FFT with padding (3/2-rule)
FFT = PFFT(MPI.COMM_WORLD, N, collapse=False)
#FFT_pad = PFFT(MPI.COMM_WORLD, N, padding=[1.5, 1.5, 1.5])
FFT_pad = FFT

# Declare variables needed to solve Navier-Stokes
U = Function(FFT, False, tensor=2)       # Velocity
U_hat = Function(FFT, tensor=2)          # Velocity transformed
H = Function(FFT, False)                 # Water height (scalar)
H_hat = Function(FFT)                    # Water height transformed
#
U_hat0 = Function(FFT, tensor=2)         # Runge-Kutta work array
U_hat1 = Function(FFT, tensor=2)         # Runge-Kutta work array
H_hat0 = Function(FFT)                   # Runge-Kutta work array
H_hat1 = Function(FFT)                   # Runge-Kutta work array

a = [1./6., 1./3., 1./3., 1./6.]         # Runge-Kutta parameter
b = [0.5, 0.5, 1.]                       # Runge-Kutta parameter
dU = Function(FFT, tensor=2)             # Right hand side of ODEs
dH = Function(FFT)                       # Right hand side of ODEs

muH = Function(FFT, tensor=2)

#curl = Function(FFT, False, tensor=2)
U_pad = Function(FFT_pad, False, tensor=2)
H_pad = Function(FFT_pad, False)
curl_pad = Function(FFT_pad, False)
UH_pad = Function(FFT_pad, False, tensor=2)

def get_local_mesh(FFT, L):
    """Returns local mesh."""
    X = np.ogrid[FFT.local_slice(False)]
    N = FFT.shape()
    for i in range(len(N)):
        X[i] = (X[i]*L[i]/N[i])
    X = [np.broadcast_to(x, FFT.local_shape(False)) for x in X]
    return X

def get_local_wavenumbermesh(FFT, L):
    """Returns local wavenumber mesh."""

    s = FFT.local_slice()
    N = FFT.shape()

    # Set wavenumbers in grid
    k = [np.fft.fftfreq(n, 1./n).astype(int) for n in N[:-1]]
    k.append(np.fft.rfftfreq(N[-1], 1./N[-1]).astype(int))
    K = [ki[si] for ki, si in zip(k, s)]
    Ks = np.meshgrid(*K, indexing='ij', sparse=True)
    Lp = 2*np.pi/L
    for i in range(2):
        Ks[i] = (Ks[i]*Lp[i]).astype(float)
    return [np.broadcast_to(k, FFT.local_shape(True)) for k in Ks]

X = get_local_mesh(FFT, L)
K = get_local_wavenumbermesh(FFT, L)
K = np.array(K).astype(float)
K2 = np.sum(K*K, 0, dtype=float)
K_over_K2 = K.astype(float) / np.where(K2 == 0, 1, K2).astype(float)

def cross(x, z):
    """Cross product z = k \times x"""
    z[0] = FFT_pad.forward(-x[1], z[0])
    z[1] = FFT_pad.forward(x[0], z[1])
    return z

def compute_curl(x, z):
    z = FFT_pad.backward(1j*(K[0]*x[1]-K[1]*x[0]), z)
    return z

def add_grad(x, z):
    z[0] += 1j*K[0]*x
    z[1] += 1j*K[1]*x
    return z

def compute_div(x, z):
    z = 1j*(K[0]*x[0]+K[1]*x[1])
    return z

def compute_rhs(rhsU, rhsH):
    H_pad[:] = FFT_pad.backward(H_hat, H_pad)
    for j in range(2):
        U_pad[j] = FFT_pad.backward(U_hat[j], U_pad[j])

    curl_pad[:] = compute_curl(U_hat, curl_pad)
    rhsU = cross(curl_pad, rhsU)
    rhsU = add_grad(-g*H_hat, rhsU)
    #rhsU -= nu*K2*U_hat

    for i in range(2):
        muH[i] = FFT_pad.forward(-U_pad[i]*H_pad, muH[i]) # vector, spectral space
    #UH_pad[0] = U_pad[0]*H_pad
    #UH_pad[1] = U_pad[1]*H_pad
    #muH[:] = FFT_pad.forward(-UH_pad, muH) # vector, spectral space
    # for some reason does not want to broadcast this
    #muH[:] = FFT_pad.forward(-U_pad*H_pad[np.newaxis,:,:], muH) # vector, spectral space
    rhsH = compute_div(muH, rhsH)

    return rhsU, rhsH

# Initialize with a bump of sea level
Lv = 10e3
ev = 2.
H[:] = 4000. + 1.*np.exp(-(X[0]-Ld/2.)**2/Lv**2 -(X[1]-Ld/2.)**2/(ev*Lv)**2)
U[0] = 0
U[1] = 0
#
H_hat = FFT.forward(H, H_hat)
for i in range(2):
    U_hat[i] = FFT.forward(U[i], U_hat[i])

# Integrate using a 4th order Rung-Kutta method
t = 0.0
tstep = 0
t0 = time()
while t < T-1e-8:
    t += dt
    tstep += 1
    U_hat1[:] = U_hat0[:] = U_hat
    H_hat1[:] = H_hat0[:] = H_hat
    for rk in range(4):
        dU, dH = compute_rhs(dU, dH)
        if rk < 3:
            H_hat = H_hat0 + b[rk]*dt*dH
            U_hat[:] = U_hat0 + b[rk]*dt*dU
        H_hat1[:] += a[rk]*dt*dH
        U_hat1[:] += a[rk]*dt*dU
    H_hat[:] = H_hat1[:]
    U_hat[:] = U_hat1[:]

    H = FFT.backward(H_hat, H)
    for i in range(2):
        U[i] = FFT.backward(U_hat[i], U[i])
    k = MPI.COMM_WORLD.reduce(np.sum(U[0]*U[0])/N[0]/N[1]/2)
    if MPI.COMM_WORLD.Get_rank() == 0:
        print("Energy = {}".format(k))
        print("t = {}".format(t))
    if plot and np.mod(tstep,100)==0:
        plt.figure()
        plt.imshow(H-4000.)
        plt.colorbar()
        plt.show()

## Transform result to real physical space
#for i in range(3):
    #U[i] = FFT.backward(U_hat[i], U[i])

# Check energy
#k = MPI.COMM_WORLD.reduce(np.sum(U*U)/N[0]/N[1]/N[2]/2)
#if MPI.COMM_WORLD.Get_rank() == 0:
#    print('Time = {}'.format(time()-t0))
#    assert round(float(k) - 0.124953117517, 7) == 0
