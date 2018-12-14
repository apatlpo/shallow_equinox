# mpirun -np 2 python pfft_example.py

import numpy as np
from mpi4py import MPI
from mpi4py_fft.mpifft import PFFT, Function

N = np.array([128, 128, 128], dtype=int)
fft = PFFT(MPI.COMM_WORLD, N, axes=(0, 1, 2), dtype=np.float, slab=True)

u = Function(fft, False)
u[:] = np.random.random(u.shape).astype(u.dtype)
u_hat = fft.forward(u)
uj = np.zeros_like(u)
uj = fft.backward(u_hat, uj)
assert np.allclose(uj, u)
print(MPI.COMM_WORLD.Get_rank(), u.shape)
