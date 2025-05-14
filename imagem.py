import numpy as np
import scipy
import matplotlib.pyplot as plt
import scipy.fftpack as fp
 

g = np.genfromtxt('incognita-fft.txt', dtype = complex)
F = scipy.fftpack.ifft2(g)
F = np.absolute(F)

plt.figure ()
plt.imshow(F, cmap='twilight')
plt.show()

############

def gaussian(F, sigma =1, vmax = .5):
    Nu, Nv = F.shape
    u = Nu * np.linspace(-vmax, vmax, Nu)
    v = Nv * np.linspace(-vmax, vmax, Nv)
    U, V = np.meshgrid(v, u)

    sigma2 = sigma**2
    G = np.exp(-(U*U + V*V) /2. /sigma2)
    G = fp.fftshift(G)

    return G/sigma2

M = gaussian.gaussian(F, sigma=1)
f_inversa = F * M
plt.figure()
plt.plot(f_inversa)
plt.show()