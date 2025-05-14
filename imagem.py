import numpy as np
import scipy
import matplotlib.pyplot as plt
import scipy.fftpack as fp
 

g = np.genfromtxt('incognita-fft.txt', dtype = complex)

F = scipy.fftpack.ifft2(g)
F = np.absolute(F)

plt.figure ()
plt.imshow(F, cmap='gray')


############

def gaussian(fc, sigma =1, vmax = .01):
    Nu, Nv = fc.shape
    u = Nu * np.linspace(-vmax, vmax, Nu)
    v = Nv * np.linspace(-vmax, vmax, Nv)
    U, V = np.meshgrid(v, u)

    sigma2 = sigma**2
    G = (np.exp(-(U*U + V*V) /(2. *sigma2)))/(2*np.pi*sigma2)
    G = fp.fftshift(G)

    return G


M = gaussian(g, sigma=.1)
f = g * M
F = scipy.fftpack.ifft2(f)
F = np.absolute(F)


plt.figure()
plt.imshow(F, cmap= 'gray')
plt.show()