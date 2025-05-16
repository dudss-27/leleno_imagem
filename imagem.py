### Grupo: Elisa Morais, Eduarda Birck e Pedro Bueno.

import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import scipy.fftpack as fp
import gauss

g = np.genfromtxt('incognita-fft.txt', dtype=complex)

#Transformada inversa 
F1 = scipy.fftpack.ifft2(g)
F1 = np.absolute(F1)
F1 /= np.max(F1)  #Normalização

#Salvando a imagem original
plt.imsave("original.png", F1, cmap="gray")

#Variações dos valores de sigma
sigmas = [1, 0.5, 0.1]

#Filtrando e salvando a imagem
for sigma in sigmas:
    # Filtro passa-baixas (desfoca)
    M = gauss.gaussian_baixas(g, sigma=sigma)
    f1 = g * M
    F2 = scipy.fftpack.ifft2(f1)
    F2 = np.absolute(F2)
    F2 /= np.max(F2)
    plt.imsave(f"baixas_sigma{sigma}.png", F2, cmap="gray")

    # Filtro passa-altas (marca o contorno)
    W = gauss.gaussian_altas(g, sigma=sigma)
    f2 = g * W
    F3 = scipy.fftpack.ifft2(f2)
    F3 = np.absolute(F3)
    F3 /= np.max(F3)
    plt.imsave(f"altas_sigma{sigma}.png", F3, cmap="gray")
    
    fig = plt.figure(figsize=[10, 4])
    grid = gs.GridSpec(1, 3)
    
    ax1 = fig.add_subplot(grid[:, 0])
    ax2 = fig.add_subplot(grid[:, 1])
    ax3 = fig.add_subplot(grid[:, 2])

    ax1.imshow(F1, cmap='gray')
    ax1.axis('off')
    ax1.set_title('Original')

    ax2.imshow(F2, cmap='gray')
    ax2.axis('off')
    ax2.set_title(f'Passa-baixas\nσ={sigma}')

    ax3.imshow(F3, cmap='gray')
    ax3.axis('off')
    ax3.set_title(f'Passa-altas\nσ={sigma}')

    plt.tight_layout()
    plt.show()
#Quanto menor o sigma no passa-baixas,a imagem vai ficando mais desfocada. 
#Quanto menor o sigma no passa-altas, a imagem vai ficando com mais contorno.
