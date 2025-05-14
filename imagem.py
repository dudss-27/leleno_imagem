import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import scipy.fftpack as fp
import gauss
 

g = np.genfromtxt('incognita-fft.txt', dtype = complex)

F1 = scipy.fftpack.ifft2(g)
F1 = np.absolute(F1)


M = gauss.gaussian_baixas(g, sigma=1) #Alterar valores de sigma: 1; .5; .1
f1 = g * M
F2 = scipy.fftpack.ifft2(f1)
F2 = np.absolute(F2)


W = gauss.gaussian_altas(g, sigma=1) #Alterar valores de sigma: 1; .5; .1
f2 = g * W
F3 = scipy.fftpack.ifft2(f2)
F3 = np.absolute(F3)


##Plot
fig = plt.figure(figsize=[10,4])
grid = gs.GridSpec(1,3)

ax1 = fig.add_subplot(grid[:,0])
ax2 = fig.add_subplot(grid[:,1])
ax3 = fig.add_subplot(grid[:,2])



#imagem original
ax1.imshow(F1, cmap='gray')
ax1.axis('off')
ax1.set_title('Imagem sem filtros')

#Filtro passa-baixas

ax2.imshow(F2, cmap='gray')
ax2.axis('off')
ax2.set_title('Filtro passa-baixas')

#filtro passa-altas

ax3.imshow(F3, cmap= 'gray')
ax3.axis('off')
ax3.set_title('Filtro passa-altas')


plt.tight_layout() #para distribuir melhor o espaço
plt.show()