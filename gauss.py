import numpy as np
import scipy.fftpack as fp

#Função Gaussiana para desfoque de imagem

def gaussian_baixas(fc, sigma =1, vmax = .01):
    '''
    Permite que passem apenas baixas frequências
    '''

    Nu, Nv = fc.shape
    u = Nu * np.linspace(-vmax, vmax, Nu)
    v = Nv * np.linspace(-vmax, vmax, Nv)
    U, V = np.meshgrid(v, u)

    sigma2 = sigma**2
    G = (np.exp(-(U*U + V*V) /(2. *sigma2)))/(2*np.pi*sigma2)
    G = fp.fftshift(G)

    return G

def gaussian_altas(fc, sigma =1, vmax = .01):
    '''
    Permite que passem apenas altas frequências
    '''


    Nu, Nv = fc.shape
    u = Nu * np.linspace(-vmax, vmax, Nu)
    v = Nv * np.linspace(-vmax, vmax, Nv)
    U, V = np.meshgrid(v, u)

    sigma2 = sigma**2
    G = 1 - (np.exp(-(U*U + V*V) /(2. *sigma2)))/(2*np.pi*sigma2)
    G = fp.fftshift(G)

    return G