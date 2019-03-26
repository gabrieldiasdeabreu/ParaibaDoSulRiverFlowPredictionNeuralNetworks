import numpy as np
import pandas as pd
class SerieTemporal:
    Serie = None
    serieDesnormalizada = None
    menorElemento = 0
    maiorElemento = 0

    def __init__(self, nomeInstancia):
        '''lê um arquivo do tipo csv'''
        self.Serie = pd.read_csv(nomeInstancia)
        self.Serie = np.array(self.Serie['Vazao']).ravel()
        # self.Serie = np.loadtxt(nomeInstancia, 'float', delimiter=',')
        print(self.Serie)
        self.serieDesnormalizada = self.Serie
        serieLinear = self.Serie.ravel()
        self.menorElemento = min(serieLinear)
        self.maiorElemento = max(serieLinear)
        self.normalizaSerie()

    def normalizaSerie(self):
        self.Serie = (self.Serie - self.menorElemento)/(self.maiorElemento-self.menorElemento)

    def desnormalizaElemento(self, elemento):
        return elemento*(self.maiorElemento-self.menorElemento) + self.menorElemento
