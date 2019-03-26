#classe para preparar a rede neural, tanto a sua alimentacao quando saida
import numpy as np
from SerieTemporal import SerieTemporal
import pandas as pd
import sklearn.preprocessing

class EntradaRna:
    Serie = None
    serieTemporal = None
    xTreino = None
    yTreino = None
    xTeste = None
    yTeste = None
    ordemEntrada = 0
    ehRecorrente = False
    # standartizador = None
    normalizador = None

    def __init__(self, nomeInstancia):
        #inicia classe de Entranda que prepara a alimentacao da rede neural
        self.Serie = pd.read_csv(nomeInstancia)
        self.Serie = np.array(self.Serie['Vazao']).ravel()

        self.normalizador = sklearn.preprocessing.MinMaxScaler((0.1, 0.9)) # seleção para nao deixar a escala saturar
        self.standartizador = sklearn.preprocessing.StandardScaler()

        # self.Serie = serieTemporal.Serie
        # self.serieTemporal = serieTemporal

        # self.xTreino = list()
        # self.yTreino = list()
        # self.xTeste = list()
        # self.yTeste = list()
        # self.preparaTreino(anoFinalTreino, ordemEntrada)
        # self.preparaTeste(anoFinalTreino, ordemEntrada)


    def configura(self, ordem, ehRecorrente):
        self.ordemEntrada = ordem
        self.ehRecorrente = ehRecorrente

    def preparaTreinoComListaMesesEspecificos(self, anoFinalTreino, ordem, ehRecorrente):
        self.configura(ordem, ehRecorrente)
        #recebe ate que ano e os meses para gerar a alimentacao da rede neural
        # print('Anos Totais :', len(self.Serie))
        # print('Anos no Treino :', len(self.Serie[:anoFinalTreino][:]))
        # xAux = list(self.Serie[:anoFinalTreino, mesEspecifico-1].ravel())
        self.xTreino = list()
        self.yTreino = list()
        xAux = list()
        for anos in self.Serie[:anoFinalTreino]:
            # for mes in mesEspecifico:
            xAux.append(anos)

        xAux = np.reshape(xAux, (-1, 1))
        # standart
        # print(xAux)
        # self.standartizador.fit(xAux)
        # xAux = self.standartizador.transform(xAux)

        #normaliza aqui
        self.normalizador.fit(xAux)
        xAux = self.normalizador.transform(xAux)

        xAux = np.ravel(xAux)
        # return 2
        # for i in xAux:
        #     print(self.serieTemporal.desnormalizaElemento(i))
        while len(xAux) > self.ordemEntrada:
            self.xTreino.append(np.array(xAux[:self.ordemEntrada]))
            self.yTreino.append(xAux[self.ordemEntrada])
            # print(xAux[:ordemEntrada], xAux[ordemEntrada])
            xAux = xAux[1:]
            # xAux = xAux[1:] .pop(0)
        self.xTreino = np.array(self.xTreino)
        self.yTreino = np.array(self.yTreino)
        # print(self.xTreino)
        if self.ehRecorrente:            
            self.xTreino = self.xTreino.reshape(self.xTreino.shape[0], self.xTreino.shape[1], 1)
        # print(self.xTreino)


    def preparaTesteComMesEspecifico(self, anoFinalTreino, mesEspecifico ):
        # prepara o treino para o mes e a partir do ano
        # print('Anos no Teste :', len(self.Serie[anoFinalTreino:]))
        self.xTeste = list()
        self.yTeste = list()
        xAux = list(self.Serie[anoFinalTreino - self.ordemEntrada:].ravel())

        # standartiza
        xAux = np.reshape(xAux, (-1, 1))
        # xAux = self.standartizador.transform(xAux)

        # normaliza aqui
        xAux = self.normalizador.transform(xAux)

        xAux = np.ravel(xAux)

        while len(xAux) > self.ordemEntrada:
            self.xTeste.append(np.array(xAux[:self.ordemEntrada]))
                               # .reshape(1,self.ordemEntrada))
            self.yTeste.append(xAux[self.ordemEntrada])
            # print(xAux[:ordemEntrada], xAux[ordemEntrada])
            # xAux.pop(0)
            xAux = xAux[1:]


        self.xTeste = np.array(self.xTeste)
        self.yTeste = np.array(self.yTeste)
        # print(self.xTeste, self.yTeste)
        if self.ehRecorrente:
            self.xTeste = self.xTeste.reshape(self.xTeste.shape[0], self.xTeste.shape[1], 1)   
            # print('depoiss')
            # print(self.xTeste, self.yTeste)

## cria Logs

    def salvaTreino(self):
        #gera um buffer com o treino realizado pela rna
        buffer = ''
        # with open('treinoDaRNA.csv', 'a') as arq:
        for i in range(self.ordemEntrada):
            buffer += 'neuronioEnt_'+str(i+1)+','
        buffer += 'resposta'+'\n'

        for x, y  in zip(self.xTreino, self.yTreino):
            for num in x:
                buffer +=str(self.serieTemporal.desnormalizaElemento(num))+','
            buffer += str(self.serieTemporal.desnormalizaElemento(y))+'\n'
        return buffer

    def escrevePrevisoes(self, previsoes, numNeuroniosCamadaOculta = ' '):
        #gera um buffer com as previsoes realizado pela rna
        listaMapes = self.calculaMape(previsoes, self.yTeste)
        # with open('testeDaRNA.csv', 'a') as arq:
        buffer = ''
        buffer += 'numNeuroniosCamadaOculta='+ str(numNeuroniosCamadaOculta)+','+'ordemRede='+ str(self.ordemEntrada)+','+'melhorMape='+ str(listaMapes[1])+','+'MapeMedio='+str(listaMapes[2])+','+'desvioPadraoMapeMedio='+ str(listaMapes[3])+'\n'
        for i in range(self.ordemEntrada):
            buffer += 'neuronioEnt_'+str(i+1)+','
        buffer += 'previsao'+','+'real'+','+'Mape'+'\n'

        for x, previsao, y, mape in zip(self.xTeste, previsoes, self.yTeste, listaMapes[0]):
            for xaux in x:
                buffer += str(self.serieTemporal.desnormalizaElemento(xaux))+','
            buffer+=str(self.serieTemporal.desnormalizaElemento(previsao[0]))+','+str(self.serieTemporal.desnormalizaElemento(y))+','+ str(mape) +'\n'
        return buffer



    def calculaMape(self, previsoes, reais):
        #retorna o mape das previsoes com minimo media e desvio padrao
        listaMapes = [np.abs((self.serieTemporal.desnormalizaElemento(previsoes[i][0]) - self.serieTemporal.desnormalizaElemento(reais[i]))/self.serieTemporal.desnormalizaElemento(reais[i]))*100 for i in range(len(previsoes))]
        return listaMapes, min(listaMapes), np.mean(listaMapes), np.std(listaMapes)


#arrumar depois
#     def preparaTreino(self, anoFinalTreino):
#         #funcao com defeito nao usar
#         print('Anos Totais :', len(self.Serie))
#         print('Anos no Treino :', len(self.Serie[:anoFinalTreino][:]))
#         xAux = list(self.Serie[:anoFinalTreino][:].ravel())
#         while len(xAux) > self.ordemEntrada:
#             self.xTreino.append(np.array(xAux[:self.ordemEntrada]))
#             self.yTreino.append(xAux[self.ordemEntrada])
#             # print(xAux[:ordemEntrada], xAux[ordemEntrada])
#             xAux.pop(0)
#         self.xTreino = np.ndarray(xTreino)
#         print(self.xTreino)
# #arrumar depois
#     def preparaTeste(self, anoFinalTreino):
#         #com defeito nao usar
#         print('Anos no Teste :', len(self.Serie[anoFinalTreino:][:]))
#         xAux = list(self.Serie[anoFinalTreino:][:].ravel())
#         while len(xAux) > self.ordemEntrada:
#             self.xTreino.append(xAux[:self.ordemEntrada])
#             self.yTreino.append(xAux[self.ordemEntrada])
#             print(xAux[:self.ordemEntrada], xAux[self.ordemEntrada])
#             xAux.pop(0)

# def preparaTesteComListaMesesEspecificos(self, anoFinalTreino, mesEspecifico):
#     # print('Anos no Teste :', len(self.Serie[anoFinalTreino:]))
#     # xAux = list(self.Serie[anoFinalTreino - self.ordemEntrada:, mesEspecifico-1].ravel())
#     xAux = list()
#     for anos in self.Serie[anoFinalTreino-self.ordemEntrada:]:
#         for mes in mesEspecifico:
#              xAux.append(anos[mes-1])
#     while len(xAux) > self.ordemEntrada:
#         self.xTeste.append(np.array(xAux[:self.ordemEntrada]))
#                            # .reshape(1,self.ordemEntrada))
#         self.yTeste.append(xAux[self.ordemEntrada])
#         # print(xAux[:ordemEntrada], xAux[ordemEntrada])
#         xAux.pop(0)
#     self.xTeste = np.array(self.xTeste)
#     self.yTeste = np.array(self.yTeste)
#     print(self.serieTemporal.desnormalizaElemento(self.xTeste))
#     print(self.serieTemporal.desnormalizaElemento(self.yTeste))
