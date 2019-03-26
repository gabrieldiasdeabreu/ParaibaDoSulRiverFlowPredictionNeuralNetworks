from EntradaRna import EntradaRna
from SerieTemporal import SerieTemporal
from keras.models import Sequential
from keras.layers import Dense, Activation, SimpleRNN, Dropout, LSTM
from keras import optimizers
import numpy as np
# from Rna import Rna

import sys
from keras import backend as K
# import numpy as np
import gc
serieTemporal = SerieTemporal('Instancias/2345067_58060000.csv')
anoFinal = 75-11
numEpocas = 100
# ordemEntrada = 12
ordensEntrada = 5#37
    # mes = int(sys.argv[1])#outubro
grupo = [1,2,3,4,5, 6,7, 8,9,10,11,12]

def redeElman(x_treino, y_treino):
    rna = Sequential()
    rna.add(LSTM(units=100,  input_shape=(None, 1)))
    # rna.add(Dropout(0.4))
    # rna.add(LSTM(units=200 ,   return_sequences=True))
    # rna.add(Dropout(0.4))
    # rna.add(LSTM(units=100 ,   return_sequences=True))
    # rna.add(Dropout(0.4))
    # rna.add(LSTM(units=50 ,   return_sequences=True))
    # rna.add(Dropout(0.4))
    # rna.add(LSTM(units=25))
    rna.add(Dropout(0.4))
    # rna.add(Dropout(0.4))
    # rna.add(Dropout(0.4))
    # print(x_treino)
    # x_treino = x_treino.reshape(1, 900, 1)
    # rna.add(Dense(units=50, input_dim=29, activation='relu'))
    # rna.add(Dense(units=100, input_dim=29, activation='relu'))
    # rna.add(Dense(units=10, activation='linear'))
    # rna.add(Dense(units=10, input_dim=29, activation='linear'))
    # # rna.add(Dense(units=60, input_dim=7, activation='RELU'))
    # rna.add(Dense(units=320, activation='relu'))
    # rna.add(Dense(units=320, activation='relu'))
    rna.add(Dense(units=1, activation='linear'))
    # print(x_treino)
    rna.compile(optimizer='adam',
                loss='mean_squared_error',
                metrics=['mean_absolute_percentage_error'])
    # print(y_treino[1])
    print('erros do treinamento:')
    print(rna.fit(x_treino, y_treino, verbose=0, epochs=10 ).history)
    return rna

def preve(rna, entrada):
    mapeMedioMes = [0]*12
    for i in range(1, 13):
        entrada.preparaTesteComMesEspecifico(anoFinal,i)
        x_teste = entrada.xTeste
        x_teste = entrada.xTeste.reshape(x_teste.shape[0], x_teste.shape[1], 1)
        y_teste = entrada.yTeste
        mapeMedioMes[i-1] = rna.evaluate(x_teste, y_teste, verbose=0)
    print('resultados::')
    print(mapeMedioMes)

entrada = EntradaRna(serieTemporal, anoFinal, 5)
entrada.preparaTreinoComListaMesesEspecificos(anoFinal, grupo)

# x_treino = serieTemporal.Serie[:,:anoFinal].ravel()#
x_treino = entrada.xTreino
x_treino = entrada.xTreino.reshape(x_treino.shape[0], x_treino.shape[1], 1)
# print(x_treino)
y_treino = entrada.yTreino
preve(redeElman(x_treino,  y_treino), entrada)
