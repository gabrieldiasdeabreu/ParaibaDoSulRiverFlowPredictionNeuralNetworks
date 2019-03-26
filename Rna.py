#classe para encapsular as tarefas da rede neural
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Dropout
from keras import optimizers
import keras
from EntradaRna import EntradaRna

class Rna:
    entradaRna = None
    mapeTreino = None
    rna = None


    def __init__(self, entradaRna):
        self.entradaRna = entradaRna

    def ArquiteturaRna(self, listaCamadas, tamanhoVal,  verbose, epocas, ondeSalvarModelos, optimizer='adam'):
        '''encapsula o metodo de criacao da rede neural e sua Arquitetura
                utilizando o keras recebendo uma lista de camadas e epocas e o otimizador'''
        self.rna = Sequential()
        self.rna.add(
            Dense(units=listaCamadas[0][0], input_dim=self.entradaRna.ordemEntrada, activation=listaCamadas[0][1])
        )
        self.rna.add(Dropout(0.5))
        listaCamadas.pop(0)
        for i in listaCamadas:
            self.rna.add(Dense(units=i[0], activation=i[1]))

        self.rna.compile(optimizer=optimizer, loss='mean_absolute_error',
                         metrics=['mean_absolute_percentage_error'])

        # salva o modelo
        salvaModelo = keras.callbacks.ModelCheckpoint(ondeSalvarModelos+'.hdf5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)

        #salva tensorBoard
        # tensorBoard = keras.callbacks.TensorBoard(log_dir=ondeSalvarModelos+'_tensorboard', histogram_freq=1, write_graph=True, write_grads=True)

        # garante que para dpois de 5 nao melhoras
        earlyStopping=keras.callbacks.EarlyStopping(monitor='val_loss', patience=100, verbose=0, mode='auto')
        tamanhoY = len(self.entradaRna.yTreino)
        hist = self.rna.fit(
            self.entradaRna.xTreino, self.entradaRna.yTreino, tamanhoY,
            validation_split=(tamanhoVal/tamanhoY), verbose=verbose,
            epochs=epocas, callbacks=[salvaModelo]
        )

        self.rna = keras.models.load_model(ondeSalvarModelos+'.hdf5')
        return hist.history['val_loss'], hist.history['mean_absolute_percentage_error']


    def previsaoSerie(self, tamanhoValidacao, anoInicial = 2):
        '''retorna as previsoes atingidas'''
        indVal = len(self.entradaRna.yTreino) - tamanhoValidacao
        treino = self.rna.evaluate(self.entradaRna.xTreino[:indVal], self.entradaRna.yTreino[:indVal], verbose=0)[1]
        val = self.rna.evaluate(self.entradaRna.xTreino[indVal:], self.entradaRna.yTreino[indVal:], verbose=0)[1]
        teste = self.rna.evaluate(self.entradaRna.xTeste, self.entradaRna.yTeste, verbose=0)[1]
        return treino, val, teste


    def salvaRede(self, destino, nomeModelo=''):
        # serialize model to JSON
        model_json = self.rna.to_json()
        with open(destino+"modeloParam_" + str(nomeModelo) + ".json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.rna.save_weights(destino+"modelPesos_" + str(nomeModelo) + ".h5")
        # print("Saved model to disk")
