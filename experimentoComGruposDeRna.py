#experimentos com rede neural alimentada por grupos separados e redes diferentes
from EntradaRna import EntradaRna
from SerieTemporal import SerieTemporal
from Rna import Rna
from RnaLstm import RnaLstm

import sys
from keras import backend as K
import numpy as np
from keras.layers import LSTM, GRU, SimpleRNN, Dense
# import numpy as np
import gc
# serieTemporal = SerieTemporal('Instancias/2345067_58060000.csv')
caminhoInstancia = 'Instancias/2345067_58060000.csv'
entrada = EntradaRna(caminhoInstancia)
anoFinal = len(entrada.Serie) - 5808
tamanhoVal = 5903
numEpocas = 1000
ordensEntrada = range(1, 37)#37 // random choice agr
gruposDeMeses = [[1]]
# buffer = ['']*12
# bufferTreino = ['']*len(gruposDeMeses)
ehRecorrente = True
# pastaDestinoModelos = 'experimentos/modeloEx_'+sys.argv[1]
pastaDestinoModelos = 'experimentos/modeloEx_'+'0'
cabecalho='ordem,grupo,qtdNeuronios,mes,mapeTreino,mapeVal,mapeTeste,anosTreino,tamanhoValidacao,numEpocas,EhRecorrente,instancia\n'
with open(pastaDestinoModelos + '/MapesTodosMeses.csv', 'a') as arq:
    arq.write(cabecalho)
#TODO: arrumar sobre escrita de modelos com mesmo nome
#TODO: arrumar minibatchs
#TODO: arrumar troca de ordens
#TODO: começar a implementar seleçao genetica
#TODO: gerar amostra para comparar com o genetico


for _ in ordensEntrada:
    ordem = np.random.choice(range(1, 48))
    print('ordem:', ordem)

    for grupo in gruposDeMeses:
        print(grupo)
        entrada.preparaTreinoComListaMesesEspecificos(anoFinal, ordem, ehRecorrente)
        # bufferTreino[gruposDeMeses.index(grupo)] += entrada.salvaTreino()
        for _ in range(1, 51):#51
            qtdNeuronios = np.random.choice(range(1, 50))
            print('numNeuronios: ', qtdNeuronios)
            if ehRecorrente:
                rna = RnaLstm(entrada)
            else:
                rna = Rna(entrada)
            listaCamadas = [(qtdNeuronios, 'relu'), (int(np.ceil(qtdNeuronios/2))+1, 'relu'), (1, 'linear')]
            rnaArquitetada = rna.ArquiteturaRna(listaCamadas, verbose=0, tamanhoVal=tamanhoVal, epocas=numEpocas,
                                                optimizer='adam',
                                                ondeSalvarModelos=pastaDestinoModelos+'/'+'grupo='+str(grupo)+'_'+
                                                                  'neuronios='+str(qtdNeuronios)+'_'+'ordem='+str(ordem)
                                                )
            # rna.salvaRede(pastaDestinoModelos+'/',
                          # 'grupo='+str(grupo)+'_'+'neuronios='+str(qtdNeuronios)+'_'+'ordem='+str(ordem))
            for mes in grupo:
                rna.entradaRna.preparaTesteComMesEspecifico(anoFinal, mes)
                mapePrevistoTreino, mapePrevistoVal, mapePrevistoTeste= rna.previsaoSerie(tamanhoVal, anoInicial=anoFinal)
                linha = str(ordem)+','+\
                        str(grupo)+','+\
                        str(qtdNeuronios)+','+\
                        str(mes)+','+\
                        str(mapePrevistoTreino)+','+\
                        str(mapePrevistoVal)+','+\
                        str(mapePrevistoTeste)+ ','+\
                        str(anoFinal)+','+ \
                        str(tamanhoVal) + ',' + \
                        str(numEpocas) + ',' + \
                        str(ehRecorrente) + ',' + \
                        caminhoInstancia + ',' + '\n'
                # buffer += linha
                print(linha)
                with open(pastaDestinoModelos + '/MapesTodosMeses.csv', 'a') as arq:
                    arq.write(linha)
                # buffer[mes-1] += entrada.escrevePrevisoes(previsoes, qtdNeuronios)
            K.clear_session()


# with open(pastaDestinoModelos+'/MapesTodosMeses.csv', 'w') as arq:
#     arq.write(buffer)
