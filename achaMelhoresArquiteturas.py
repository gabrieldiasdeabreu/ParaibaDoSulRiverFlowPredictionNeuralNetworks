import pandas as pd
import glob
import numpy as np
from subprocess import call

nomePastaExperimento = 'experimentoGruposUnicosComNovoEnaSud/'
def pegaMinMesUmaExecucao(caminho):
    """Funcao para ler o csv dos Mapes de 
    uma execucao e montar a tabela 
    de escolha das melhores arquiteturas por mes

    :caminho: diretorio do arquivo
    :returns: tabela pandas

    """
    

    df = pd.read_csv(caminho+'MapesTodosMeses.csv', index_col=False, skiprows=1, names=['ordem', 'grupo', 'qtdNeuronios', 'mes', 'mape'])
    # print(df)
    listaIndMin = (df.groupby('mes')['mape']).idxmin()
    # print("melhores da execucao:" ,caminho)
    # print(df.iloc[listaIndMin])
    return caminho, df.iloc[listaIndMin]


def selecionaMelhores(listaMapes):
    """a partir da lista de Mapes com o caminho 
    seleciona os melhores mapes por mes e gera 
    uma nova tabela

    :listaMapes: TODO
    :returns: TODO

    """    

    df = pd.DataFrame()
    listaMapes = np.transpose(listaMapes)
    lista = listaMapes[1]#[pd.DataFrame(x[1]) for x in listaMapes]
    df = pd.concat(lista, ignore_index=True)
    # print(df)
    listaIndMin = (df.groupby('mes')['mape']).idxmin()
    # print("melhores da execucao:" ,caminho)
    # print(df.iloc[listaIndMin])
    df = df.iloc[listaIndMin]
    print(df)
    print((df.index.values)/12)
    listaArquivos = listaMapes[0][(df.index.values)//12]
    df.to_csv(nomePastaExperimento+'caminhoResultadosExperimento.csv')
    return listaArquivos, df 


def copiaMelhoresModelos(resultados):
    """copia os melhores modelos para a pasta melhoresModelos

    :resultados: TODO
    :returns: TODO

    """

    diretorioMelhoresModelos = nomePastaExperimento+'melhoresModelos/'
    call(['rm','-R', diretorioMelhoresModelos] )
    call(['mkdir', diretorioMelhoresModelos] )
    for caminho, parametros in zip(resultados[0], resultados[1].iterrows()):
        arquivo = caminho
        # print(parametros[1]['ordem'])
 #       grupo=[10]_neuronios=10_ordem=10_.hdf5 
        aux = parametros[1]
        arquivoSalvo= arquivo + 'grupo='+str(aux['grupo'])+'_neuronios='+str(aux['qtdNeuronios'])+'_ordem='+str(aux['ordem'])+'_'+'.hdf5'
#        grupo=[9]_neuronios=9_ordem=9__tensorboard
        arquivoSalvoTf= arquivo + 'grupo='+str(aux['grupo'])+'_neuronios='+str(aux['qtdNeuronios'])+'_ordem='+str(aux['ordem'])+'__'+'tensorboard'
#        arquivoParam= arquivo + 'modeloParam_grupo='+str(aux['grupo'])+'_neuronios='+str(aux['qtdNeuronios'])+'_ordem='+str(aux['ordem'])+'.json'
 #       arquivoPeso=arquivo+'modelPesos_grupo='+str(aux['grupo'])+'_neuronios='+str(aux['qtdNeuronios'])+'_ordem='+str(aux['ordem'])+'.h5'
        #print(arquivoPeso)
        #print(arquivoParam)
        print(arquivoSalvo)
        print(arquivoSalvoTf)
        call(['cp', arquivoSalvo, diretorioMelhoresModelos])
        call(['cp', '-r' , arquivoSalvoTf, diretorioMelhoresModelos])
#        call(['cp', arquivoPeso, diretorioMelhoresModelos])
 #       call(['cp', arquivoParam, diretorioMelhoresModelos])



caminhoRE = nomePastaExperimento +'experimentos_*/modeloEx_*/'
listaMapes = list()
caminhos = sorted(glob.glob(caminhoRE))
print(caminhos)
for caminho in caminhos:
    listaMapes.append( pegaMinMesUmaExecucao(caminho))
resultado = selecionaMelhores(listaMapes)
copiaMelhoresModelos(resultado)
# print(listaMapes)


