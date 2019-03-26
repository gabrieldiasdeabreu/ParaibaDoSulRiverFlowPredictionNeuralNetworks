#!/usr/bin/env bash
#Script para chamar todos os experimentos e guarda-los em uma pasta especifica
#exigindo a saida de um arquivo com nome experimentos e o comando 7z instalado
numeroExecucoes=15
pastaDestino='experimentoGruposUnicosComNovoEnaSud'
comando="python facaMeuTrabalhoTodosMesesComRnasAgrupadas.py"
#
rm -R $pastaDestino
mkdir $pastaDestino
rm $pastaDestino.7z
for i in {0..$numeroExecucoes}
do
#    echo execucaoSh => $i
    nice -n 20 $comando
    mv experimentos $pastaDestino/experimentos_$i
#    mv modelo $pastaDestino/modelo_EX=$i
done

echo "Fazendo backup do codigo utilizado"
cp *.py pastaDestino/
cp *.sh pastaDestino/
echo "terminando, vou comprimir"
7z a $pastaDestino.7z $pastaDestino
echo "tudo pronto :-))"
