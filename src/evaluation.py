import sys
from pathlib import Path

RAIZ = Path(__file__).parent.parent
sys.path.append(str(RAIZ))

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

def evaluarModelo(modelo, yTest, yPred):
    matrizConfusion = confusion_matrix(yTest, yPred)
    exactitud = accuracy_score(yTest,yPred)

    return { "exactitud" : exactitud, "matriz" : matrizConfusion}

def compararModelos(resultados):

    mejorExactitud = 0
    mejorKernel = ""

    for kernel, info in resultados.items() :
        exactitud = info["exactitud"]
        mc = info["matriz"]

        if mejorExactitud < exactitud :
            mejorExactitud = exactitud
            mejorKernel = kernel

        print(f"El modelo {kernel} tiene una exactitud del {exactitud * 100}%")

    print(f"El mejor kernel fue el {mejorKernel}, con una exactitud del {mejorExactitud * 100}%")

    return