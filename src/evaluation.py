import sys
from pathlib import Path

RAIZ = Path(__file__).parent.parent
sys.path.append(str(RAIZ))

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

def evaluarModelo(modelo, yTest, yPred):
    matrizConfusion = confusion_matrix(yTest, yPred)
    exactitud = accuracy_score(yTest,yPred)

    return exactitud, matrizConfusion

def compararModelos(resultados):
    return