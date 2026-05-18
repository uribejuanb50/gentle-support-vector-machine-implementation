import sys
from pathlib import Path

RAIZ = Path(__file__).parent.parent
sys.path.append(str(RAIZ))

from sklearn.svm import SVC


def entrenarModelo(kernel, xTrain, yTrain) :
    model = SVC(kernel = kernel)
    model.fit(xTrain, yTrain)
    return model
