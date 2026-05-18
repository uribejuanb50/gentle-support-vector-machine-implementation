from pathlib import Path
import sys

RAIZ  = Path(__file__).parent.parent
sys.path.append(str(RAIZ))

import pandas as pd
import numpy as np
import math

from sklearn.model_selection import train_test_split

def prepararDatos(nombreArchivo) :

    rutaArchivo = RAIZ / "data" / nombreArchivo

    dataFrame = pd.read_csv(rutaArchivo)

    caracteristicas = ["Age", "EstimatedSalary"]
    dfCaracteristicas = dataFrame[caracteristicas]

    dfObjetivo = dataFrame["Purchased"]

    return train_test_split(dfCaracteristicas, dfObjetivo, random_state = 42, test_size = 0.25)

#devuelve el corte que se va a elegir como testeo y el resto como training 

if __name__ == "__main__" :

    dataFrame = prepararDatos("Advertisement.csv")
    Xtrain, Xtest, yTrain, ytest = dataFrame
    print(f"entrenamientoX: {Xtrain},\n testX {Xtest} \n trainingY : {yTrain} \n TesteoY: {ytest}")

    pass