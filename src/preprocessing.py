import sys
from pathlib import Path

RAIZ = Path(__file__).parent.parent
sys.path.append(str(RAIZ))

import pandas as pd
import math

def promedioColumna(columna) :
    return sum(columna) / len(columna)

def desviacionEstandarColumna(columna, media) :
    sumador = 0

    for digito in columna :
        sumador += math.pow(digito - media, 2)

    return sumador / len(columna)

def normalizarDataFrame(columna, media, desviacionEstandar) :

    dfCopia = columna.copy()

    dfCopia = dfCopia.apply(lambda x : (x - media) / desviacionEstandar)

    return dfCopia

if __name__ == "__main__" :

    pass
