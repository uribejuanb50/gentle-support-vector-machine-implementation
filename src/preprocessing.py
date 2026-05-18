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

def estadisticasColumna(columna) : 
    media = promedioColumna(columna)
    return media, desviacionEstandarColumna(columna, media)

def normalizarDataFrame(dataFrame) :

    dfCopia = dataFrame.copy()

    if(isinstance(dfCopia, pd.Series)):
        media, desviacionEstandar = estadisticasColumna(dfCopia)

        dfCopia = dfCopia.apply(lambda x : (x - media) / desviacionEstandar)

    else:
        for columna in dfCopia.columns :
            media, desviacionEstandar = estadisticasColumna(dfCopia[columna])
            dfCopia[columna] = dfCopia[columna].apply(lambda x : (x - media)/desviacionEstandar)

    return dfCopia

if __name__ == "__main__" :

    pass
