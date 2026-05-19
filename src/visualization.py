import sys
from pathlib import Path

RAIZ = Path(__file__).parent.parent
sys.path.append(str(RAIZ))

from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def graficarBarras(resultados) :
    
    ejeX = []
    ejeY = []

    for key, info in resultados.items() :
        ejeX.append(key)
        ejeY.append(info["exactitud"])

    print(f"ejeX:\n {ejeX}\n ejeY:\n {ejeY}")

    plt.bar(ejeX, ejeY)
    plt.title("Efectividad funciones kernel")
    plt.xlabel("Tipos de kernel")
    plt.ylabel("Porcentaje de exactitud")
    plt.ylim(0, 1) 
    
    plt.savefig(RAIZ / "outputs" / "plots" / "barras.png")
    
    return

def graficarMatrices(resultado) :

    for key, info in resultado.items() :
        disp = ConfusionMatrixDisplay(info["matriz"])
        disp.plot()
        plt.title(f"Matriz de confusión del kernel {key}")

        plt.savefig( RAIZ / "outputs" / "confusion" / f"matriz_{key}.png")
        plt.clf()
            
if __file__ == "__main__" :
    pass