import src.data_loader as dataLoader
import src.preprocessing as preprocessing

import pandas as pd

def main():

    print("entro")
    dataFrame = dataLoader.prepararDatos("Advertisement.csv")

    xTrain, xTest, yTrain, yTest = dataFrame

    print(f"Xtrain:\n {xTrain}")

    mediaAge = preprocessing.promedioColumna(xTrain["Age"])
    desviacionEstandarAge = preprocessing.desviacionEstandarColumna(xTrain["Age"], mediaAge)

    mediaEstimatedSalary = preprocessing.promedioColumna(xTrain["EstimatedSalary"])
    desviacionEstandarEstimatedSalary = preprocessing.desviacionEstandarColumna(xTrain["EstimatedSalary"],mediaEstimatedSalary)

    caracteristicasTraining = pd.DataFrame()
    caracteristicasTraining["Age"] = preprocessing.normalizarDataFrame(xTrain["Age"], mediaAge, desviacionEstandarAge)
    caracteristicasTraining["EstimatedSalary"] = preprocessing.normalizarDataFrame(xTrain["EstimatedSalary"], mediaEstimatedSalary, desviacionEstandarEstimatedSalary)
    
    caracteristicasTesting = pd.DataFrame()
    caracteristicasTesting["Age"] = preprocessing.normalizarDataFrame(xTest["Age"], mediaAge, desviacionEstandarAge)
    caracteristicasTesting["EstimatedSalary"] = preprocessing.normalizarDataFrame(xTest["EstimatedSalary"],mediaEstimatedSalary, desviacionEstandarEstimatedSalary)

    objetivoTraining = yTrain
    objetivoTesting = yTest

    print(f"xtrain: \n{xTrain}, objetivoTraining:\n{objetivoTraining}")

    return


if __name__ == "__main__" :
    main()