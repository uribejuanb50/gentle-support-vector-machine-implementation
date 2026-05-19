import src.data_loader as dataLoader
import src.preprocessing as preprocessing
import src.model as model
import src.evaluation as evaluation
import src.visualization as visualization

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

    modeloLinear = model.entrenarModelo("linear", caracteristicasTraining, objetivoTraining)
    prediccionLinear = modeloLinear.predict(caracteristicasTesting)

    modeloPoly = model.entrenarModelo("poly", caracteristicasTraining, objetivoTraining)
    prediccionPoly = modeloPoly.predict(caracteristicasTesting)

    modeloRBF = model.entrenarModelo("rbf", caracteristicasTraining, objetivoTraining)
    prediccionRBF = modeloRBF.predict(caracteristicasTesting)

    resultados = {}
    resultados["linear"] = evaluation.evaluarModelo(modeloLinear, objetivoTesting, prediccionLinear)
    resultados["poly"] = evaluation.evaluarModelo(modeloPoly, objetivoTesting, prediccionPoly)
    resultados["rbf"] = evaluation.evaluarModelo(modeloRBF, objetivoTesting, prediccionRBF)



    print(f"predccion:\n{resultados}  ")

    evaluation.compararModelos(resultados)
    visualization.graficarBarras(resultados)
    visualization.graficarMatrices(resultados)

    return


if __name__ == "__main__" :
    main()