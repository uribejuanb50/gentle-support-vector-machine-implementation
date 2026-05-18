import src.data_loader as dataLoader
import src.preprocessing as preprocessing

def main():

    print("entro")
    dataFrame = dataLoader.prepararDatos("Advertisement.csv")

    xTrain, xTest, yTrain, yTest = dataFrame

    print(f"Xtrain:\n {xTrain}")

    caracteristicasTraining = preprocessing.normalizarDataFrame(xTrain)[:]
    caracteristicasTesting = preprocessing.normalizarDataFrame(xTest)[:]

    objetivoTraining = preprocessing.normalizarDataFrame(yTrain)[:]
    objetivoTesting = preprocessing.normalizarDataFrame(yTest)[:]

    print(f"xtrain: \n{xTrain}, objetivoTraining:\n{objetivoTraining}")
    
    return


if __name__ == "__main__" :
    main()