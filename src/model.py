from sklearn.svm import SVC

def entrenarModelo(kernel, xTrain, yTrain) :
    model = SVC(kernel = kernel)
    model.fit(xTrain, yTrain)
    return model
