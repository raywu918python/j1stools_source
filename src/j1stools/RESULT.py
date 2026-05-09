class DataBuilderResult:
    def __init__(self, xtrain, xtest, ytrain, ytest, xval=None, yval=None):
        self.xtrain = xtrain
        self.xtest = xtest
        self.ytrain = ytrain
        self.ytest = ytest
        self.xval = xval
        self.yval = yval
