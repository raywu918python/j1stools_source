class DataBuilderResult:
    def __init__(self, xtrain, xtest, ytrain, ytest, xval=None, yval=None, xtest_future_return=None):
        self.xtrain = xtrain
        self.xtest = xtest
        self.ytrain = ytrain
        self.ytest = ytest
        self.xval = xval
        self.yval = yval
        self.xtest_future_return = xtest_future_return


class TrainResult:
    def __init__(self, top, yproba):
        self.top = top
        self.yproba = yproba
