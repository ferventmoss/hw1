import week1
import numpy
from sklearn import svm
import scipy.optimize
from math import exp
from math import log

data = week1.data 
styles = {}
cntIdx = 0
revIdx = 1

def getStylesStat(datum, sty):
  key = datum['beer/style']
  if key in sty: #and datum.has_key('review/taste'):
      sty[key][cntIdx] += 1;
      sty[key][revIdx] += datum['review/taste']
  else:#elif datum.has_key('review/taste'):
      sty[key] = [1, datum['review/taste']]

def feature(datum):
  feat = [1]
  key = datum['beer/style']
  feat.append(key)
  getStylesStat(datum, styles)
  return feat

X = [feature(d) for d in data]

### HW1 Answer
#for key in styles:
#    print "styles: {}".format(key)
#    print "number of reviews: {}".format(styles[key][cntIdx])
#    print "average value of 'review/taste': {}".format(float(styles[key][revIdx]) / float(styles[key][cntIdx]))
#

### HW2 Answer
def featureAIPA(datum):
  feat = [1]
  if datum['beer/style'] == "American IPA":
    feat.append(1)
  else:
    feat.append(0)
  return feat

#X = [featureAIPA(d) for d in data]
#y = [d['review/overall'] for d in data]
#theta,residuals,rank,s = numpy.linalg.lstsq(X, y)
#print "theta = {} residauls = {}".format(theta, residuals)

### HW3 Answer
def makePrediction(X, y, theta):
    X = numpy.matrix(X)
    y = numpy.matrix(y)
    predictions = numpy.matrix(theta) * X.T
    result = predictions - y
    sumofsquare = 0
    for elem in result.A1:
        sumofsquare += elem**2

    print "testing MSE = {}".format(sumofsquare/len(result.A1))

def getHw3Answer(data):
    halfsz = len(data)/2
    sz     = len(data)
    trainningData = data[:halfsz];
    testingData   = data[halfsz:sz];
    experiments = [trainningData, testingData]

    X = [featureAIPA(d) for d in trainningData] 
    y = [d['review/taste'] for d in trainningData] 
    theta,residuals,rank,s = numpy.linalg.lstsq(X, y)
    print "training data MSE = {}".format(float(residuals) / float(len(trainningData)))
    X = [featureAIPA(d) for d in testingData] 
    y = [d['review/taste'] for d in testingData] 
    makePrediction(X, y, theta)


### HW4 Answer ###

def feature(datum, style):
  feat = [1]
  if datum['beer/style'] == style:
    feat.append(1)
  else:
    feat.append(0)
  return feat

#X = [feature(d) for d in data]
#y = [d['review/taste'] for d in data]
#theta,residuals,rank,s = numpy.linalg.lstsq(X, y)
#print "theta = {} residauls = {}".format(theta, residuals)

def getStyleOrderMap(stylesStat):
    stylesOrderMap = {}
    stylesList = [style for style in stylesStat if stylesStat[style][cntIdx] >= 50]
    nStyles = len(stylesList)
    for idx in range(len(stylesList)):
        stylesOrderMap[stylesList[idx]] = idx + 1
    return stylesOrderMap, nStyles

def feature(stylesOrderMap, style, nStyles):
    feat = [0 for i in range(nStyles)]
    feat[0] = 1;
                                                               
    if stylesOrderMap[style] < nStyles:
        feat[stylesOrderMap[style]] = 1

    return feat

def filter(stylesStat, data):
    halfsz = len(data)/2
    sz     = len(data)
    trainningData = data[:halfsz];
    testingData   = data[halfsz:sz];
    training = [d for d in data if stylesStat[d['beer/style']][cntIdx] >= 50 and d.has_key('review/taste')]
    testing =  [d for d in testingData if stylesStat[d['beer/style']][cntIdx] >= 50 and d.has_key('review/taste')]
    return [training, testing]

def getHw4Answer():
    stylesStat = {};
    for d in week1.data: 
        getStylesStat(d, stylesStat)

    experiments = filter(stylesStat, week1.data)
    stylesOrderMap, nStyles = getStyleOrderMap(stylesStat) 
    X = [feature(stylesOrderMap, d['beer/style'], nStyles) for d in experiments[0]] 
    y = [d['review/taste'] for d in experiments[0]] 
    theta,residuals,rank,s = numpy.linalg.lstsq(X, y)
    print "theta = {}".format(theta)
    print "training MSE = {}".format(float(residuals) / float(len(experiments[0])))
    X1 = [feature(stylesOrderMap, d['beer/style'], nStyles) for d in experiments[1]] 
    y1 = [d['review/taste'] for d in experiments[1]] 
    makePrediction(X1, y1, theta)

def getHw5Answer(data):
    halfsz = len(data)/2
    sz     = len(data)

    X = [[d['beer/ABV'], d['review/taste']] for d in data]
    y = ['American IPA' in d['beer/style'] for d in data] 

    X_train = X[:halfsz];
    y_train = y[:halfsz];

    X_test = X[halfsz:sz];
    y_test = y[halfsz:sz];

    X = numpy.matrix(X)
    clf = svm.SVC(C=1000, kernel='sigmoid')
    clf.fit(X_train, y_train) 

    test_predictions = clf.predict(X_train)
    match = [(x==y) for x,y in zip(test_predictions, y_test)]
    print "accuracy for trainning set = {}".format(sum(match) * 1.0 / len(match))

    train_predictions = clf.predict(X_test)
    match = [(x==y) for x,y in zip(train_predictions, y_train)]
    print "accuracy for test set = {}".format(sum(match) * 1.0 / len(match))

def getHw6Answer(data):
    halfsz = len(data)/2
    sz     = len(data)

    X = [[d['beer/ABV'], d['review/palate']] for d in data]
    y = ['American IPA' in d['beer/style'] for d in data] 

    X_train = X[:halfsz];
    y_train = y[:halfsz];

    X_test = X[halfsz:sz];
    y_test = y[halfsz:sz];

    X = numpy.matrix(X)
    clf = svm.SVC(C=1000, kernel='sigmoid')
    clf.fit(X_train, y_train) 

    test_predictions = clf.predict(X_train)
    match = [(x==y) for x,y in zip(test_predictions, y_test)]
    print "accuracy for trainning set = {}".format(sum(match) * 1.0 / len(match))

    train_predictions = clf.predict(X_test)
    match = [(x==y) for x,y in zip(train_predictions, y_train)]
    print "accuracy for test set = {}".format(sum(match) * 1.0 / len(match))

def getHw7Answer(data):
    halfsz = len(data)/2
    sz     = len(data)

    X = [[d['beer/ABV'], d['review/palate']] for d in data]
    y = ['American IPA' in d['beer/style'] for d in data] 

    X_train = X[:halfsz];
    y_train = y[:halfsz];

    X_test = X[halfsz:sz];
    y_test = y[halfsz:sz];

    X = numpy.matrix(X)
    Cs = [0.1, 10, 1000, 100000]
    for c in Cs:
        clf = svm.SVC(C=c, kernel='sigmoid')
        clf.fit(X_train, y_train) 

        test_predictions = clf.predict(X_train)
        match = [(x==y) for x,y in zip(test_predictions, y_test)]
        print "C  = {}".format(c)
        print "accuracy for trainning set = {}".format(sum(match) * 1.0 / len(match))

        train_predictions = clf.predict(X_test)
        match = [(x==y) for x,y in zip(train_predictions, y_train)]
        print "accuracy for test set = {}".format(sum(match) * 1.0 / len(match))

def inner(x,y):
  return sum([x[i]*y[i] for i in range(len(x))])

def sigmoid(x):
  return 1.0 / (1 + exp(-x))

# NEGATIVE Log-likelihood
def f(theta, X, y, lam):
  loglikelihood = 0
  for i in range(len(X)):
    logit = inner(X[i], theta)
    loglikelihood -= log(1 + exp(-logit))
    if not y[i]:
      loglikelihood -= logit
  for k in range(len(theta)):
    loglikelihood -= lam * theta[k]*theta[k]
  print "ll =", loglikelihood
  return -loglikelihood

# NEGATIVE Derivative of log-likelihood
def fprime(theta, X, y, lam):
  dl = [0.0]*len(theta)
  for i in range(len(X)):
    # Fill in code for the derivative
    sig = 1.0 - sigmoid(inner(X[i], theta))
    for k in range(len(dl)):
        dl[k] += (X[i][k] * sig)
        if not y[i]:
            dl[k] -= X[i][k]
  dl -= 2 * lam * theta
  # Negate the return value since we're doing gradient *ascent*
  return numpy.array([-x for x in dl])


def getHw8Answer(data): 
    # Extract features and labels from the data
    halfsz = len(data)/2
    sz     = len(data)

    X = [[d['beer/ABV'], d['review/taste']] for d in data]
    y = ['American IPA' in d['beer/style'] for d in data] 

    X_train = X[:halfsz]
    y_train = y[:halfsz]

    X_test = X[halfsz:sz];
    y_test = y[halfsz:sz];


    # If we wanted to split with a validation set:
    #X_valid = X[len(X)/2:3*len(X)/4]
    #X_test = X[3*len(X)/4:]

    # Use a library function to run gradient descent (or you can implement yourself!)
    theta,l,info = scipy.optimize.fmin_l_bfgs_b(f, [0]*len(X[0]), fprime, args = (X_train, y_train, 1.0))
    print "Final log likelihood =", -l

    test_predictions = [y >= 0.5 for y in [sigmoid(inner(x, theta)) for x in X_test]]
    match = [(x==y) for x,y in zip(test_predictions, y_test)] 
    # Compute the accuracy
    print "Accuracy = {}".format((sum(match) * 1.0) / len(match))

getHw8Answer(week1.data)
