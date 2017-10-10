import week1
import numpy

data = week1.data 
styles = {}
cntIdx = 0
revIdx = 1

def feature(datum):
  feat = [1]
  key = datum['beer/style']
  feat.append(key)
  if key in styles:
      styles[key][cntIdx] += 1;
      styles[key][revIdx] += datum['review/taste']
  else:
      styles[key] = [1, datum['review/taste']]
  return feat

X = [feature(d) for d in data]

### HW1 Answer
#for key in styles:
#    print "styles: {}".format(key)
#    print "number of reviews: {}".format(styles[key][cntIdx])
#    print "average value of 'review/taste': {}".format(float(styles[key][revIdx]) / float(styles[key][cntIdx]))
#

### HW2 Answer
def feature(datum):
  feat = [1]
  if datum['beer/style'] == "American IPA":
    feat.append(1)
  else:
    feat.append(0)
  return feat

#X = [feature(d) for d in data]
#y = [d['review/overall'] for d in data]
#theta,residuals,rank,s = numpy.linalg.lstsq(X, y)
#print "theta = {} residauls = {}".format(theta, residuals)

### HW3 Answer
#halfsz = len(data)/2
#sz     = len(data)
#trainningData = data[:halfsz];
#testingData   = data[halfsz:sz];
#experiments = [trainningData, testingData]
#
#for experiment in experiments: 
#    X = [feature(d) for d in experiment] 
#    y = [d['review/overall'] for d in experiment] 
#    theta,residuals,rank,s = numpy.linalg.lstsq(X, y)
#    if experiment == trainningData:
#        print "training result:"
#    else:
#        print "testing result:"
#    print "MSE = {}".format(float(residuals) / float(len(experiment)))
#
### HW4 Answer ###

def feature(datum, style):
  feat = [1]
  if datum['beer/style'] == style:
    feat.append(1)
  else:
    feat.append(0)
  return feat

#X = [feature(d) for d in data]
#y = [d['review/overall'] for d in data]
#theta,residuals,rank,s = numpy.linalg.lstsq(X, y)
#print "theta = {} residauls = {}".format(theta, residuals)

halfsz = len(data)/2
sz     = len(data)
trainningData = data[:halfsz];
testingData   = data[halfsz:sz];
experiments = [trainningData, testingData]

def filterStyles(style):
    if styles[style][cntIdx] >= 50:
        return style

stylesList = [ style for style in styles if styles[style][cntIdx >= 50]]
nStyles = len(stylesList)
print stylesList
stylesOrderMap = {}

for idx in range(len(stylesList)):
    stylesOrderMap[stylesList[idx]] = idx


def feature(datum, style):
    feat = [0 for i in range(nStyles)]
    feat[0] = 1;

    if style in stylesOrderMap and stylesOrderMap[style] + 1 < nStyles:
        feat[stylesOrderMap[style] + 1] = 1
    #print feat
    return feat

def filterData(datum):
    style = datum['beer/style']
    if styles[style][cntIdx] >= 50:
        return datum

filtered_training_data = [d for d in trainningData if styles[d['beer/style']][cntIdx] >= 50]
filtered_testing_data =  [d for d in testingData if styles[d['beer/style']][cntIdx] >= 50]
stylesExperiment = [filtered_training_data, filtered_testing_data]

def runExp(experiments):
    for experiment in experiments: 
        X = [feature(d, d['beer/style']) for d in experiment] 
        y = [d['review/overall'] for d in experiment] 
        theta,residuals,rank,s = numpy.linalg.lstsq(X, y)
        if experiment == filtered_training_data: 
            print "training result:"
        else:
            print "testing result:"

        print "theta = {}".format(theta)
        print "residuals = {}".format(residuals)
        #print "MSE = {}".format(float(residuals) / float(len(experiment)))
runExp(stylesExperiment)
