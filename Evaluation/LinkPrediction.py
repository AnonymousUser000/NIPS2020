import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from scipy.io import loadmat


dim = 32
outEmbPath = ''
inEmbPath = ''
trainPath = ''
testPath = ''

outEmb = loadmat(outEmbPath)['outEmb']
inEmb = loadmat(inEmbPath)['inEmb']

numNodes = outEmb.shape[0]

trainSet = loadmat(trainPath)['trainSet']
lenTrainSet = trainSet.shape[0]
trainX = np.zeros((lenTrainSet, dim * 4))
trainY = np.zeros((lenTrainSet, 1))
for i in range(lenTrainSet):
    edge = trainSet[i]
    u = edge[0] - 1
    v = edge[1] - 1
    if edge[2] > 0:
        trainY[i] = 1
    else:
        trainY[i] = 0
    trainX[i, : dim] = outEmb[u]
    trainX[i, dim: dim * 2] = inEmb[u]
    trainX[i, dim * 2: dim * 3] = outEmb[v]
    trainX[i, dim * 3: dim * 4] = inEmb[v]

testSet = loadmat(testPath)['testSet']
lenTestSet = testSet.shape[0]
testX = np.zeros((lenTestSet, dim * 4))
testY = np.zeros((lenTestSet, 1))
for i in range(lenTestSet):
    edge = testSet[i]
    u = edge[0] - 1
    v = edge[1] - 1
    if edge[2] > 0:
        testY[i] = 1
    else:
        testY[i] = 0
    testX[i, : dim] = outEmb[u]
    testX[i, dim: dim * 2] = inEmb[u]
    testX[i, dim * 2: dim * 3] = outEmb[v]
    testX[i, dim * 3: dim * 4] = inEmb[v]

lr = LogisticRegression()
lr.fit(trainX, trainY)
testScore = lr.predict_proba(testX)[:, 1]
testPred = lr.predict(testX)

lp_auc_score = roc_auc_score(testY, testScore, average='macro')

lp_f1_score_macro = f1_score(testY, testPred, average='macro')

print(lp_auc_score)
print(lp_f1_score_macro)


