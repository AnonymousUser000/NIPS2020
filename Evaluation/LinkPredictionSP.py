import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score
from scipy.io import loadmat


dim = 64
embPath = ''
trainPath = ''
testPath = ''

emb = loadmat(embPath)['emb']
numNodes = emb.shape[0]

trainSet = loadmat(trainPath)['trainSet']
lenTrainSet = trainSet.shape[0]
trainX = np.zeros((lenTrainSet, dim * 2))
trainY = np.zeros((lenTrainSet, 1))
for i in range(lenTrainSet):
    edge = trainSet[i]
    u = edge[0] - 1
    v = edge[1] - 1
    if edge[2] > 0:
        trainY[i] = 1
    else:
        trainY[i] = 0
    trainX[i, : dim] = emb[u]
    trainX[i, dim: dim * 2] = emb[v]

testSet = loadmat(testPath)['testSet']
lenTestSet = testSet.shape[0]
testX = np.zeros((lenTestSet, dim * 2))
testY = np.zeros((lenTestSet, 1))
for i in range(lenTestSet):
    edge = testSet[i]
    u = edge[0] - 1
    v = edge[1] - 1
    if edge[2] > 0:
        testY[i] = 1
    else:
        testY[i] = 0
    testX[i, : dim] = emb[u]
    testX[i, dim: dim * 2] = emb[v]

lr = LogisticRegression()
lr.fit(trainX, trainY)

testScore = lr.predict_proba(testX)[:, 1]
testPred = lr.predict(testX)

lp_auc_score = roc_auc_score(testY, testScore, average='macro')
lp_f1_score_macro = f1_score(testY, testPred, average='macro')

print(lp_auc_score)
print(lp_f1_score_macro)

