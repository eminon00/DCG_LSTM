
import numpy as np
import matplotlib.pyplot as plt
def load_test_DataSet(fileName):
    Entity_N = 10
    datadict = {}
    for i in range(Entity_N):
        datadict[i] = []
    numFeat = len(open(fileName).readline().split(','))  #
    fr = open(fileName)
    lines = fr.readlines()
    P = 0
    for line in lines:
        lineArr =[]
        curLine = line.strip().split(',')
        for i in range(1,numFeat):
            #print P,curLine
            lineArr.append(float(curLine[i]))
        datadict[P%10].append(lineArr)
        P += 1
    fr.close()
    return datadict

datadict = load_test_DataSet('./LPL/lpl2019summer_w1d1_dmo_vs_edg_1.csv')
N = 10
X1_Mat = [[] for i in range(N)]
X2_Mat = [[] for i in range(N)]
T = len(datadict[0])
for t in range(T):
    for i in range(N):
        X1_Mat[i].append(datadict[i][t][2])
        X2_Mat[i].append(datadict[i][t][3])
X1_Mat = np.array(X1_Mat)
X2_Mat = np.array(X2_Mat)
index = np.arange(1, T + 1, 1)
a_linewidth = 1.0
for i in range(N):
    plt.subplot(10, 1, i+1)
    plt.plot(index, X2_Mat[i], color='blue', linewidth=a_linewidth)
    plt.plot(index, X1_Mat[i], color='green', linewidth=a_linewidth)
    plt.xticks([])
    plt.yticks([])
plt.savefig("./saved_result/Figure/" + str("Figure2"))