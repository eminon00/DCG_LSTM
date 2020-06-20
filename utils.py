import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from numpy import *
import numpy as np
from sklearn.neighbors import KDTree
import os



def DCG(location,long_score,wide,high):
    object_num=10
    aver_T = int(len(long_score)/object_num)
    final_OF = []
    for t in range(aver_T):
        W = [0.0 for i in range(object_num)]
        loca = location[t*object_num:(t+1)*object_num]
        sub_score = long_score[t * object_num:(t * object_num + object_num)]
        maxp = np.argmax(sub_score)  # W
        ap = maxp
        max_ap = min(sub_score)
        for i in range(object_num):
            if abs(loca[i][0] - loca[maxp][0]) <= wide and abs(loca[i][1] - loca[maxp][1]) <= high:
                if max_ap<sub_score[i] and i!=maxp:
                    ap=i
                    max_ap = sub_score[i]

        for i in range(object_num):
            for j in range(i, object_num):
                if abs(loca[i][0] - loca[j][0]) <= wide and abs(loca[i][1] - loca[j][1]) <= high:
                    W[i] += 1
                    W[j] += 1

        W = np.array(W)
        OF = W.dot(sub_score)
        W1 = W[maxp]
        W2 = W[ap]

        final_OF.append((sub_score[maxp]*W1+sub_score[ap]*W2)/(W1+W2))
        #final_OF.append((sub_score[maxp]+sub_score[ap])/2)

    return final_OF

def TOP_alpha_2(all_of,y_test,alpha):
    N=10
    T = int(len(all_of)/N)
    of = []
    for i in range(T):
        of.append(max(all_of[i*N:i*N+N])[0])
    of = np.array(of)
    print(of[:20])
    print(of[-20:])
    temp_label_pred = [1 for i in range(T)]
    index = np.argsort(of)[::-1]#da dao xiao
    t=0
    num = 0
    while t<T and num<int(T*alpha):
        if min(y_test[t*N:t*N+N])<0.5:
            temp_label_pred[index[t]] = -1
            num+=1
        t+=1

    return temp_label_pred

def TOP_alpha_1(all_of,alpha):
    N=10
    T = int(len(all_of)/N)
    of = []
    for i in range(T):
        of.append(max(all_of[i*N:i*N+N]))
    of = np.array(of)

    temp_label_pred = [1 for i in range(T)]
    index = np.argsort(of)[::-1]#da dao xiao
    for t in range(int(T*alpha)):
        temp_label_pred[index[t]] = -1

    return temp_label_pred
def TOP_alpha(all_of,alpha):
    N=10
    T = int(len(all_of)/N)
    of = []
    for i in range(T):
        of.append(max([all_of[i*N+j] for j in range(N)]))
    of = np.array(of)
    temp_label_pred = [1 for i in range(T)]
    index = np.argsort(of)#xiao dao da/-1???
    for t in range(int(T*alpha)):
        temp_label_pred[index[t]] = -1

    return temp_label_pred



def chage_reallabel(test_LPL,number_of_memories):

    N = 10
    number_of_memories1 = int(number_of_memories /N)#int(number_of_memories /N)
    T = int(len(test_LPL) / N)
    temp_label = [1 for i in range(T-number_of_memories1)]
    com = 0
    for i in range(N):
        for t in range(T - 2, number_of_memories1 -1, -1):
            blood = int(test_LPL[t * N + i:t * N + i + 1][3])
            ex_blood = int(test_LPL[t * N + i+ N:t * N + i + N+1][3])
            if blood == 3:
                com = 1
                temp_label[t-number_of_memories1] = -1
            if com == 1 and blood != 3:
                if blood > ex_blood:
                    temp_label[t-number_of_memories1] = -1
                    temp_label[t - 1-number_of_memories1] = -1
                elif blood < ex_blood:
                    com = 0
    test_label = test_LPL.iloc[:, -1].values
    for t in range(T-number_of_memories1):
        if temp_label[t] != -1:
            for i in range(N):
                test_label[t * 10 + i] = 1

    #print(test_LPL)
    #print(temp_label)
    return temp_label

def DCGG(loca,sub_score, wide, high,N):
    W = [0.0 for i in range(N)]

    maxp = np.argmax(sub_score)  # W
    ap = maxp
    max_ap = min(sub_score)
    for i in range(N):
        if abs(loca[i][0] - loca[maxp][0]) <= wide and abs(loca[i][1] - loca[maxp][1]) <= high:
            if max_ap < sub_score[i] and i != maxp:
                ap = i
                max_ap = sub_score[i]

    for i in range(N):
        for j in range(i, N):
            if abs(loca[i][0] - loca[j][0]) <= wide and abs(loca[i][1] - loca[j][1]) <= high:
                W[i] += 1
                W[j] += 1

    W = np.array(W)
    OF = W.dot(sub_score)
    W1 = W[maxp]
    W2 = W[ap]
    # final_OF.append((sub_score[maxp]*W1+sub_score[ap]*W2)/(W1+W2))
    # final_OF.append((sub_score[maxp] + sub_score[ap]) / 2)
    y = (sub_score[maxp] + sub_score[ap]) / 2
    return W

def DCG_ReadLPLDataset(test_filename, number_of_memories, wide, high):
    #test_filename = pd.read_csv('./catalogue.csv')
    #training_LPL = pd.DataFrame(columns=('name', 'x', 'y', 'blood','blue','tag'))
    N = 10

    sc = MinMaxScaler(feature_range=(0, 1))  # (0，1)
    deltaT = int(number_of_memories / N)
    x_train = []
    y_train = []
    for root, dir, files in os.walk('./LPL/'):
        test_filename = os.path.join(root, test_filename)
        for file in files:
            file_name = os.path.join(root, file)
            if test_filename!=file_name:
                training_LPL = pd.read_csv(file_name,header=None)
                training_set = training_LPL.iloc[:, 1:-1].values
                #buf_temp_label = chage_reallabel(training_LPL, number_of_memories)
                location = training_LPL.iloc[:, [1, 2]].values
                aver_T = int(len(training_set) / N)
                W = []
                for t in range(aver_T):
                    loca = location[t * N:(t + 1) * N]
                    sub_score = training_set[t * N:(t * N + N), 2]
                    W.extend(DCGG(loca, sub_score, wide, high, N))
                W = np.array(W)
                training_set = np.c_[training_set,W]
                training_set_scaled = sc.fit_transform(training_set)
                for t in range(deltaT, aver_T):
                    x_train.append(training_set_scaled[(t- deltaT)*N:t * N, :])
                    y_train.append(training_set_scaled[t * N:t * N + N, 2])


    test_LPL = pd.read_csv(test_filename,header=None)
    temp_label = chage_reallabel(test_LPL, number_of_memories)
    test_set = test_LPL.iloc[:, 1:-1].values
    location = test_LPL.iloc[:, [1, 2]].values
    x_test = []
    y_test = []
    W = []
    for t in range(int(len(test_set) / N)):
        loca = location[t * N:(t + 1) * N]
        sub_score = test_set[t * N:(t * N + N), 2]
        W.extend(DCGG(loca, sub_score, wide, high, N))
    W = np.array(W)
    test_set = np.c_[test_set, W]
    test_set = sc.fit_transform(test_set)
    for t in range(deltaT, int(len(test_set )/ N)):
        x_test.append(test_set[(t - deltaT) * N:t * N, :])
        y_test.append(test_set[t * N:t * N + N, 2])  #
    #print(len(x_test),len(y_test))
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], deltaT*N, 5))
    x_test, y_test = np.array(x_test), np.array(y_test)
    x_test = np.reshape(x_test, (x_test.shape[0], deltaT*N, 5))
    location = test_LPL.iloc[number_of_memories:, [1, 2]].values
    return x_train,y_train,x_test,y_test,temp_label, location

def LSTM_ReadLPLDataset(test_filename, number_of_memories):
    #test_filename = pd.read_csv('./catalogue.csv')
    #training_LPL = pd.DataFrame(columns=('name', 'x', 'y', 'blood','blue','tag'))
    N = 10
    # 归一化
    sc = MinMaxScaler(feature_range=(0, 1))  # 定义归一化：归一化到(0，1)之间
    deltaT = int(number_of_memories / N)
    x_train = []
    y_train = []
    for root, dir, files in os.walk('./LPL/'):
        test_filename = os.path.join(root, test_filename)
        for file in files:
            file_name = os.path.join(root, file)
            if test_filename!=file_name:
                training_LPL = pd.read_csv(file_name,header=None)
                training_set = training_LPL.iloc[:, 1:-1].values
                training_set_scaled = sc.fit_transform(training_set)
                for t in range(deltaT, int(len(training_set_scaled)/N)):
                    #if buf_temp_label[t-deltaT]!=-1:
                    x_train.append(training_set_scaled[(t- deltaT)*N :t*N, :])
                    y_train.append(training_set_scaled[t*N:t*N+N, 2])  #


    test_LPL = pd.read_csv(test_filename,header=None)
    temp_label = chage_reallabel(test_LPL, number_of_memories)
    test_set = test_LPL.iloc[:, 1:-1].values
    test_set = sc.transform(test_set)
    x_test = []
    y_test = []
    for t in range(deltaT, int(len(test_set )/ N)):
        x_test.append(test_set[(t- deltaT) * N :t * N, :])
        y_test.append(test_set[t * N:t * N + N, 2])  #
    print(len(x_test),len(y_test))

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], number_of_memories, 4))

    x_test, y_test = np.array(x_test), np.array(y_test)
    x_test = np.reshape(x_test, (x_test.shape[0], number_of_memories, 4))
    location = test_LPL.iloc[number_of_memories:, [1, 2]].values
    return x_train,y_train,x_test,y_test,temp_label, location

def isf_ReadLPLDataset(test_filename, number_of_memories):
    N = 10

    sc = MinMaxScaler(feature_range=(0, 1))
    deltaT = int(number_of_memories / N)
    test_LPL = pd.read_csv(test_filename, header=None)
    temp_label = chage_reallabel(test_LPL, number_of_memories)
    test_set = test_LPL.iloc[:, 1:-1].values
    test_set = sc.fit_transform(test_set)
    x_test = []
    for t in range(deltaT, int(len(test_set) / N)):
        x_test.append(test_set[(t - deltaT) * N:t * N, :])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], number_of_memories,4))
    return x_test, temp_label

def ReadLPLDataset(_file_name, _normalize=True):
    abnormal = pd.read_csv(_file_name, header=None, index_col=None, skiprows=0, sep=',')
    abnormal_data = abnormal.iloc[:, [1, 2, 3, 4]].values
    abnormal_label = abnormal.iloc[:, 5].values
    # Normal = 0, Abnormal = 1 => # Normal = 1, Abnormal = -1
    location = abnormal.iloc[:, [1, 2]].values
    # abnormal_data = np.expand_dims(abnormal_data, axis=1)

    #change label
    N = 10
    T = int(len(abnormal_label)/N)
    temp_label = [1 for i in range(T)]
    com = 0
    for i in range(N):
        for t in range(T - 2, -1, -1):
            blood = abnormal_data[t*10+i][2]
            ex_blood = abnormal_data[t*10+i + 1][2]
            if blood == 3:
                com = 1
                temp_label[t] = -1
            if com == 1 and blood != 3:
                if blood > ex_blood:
                    temp_label[t] = -1
                    temp_label[t - 1] = -1
                elif blood < ex_blood:
                    com = 0

    for t in range(T):
        if temp_label[t] != -1:
            for i in range(N):
                abnormal_label[t*10+i] = 0

    #MinMaxScaler
    abnormal_label = np.expand_dims(abnormal_label, axis=1)
    if _normalize == True:
        scaler = MinMaxScaler(feature_range=(0, 1))
        abnormal_data = scaler.fit_transform(abnormal_data)

    abnormal_label[abnormal_label == 1] = -1
    abnormal_label[abnormal_label == 0] = 1


    return abnormal_data, abnormal_label, temp_label,location



def reconstruction_error1(predicted_bb,y_test):
    Ree = []
    #print(predicted_bb)
    #print(y_test)
    for t in range(len(y_test)):
        max = 0
        for i in range(10):
            temp = np.square(predicted_bb[t][i] - y_test[t][i])**0.5
            if temp>max:
                max = temp
        Ree.append(max)
    #print(Ree)
    Ree = np.array(Ree)
    return Ree
def reconstruction_error(predicted_bb,y_test):
    Ree = []
    #print(predicted_bb)
    #print(y_test)
    for t in range(len(y_test)):
        for i in range(10):
            Ree.append(np.square(predicted_bb[t][i] - y_test[t][i])**0.5)
    #print(Ree)
    Ree = np.array(Ree)
    return Ree


def roc_AUC(label, score):
    roc_auc = 0.0
    N = len(label)
    o_num = label.count(-1)#label.count(-1)#list
    print(o_num)
    no_num = N - o_num
    for i in range(N):
        if label[i] == -1:
            for j in range(N):
                if label[j] == 1:
                    if score[i]>score[j]:
                        roc_auc+=1
                    elif  score[i]==score[j]:
                        roc_auc += 0.5
    roc_auc = roc_auc / (o_num * no_num)
    return round(roc_auc,3)


def F1Score(classLabels,predLabels):

    N = len(classLabels)
    TN = 0.0
    FP = 0.0
    FN = 0.0
    TP = 0.0
    for i in range(N):
        if classLabels[i] == 1 and predLabels[i] == 1:
            TN += 1
        elif classLabels[i] == 1 and predLabels[i] == -1:
            FP += 1
        elif classLabels[i] == -1 and predLabels[i] == 1:
            FN += 1
        else:
            TP += 1
    if TP==0 and FP==0:
        precision = 1
    else:
        precision = TP*1.0/(TP+FP)
    if TP == 0 and FN==0:
        recall=0.0
    else:
        recall = TP*1.0/(TP+FN)
    if (precision+recall)==0:
        F1 = 0
    else:
        F1 = 2*precision*recall/(precision+recall)
    print("f1",TP,(TP+FN))
    return round(precision,3),round(recall,3),round(F1,3)

def AccAndRp(classLabels,predStrengths):
    N = 0
    M = 0
    for item in classLabels:
        if item == -1:
            N+=1
    predIndex = np.argsort(-np.array(predStrengths))
    #predIndex = np.argsort(np.array(predStrengths))[::-1]
    RiSum = 0
    for i in range(N):
        if classLabels[predIndex[i]]==-1:
            #print i
            M+=1
            RiSum+=i+1
    Acc = M*1.0/N
    if RiSum == 0:
        Rp = 0
    else:
        Rp = M*(M+1.0)/(2*RiSum)
    print("Acc,Rp",M,N)
    #print "Accuracy is : ",Acc
    #print "Rank-Power is : ",Rp
    #print Acc,Rp
    return round(Acc,3),round(Rp,3)
# ReadS5Dataset('./YAHOO/data/A1Benchmark/real_1.csv')
# ReadGDDataset('./GD/data/Genesis_AnomalyLabels.csv')
# ReadHSSDataset('./HSS/data/HRSS_anomalous_standard.csv')
def apk(dataSet,m):
    N = len(dataSet)
    tree = KDTree(dataSet, leaf_size=2)
    sortedNN = []
    for i in range(N):
        dist, ind = tree.query([dataSet[i]], k=N)
        sortedD = [int(j) for j in ind[0][:].tolist()]
        if i in sortedD: #Consider the position of its own
            del sortedD[sortedD.index(i)]
        else:
            del sortedD[0]
        sortedNN.append(sortedD)

    OriMat = np.zeros([N, N], dtype=int8)  # MNG
    CommunityTag = []  # connected-branch-tag
    IDict = {}  # the number of edges
    VDict = {}  # the number of points
    cList = []
    for i in range(2*m):
        cList.append(0)
    CNum = 0
    minVarSum = inf
    for i in range(N): #Initialization
        CommunityTag.append(i)
        IDict[i] = 0
        VDict[i] = 1
    apk = 1
    iter = 0  # k
    while iter < N - 1:
        #Update the connected branch structure
        for i in range(N):
            ri = sortedNN[i][iter]
            if OriMat[ri][i] == 1:  # add edge
                Tagi = CommunityTag[i]
                Tagri = CommunityTag[ri]
                if Tagi == Tagri:  #edge side in the same connected branch
                    IDict[Tagi] += 1
                if Tagi != Tagri:  # Update
                    for j in range(N):  #
                        if CommunityTag[j] == Tagri:
                            CommunityTag[j] = Tagi
                    IDict[Tagi] += IDict[Tagri] + 1
                    VDict[Tagi] += VDict[Tagri]
                    del IDict[Tagri]
                    del VDict[Tagri]
            OriMat[i][ri] = 1

        #update cList
        for i in range(2*m-1):
            cList[i] = cList[i+1]
        cList[-1] = CNum
        CSet = set(CommunityTag)
        maxOClusterSize = 1
        for i in CSet:
            if VDict[i] <= 1:
                CNum += 1
                ClusterSize = CommunityTag.count(i)
                if ClusterSize > maxOClusterSize:
                    maxOClusterSize = ClusterSize
            else:
                if 2.0 * IDict[i] / (VDict[i] * (VDict[i] - 1)) == 1:
                    CNum += 1
                    ClusterSize = CommunityTag.count(i)
                    if ClusterSize>maxOClusterSize:
                        maxOClusterSize = ClusterSize

        bufarr = cList[:]
        bufarr.append(CNum)
        bufarr = np.array(bufarr)
        #print bufarr
        VarSum = var(bufarr)
        if VarSum == 0:
            apk = iter - m + 1
            if apk>maxOClusterSize:
                #print "maxOClusterSize",maxOClusterSize
                break
        if VarSum < minVarSum and iter - m + 1>0:
            apk = iter - m + 1
            minVarSum = VarSum
        if len(CSet) == 1:#<=2
            iter += 1
            break

        iter += 1

    return apk