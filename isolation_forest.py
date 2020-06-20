import pathlib
import sys

from sklearn.ensemble import IsolationForest
from utils import *

#Comparison algorithm 2

def IFModel(x_train, _estimators=100, _contamination=0.1):
    clf = IsolationForest(n_estimators=_estimators, contamination=_contamination)
    clf.fit(x_train)
    score = clf.decision_function(x_train)

    return max(score[-10:])

if __name__ == '__main__':
    try:
        sys.argv[1]
    except IndexError:
        wide = 138
        high = 76
        Run = True  # False
        number_of_memories = 200
        if Run == True:
            dataset = 0
            Table = []
            N = 10
            test_filename_list = pd.read_csv('./catalogue.csv', header=None)
            avg_Acc = []
            avg_Rp = []
            for test_filename in test_filename_list.values:
                test_filename = './LPL/'+test_filename[0] + '.csv'
                x_test, temp_label = isf_ReadLPLDataset(test_filename, number_of_memories)
                score = []
                for t in range(len(x_test)):
                    score.append(IFModel(x_test[t], _estimators=100, _contamination=0.1))
                print(len(score),len(temp_label))
                short_score = []
                for t in range(int(len(score) / N)):
                    short_score.append(np.max(score[t * N:t * N + N]))
                Acc, Rp = AccAndRp(temp_label, score)
                avg_Acc.append(Acc)
                avg_Rp.append(Rp)
                Table.append([test_filename, Acc, Rp])
            Table.append(np.mean(np.array(avg_Acc)))
            Table.append(np.mean(np.array(avg_Rp)))
            np.savetxt('./saved_result/Table/' + 'ISF_' + '_Table.txt', Table, delimiter=' & ', fmt='%s')

