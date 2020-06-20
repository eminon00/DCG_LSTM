import pathlib
import sys

from sklearn.neighbors import lof
from utils import *

#Comparison algorithm 1

def LOFModel(_abnormal_data, _neighbor=5, _job=2, _contamination=0.05, _metric='euclidean'):
    clf = lof.LocalOutlierFactor(n_neighbors=_neighbor, n_jobs=_job, metric=_metric)
    y_pred = clf.fit_predict(_abnormal_data)
    score = clf.negative_outlier_factor_#
    return score, y_pred




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
                abnormal_data, abnormal_label, temp_label, location = ReadLPLDataset(test_filename)
                #k = apk(abnormal_data, 4)
                #print(k)
                score, y_pred = LOFModel(abnormal_data, 5, 2, 'euclidean')
                #print(len(score),len(temp_label))
                short_score = []
                for t in range(int(len(score)/N)):
                    short_score.append(np.max(score[t*N:t*N+N]))
                Acc, Rp = AccAndRp(temp_label, short_score)
                avg_Acc.append(Acc)
                avg_Rp.append(Rp)
                Table.append([test_filename, Acc, Rp])
            Table.append(np.mean(np.array(avg_Acc)))
            Table.append(np.mean(np.array(avg_Rp)))
            np.savetxt('./saved_result/Table/' + 'lof_' + '_Table.txt', Table, delimiter=' & ', fmt='%s')

