
import matplotlib.pyplot as plt
import pandas as pd


def draw_gragh(test_filename, wide, high):
    N = 10
    training_LPL = pd.read_csv(test_filename, header=None)
    location = training_LPL.iloc[:, [1, 2]].values
    aver_T = int(len(location) / N)
    W = []
    for t in range(44,aver_T):
        loca = location[t * N:(t + 1) * N]
        x = []
        y = []
        for i in range(N):
            for j in range(i, N):
                if abs(loca[i][0] - loca[j][0]) <= wide and abs(loca[i][1] - loca[j][1]) <= high:
                    x = [loca[i][0],loca[j][0]]
                    y = [loca[i][1],loca[j][1]]
                    plt.plot(x, y, 'r', zorder=1, lw=3)
                    plt.scatter(x,y, s=120, zorder=2)
                    plt.xticks([])
                    plt.yticks([])
                    #x.append(loca[i])
                    #y.append(loca[j])
        #x = np.array(x)
        #y = np.array(y)

        plt.show()
        plt.clf()
    return

wide = 138
high = 76
draw_gragh('./LPL/lpl2019summer_w1d1_dmo_vs_edg_1.csv', wide, high)
#plt.savefig("./saved_result/Figure/" + str("Figure2"))