
from utils import *
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dropout, Dense, LSTM
import pandas as pd



##Comparison algorithm 3

def LSTMModel(x_train, y_train, x_test, y_test,number_of_memories):

    np.random.seed(7)
    np.random.shuffle(x_train)
    np.random.seed(7)
    np.random.shuffle(y_train)
    tf.random.set_seed(7)

    model = tf.keras.Sequential([
        LSTM(number_of_memories, return_sequences=True),
        Dropout(0.2),
        LSTM(100),
        Dropout(0.2),
        Dense(10)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss='huber_loss')

    checkpoint_save_path = "./checkpoint/LSTM_LPL.ckpt"

    # if os.path.exists(checkpoint_save_path + '.index'):
    # print('-------------load the model-----------------')
    # model.load_weights(checkpoint_save_path)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                     save_weights_only=True,
                                                     save_best_only=True,
                                                     monitor='val_loss')

    history = model.fit(x_train, y_train, batch_size=64, epochs=1, validation_data=(x_test, y_test), validation_freq=1)

    model.summary()
    '''
    file = open('./weights.txt', 'w')  
    for v in model.trainable_variables:
        file.write(str(v.name) + '\n')
        file.write(str(v.shape) + '\n')
        file.write(str(v.numpy()) + '\n')
    file.close()

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()
    '''
    ################## predict ######################

    predicted_bb = model.predict(x_test)

    '''
    plt.plot(y_test[:100], color='red', label='MaoTai Stock Price')
    plt.plot(predicted_bb[:100], color='blue', label='Predicted MaoTai Stock Price')
    plt.title('MaoTai Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('MaoTai Stock Price')
    plt.legend()
    plt.show()
    '''
    #del model
    return predicted_bb


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
                test_filename = test_filename[0] + '.csv'
                x_train, y_train, x_test, y_test, temp_label, location = LSTM_ReadLPLDataset(test_filename,
                                                                                                number_of_memories)
                predicted_bb = LSTMModel(x_train, y_train, x_test, y_test,number_of_memories)
                #Ree = reconstruction_error(predicted_bb, y_test)
                np.savetxt('./saved_result/LSTM/' + 'predicted_bb_' + test_filename, predicted_bb)
                short_score = reconstruction_error1(predicted_bb, y_test)
                Acc, Rp = AccAndRp(temp_label, short_score)
                avg_Acc.append(Acc)
                avg_Rp.append(Rp)
                Table.append([test_filename, Acc, Rp])
            Table.append(np.mean(np.array(avg_Acc)))
            Table.append(np.mean(np.array(avg_Rp)))
            np.savetxt('./saved_result/Table/' + 'LSTM_' + '_Table.txt', Table, delimiter=' & ', fmt='%s')
        else:
            test_filename_list = pd.read_csv('./catalogue.csv', header=None)
            Table = []
            avg_f1 = []
            avg_auc = []
            for test_filename in test_filename_list.values:
                print(test_filename)
                test_filename = test_filename[0] + '.csv'
                x_train, y_train, x_test, y_test, temp_label, location = LSTM_ReadLPLDataset(test_filename,
                                                                                                number_of_memories)
                predicted_bb = pd.read_csv('./saved_result/LSTM/predicted_bb_' + test_filename, header=None, sep=' ')
                predicted_bb = np.array(predicted_bb)
                Ree = reconstruction_error(predicted_bb, y_test)
                alpha = temp_label.count(-1) / len(temp_label)
                #temp_label_pred = TOP_alpha_1(Ree, alpha)
                y_pred, temp_label_pred = DCG(location, Ree, wide, high, temp_label, alpha)
                #temp_label_pred = TOP_alpha_1(Ree, 0.3)
                # print(temp_label)
                # print(temp_label_pred)
                final_p, final_r, final_f = F1Score(temp_label, temp_label_pred)
                # print('LSTM########################################')
                # print('avg_precision=' + str(final_p))
                # print('avg_recall=' + str(final_r))
                # print('avg_f1=' + str(final_f))
                short_score = reconstruction_error1(predicted_bb, y_test)
                auc = roc_AUC(temp_label, short_score)
                avg_auc.append(auc)
                avg_f1.append(final_f)
                Table.append([test_filename, auc, final_p, final_r, final_f])
            Table.append(np.mean(np.array(avg_auc)))
            Table.append(np.mean(np.array(avg_f1)))
            np.savetxt('./saved_result/Table/' + 'DCG_' + '_Table.txt', Table, delimiter=' & ', fmt='%s')