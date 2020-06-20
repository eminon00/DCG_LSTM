from utils import *

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dropout, Dense, LSTM
import pandas as pd



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

    checkpoint_save_path = "./checkpoint/DCG_LPL.ckpt"



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
                x_train, y_train, x_test, y_test, temp_label, location = DCG_ReadLPLDataset(test_filename,
                                                                                                number_of_memories, wide, high)
                predicted_bb = LSTMModel(x_train, y_train, x_test, y_test,number_of_memories)
                #np.savetxt('./saved_result/DCG/' + 'predicted_bb_' + test_filename, predicted_bb)
                Ree = reconstruction_error(predicted_bb, y_test)
                short_score = DCG(location,Ree,wide,high)
                Acc, Rp = AccAndRp(temp_label, short_score)
                avg_Acc.append(Acc)
                avg_Rp.append(Rp)
                Table.append([test_filename, Acc, Rp])
            Table.append(np.mean(np.array(avg_Acc)))
            Table.append(np.mean(np.array(avg_Rp)))
            np.savetxt('./saved_result/Table/' + 'DCG_' + '_Table.txt', Table, delimiter=' & ', fmt='%s')
