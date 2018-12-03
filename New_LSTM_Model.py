import numpy as np
import os
import pandas
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from tensorflow import set_random_seed

name_for_saved_model = "myLSTM_model"

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
set_random_seed(seed)


dir_path = os.path.dirname(os.path.realpath(__file__))+ "\PointingDynamicsDataset\P1"

def reset_time(time, target):

    #reset time to 0 when target changes
    old_target = target[0]
    target_changed_pos = []
    for i in range(len(target)):
        if old_target != target[i]:
            old_target = target[i]
            target_changed_pos.append(i)

    target_changed_counter = 0
    for i in range(len(time)):
        if target_changed_pos[target_changed_counter] == i:
            time[i] = 0
            if len(target_changed_pos) < target_changed_counter:
                target_changed_counter += 1
        else:
            if i > 0:
                time[i] = time[i - 1] + 0.002

    return time

def save_model(name):
    model_json = model.to_json()
    with open(name + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(name+ ".h5")
    print("Saved model to disk")

def read_CSV_File(file_path):
    # load dataset
    dataframe = pandas.read_csv(file_path, delimiter=',', skiprows=1, header=None, usecols=(24, 28, 29, 31, 32, 33))
    dataset = dataframe.values

    width = []
    target = []
    sga = []
    last_sgv = []
    last_sgy = []
    time = []

    width.extend(dataset[:, 2])
    time.extend(dataset[:, 0])
    target.extend(dataset[:, 1])
    sga.extend(dataset[:, 5])
    last_sgy.extend(dataset[:, 3])
    last_sgv.extend(dataset[:, 4])

    # get sga-1
    last_sga = np.array(sga)
    last_sgy = np.array(last_sgy)
    last_sgv = np.array(last_sgv)

    # add 0 as first element, delete last element
    last_sga = np.insert(last_sga, 0, 0)
    last_sga = last_sga[:-1]
    last_sgy = np.insert(last_sgy, 0, 0)
    last_sgy = last_sgy[:-1]
    last_sgv = np.insert(last_sgv, 0, 0)
    last_sgv = last_sgv[:-1]

    # reset time to 0 when target changes
    time = reset_time(time, target)

    #  Reshape
    width = np.reshape(width, (-1, 1))
    last_sga = np.reshape(last_sga, (-1, 1))
    last_sgy = np.reshape(last_sgy, (-1, 1))
    last_sgv = np.reshape(last_sgv, (-1, 1))
    time = np.reshape(time, (-1, 1))
    target = np.reshape(target, (-1, 1))

    # add last_sga and time to input
    input = np.concatenate((last_sga, last_sgv), axis=1)
    input = np.concatenate((input, last_sgy), axis=1)
    input = np.concatenate((input, width), axis=1)
    input = np.concatenate((input, target), axis=1)
    input = np.concatenate((input, time), axis=1)

    return [input, sga]


input_fit = []
sga_fit = []

for file in os.listdir(dir_path):
    print(file)
    file_path = os.path.join(dir_path, file)

    data_fit = read_CSV_File(file_path)
    input_fit.extend(data_fit[0])
    sga_fit.extend(data_fit[1])

# MinMax Scaler to 0 - 1
scaler = MinMaxScaler(feature_range=(0, 1))
input_fit = scaler.fit_transform(input_fit)
input_fit = np.reshape(input_fit, (-1, 6))

sga_fit = np.reshape(sga_fit, (-1,1))
sga_fit = scaler.fit_transform(sga_fit)

train_X = input_fit
train_y = sga_fit

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))


# design network
model = Sequential()
model.add(LSTM(40, return_sequences= True,input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(LSTM(20, return_sequences= True))
model.add(LSTM(3))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')

# fit network
model.fit(train_X, train_y, epochs=30, batch_size=100, verbose=1)

score = model.evaluate(train_X, train_y, verbose=1)
print(score*100)

save_model(name_for_saved_model)