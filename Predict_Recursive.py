import matplotlib.pyplot as plt
import os
import pandas
import numpy as np
from keras.models import model_from_json

testperson = "\P1"
model_name = "LSTM_model"

dir_path = os.path.dirname(os.path.realpath(__file__))\
           + "\PointingDynamicsDataset"+testperson

def reset_time(time, target):
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


def read_CSV_File(file_path):
    dataframe = pandas.read_csv(file_path,delimiter=',',skiprows=1,
                                header=None, usecols=(24, 28, 29, 31, 32, 33))
    dataset = dataframe.values

    permanent_input = []
    target = []
    time = []
    sg_values = []
    sga = []

    permanent_input.extend(dataset[:, 2])
    time.extend(dataset[:, 0])
    target.extend(dataset[:, 1])
    sga.extend(dataset[:, 5])
    sg_values.extend(dataset[:, 3:5])

    # reset time to 0 when target changes
    time = reset_time(time, target)

    # starting sg values
    starting_values = [sg_values[0][0], sg_values[0][1], sga[0], len(sga)]

    #  Reshape
    permanent_input = np.reshape(permanent_input, (-1, 1))
    time = np.reshape(time, (-1, 1))
    target = np.reshape(target, (-1, 1))

    # add last_sga and time to input
    permanent_input = np.concatenate((permanent_input, target), axis=1)
    permanent_input = np.concatenate((permanent_input, time), axis=1)

    return [permanent_input, sga, starting_values]


def load_model(name):
    # load json and create model
    json_file = open(name+".json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(name+".h5")
    print("Loaded model from disk")

    # Compile loaded model
    loaded_model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    return loaded_model


def show_results(time, sga_actual, sga_predict):
    # search for time reset positions
    time_reset_pos = []
    for i in range(len(time)):
        if time[i] == 0:
            time_reset_pos.append(i)

    # for every time period between time resets
    # show comparison of the sga predictions and actual sga values
    for i in range(len(time_reset_pos)):
        if time_reset_pos[i] == time_reset_pos[-1]:
            predicted_sga_period = np.array(sga_predict[time_reset_pos[i]:-1])
            actual_sga_period = np.array(sga_actual[time_reset_pos[i]:-1])
            time_period = np.array(time[time_reset_pos[i]:-1])
        else:

            predicted_sga_period = np.array(sga_predict[time_reset_pos[i]:time_reset_pos[i + 1] - 1])
            actual_sga_period = np.array(sga_actual[time_reset_pos[i]:time_reset_pos[i + 1] - 1])
            time_period = np.array(time[time_reset_pos[i]:time_reset_pos[i + 1] - 1])

        plt.plot(time_period, predicted_sga_period, time_period, actual_sga_period)
        plt.title('SGA Prediction')
        plt.ylabel('sga in m/s^2')
        plt.xlabel('time in s')
        plt.legend(['predicted', 'actual'], loc='upper left')
        plt.show()


perm_input = []
sga_actual = []
starting_sg_values = []

# read the data files
for file in os.listdir(dir_path):
    print(file)
    file_path = os.path.join(dir_path, file)

    data_fit = read_CSV_File(file_path)
    perm_input.extend(data_fit[0])
    sga_actual.extend(data_fit[1])
    starting_sg_values.append(data_fit[2])

perm_input = np.reshape(perm_input, (-1, 3))
sga_fit = np.reshape(sga_actual, (-1, 1))

# load already trained model
loaded_model = load_model(model_name)

# Predict recursive with new Data
sga_predict = []
starting_values_counter = 0
sga_predict = []
new_data_start_pos = []

# calculate starting positions of new data files
for i in range(len(starting_sg_values)):
    if i != 0:
        starting_sg_values[i][3] += starting_sg_values[i-1][3]
        new_data_start_pos.append(starting_sg_values[i][3])

# calculate new Data and feed it to the loaded NN
for i in range(len(sga_fit)):
    input_predict = []

    # first input of any data file is given
    if i == 0 or i in new_data_start_pos:
        sgv = starting_sg_values[starting_values_counter][1]
        sgy = starting_sg_values[starting_values_counter][0]
        sga = starting_sg_values[starting_values_counter][2]
        starting_values_counter += 1

    # else calculate sga = output of last prediction
    # sgy and sgv are calculated with last sga value
    # 0.002 is one timestep
    else:
        sgv = sgv + sga_predict[i-1]*0.002
        sgy = sgy + sgv * 0.002
        sga = sga_predict[i - 1]

    # add all values together
    input_predict.append(sga)
    input_predict.append(sgv)
    input_predict.append(sgy)
    input_predict.extend(perm_input[i])

    input_predict = np.reshape(input_predict, (-1, 6))

    # Reshape into 3D
    input_predict = input_predict.reshape((input_predict.shape[0],
                                           1, input_predict.shape[1]))

    # predict next value
    print(len(sga_fit)-i)
    sga_predict.extend(loaded_model.predict(input_predict)[0])

show_results(perm_input[:,2], sga_actual, np.reshape(sga_predict, (-1, 1)))
