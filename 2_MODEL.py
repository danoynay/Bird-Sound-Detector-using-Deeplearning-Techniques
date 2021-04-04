import os
import pandas as pandas
import matplotlib.pyplot as pyplot
import numpy as numpy
import pickle
from scipy.io import wavfile
from tqdm import tqdm
from python_speech_features import mfcc
from CONFIGURATION import Config
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, LSTM
from tensorflow.keras.layers import Dropout, Dense, TimeDistributed
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint

#---------------------------
#check the pickle if there is an existing model
def check_data():
    if os.path.isfile(config.p_path):
        print("Loading existing data for {} model" .format(config.mode))
        with open(config.p_path, "rb") as handle:
            temporary = pickle.load(handle)
            return temporary
    else:
        return None

#----------------------------
#Generate the data to be feed into the model
def build_rand_feat(): #for prediction
    temporary = check_data()  #check data if wala pa nag exist and ipa run
    if temporary:
        return temporary.data[0], temporary.data[1] #return tuple
    X = []
    y = []
    _min, _max = float("inf"), -float("inf")
    for _ in tqdm(range(n_samples)):
        try:
            random_class = numpy.random.choice(class_dist.index, p = prob_dist) #random class
            file = numpy.random.choice(dataframe[dataframe.label == random_class].index) #do 
            rate, wav = wavfile.read("clean/" + file)
            label = dataframe.at[file, "label"]
            random_index = numpy.random.randint(0, wav.shape[0] - config.step)
            sample = wav[random_index: random_index + config.step]
            X_sample = mfcc(sample, rate, numcep = config.nfeat, nfilt = config.nfilt, nfft = config.nfft)
            _min = min(numpy.amin(X_sample), _min)
            _max = max(numpy.amax(X_sample), _max)
            X.append(X_sample) #changed for prediction
            y.append(classes.index(label))
        except ValueError as e:
            print(e)
    config.min = _min
    config.max = _max
    X, y = numpy.array(X), numpy.array(y)
    X = (X - _min) / (_max - _min)
    if config.mode == "CNN":
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    elif config.mode == "LSTM":
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2])
    y = to_categorical(y, num_classes = 10)
    config.data = (X, y)

    #once store the data within config save entire pickles
    with open(config.p_path, "wb") as handle:
        pickle.dump(config, handle, protocol = pickle.HIGHEST_PROTOCOL)
    return X, y

#-----------------------------
#CNN function
def get_conv_model():
    #CNN layers
    model = Sequential()
    model.add(Conv2D(16, (3, 3), activation = "relu", strides = (1, 1),
                     padding = "same", input_shape = input_shape))
    model.add(Conv2D(32, (3, 3), activation = "relu", strides = (1, 1),
                     padding = "same"))
    model.add(Conv2D(64, (3, 3), activation = "relu", strides = (1, 1),
                 padding = "same"))
    model.add(Conv2D(128, (3, 3), activation = "relu", strides = (1, 1),
             padding = "same"))
    #Pooling layer
    model.add(MaxPool2D((2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128, activation = "relu"))
    model.add(Dense(64, activation = "relu"))
    model.add(Dense(10, activation = "softmax"))
    model.summary()
    model.compile(loss = "categorical_crossentropy",
                  optimizer = "adam",
                  metrics = ["accuracy"])
    return model

#RNN function (LSTM)
def get_recurrent_model():
    #shape of data for RNN is (n, time, feat)
    model = Sequential()
    model.add(LSTM(128, return_sequences = True, input_shape = input_shape))
    model.add(LSTM(128, return_sequences = True))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(64, activation = "relu")))
    model.add(TimeDistributed(Dense(32, activation = "relu")))
    model.add(TimeDistributed(Dense(16, activation = "relu")))
    model.add(TimeDistributed(Dense(8, activation = "relu")))
    model.add(Flatten())
    model.add(Dense(10, activation = "softmax"))
    model.summary()
    model.compile(loss = "categorical_crossentropy",
                  optimizer = "adam",
                  metrics = ["accuracy"])
    return model


#-----------------------------
#Open data frame
dataframe = pandas.read_csv('BIRDS.csv')
dataframe.set_index('filename', inplace = True)

for file in dataframe.index:
    rate, signal = wavfile.read('clean/' + file)
    dataframe.at[file, 'length'] = signal.shape[0]/rate

classes = list(numpy.unique(dataframe.label))
class_dist = dataframe.groupby(['label'])['length'].mean()

#-------------------------------
n_samples = int(dataframe["length"].sum() / 0.5) #NUMBER OF SAMPLES devide 
prob_dist = class_dist / class_dist.sum() #Probability Distribution
choices = numpy.random.choice(class_dist.index, p = prob_dist)


#----------------------------
#SETTING of Nueral Network 
config = Config(mode = "LSTM")


if config.mode == "CNN":
    X, y = build_rand_feat()
    y_flat = numpy.argmax(y, axis = 1)
    input_shape = (X.shape[1], X.shape[2], 1)
    model = get_conv_model()

elif config.mode == "LSTM":
    X, y = build_rand_feat()
    y_flat = numpy.argmax(y, axis = 1)
    input_shape = (X.shape[1], X.shape[2])
    model = get_recurrent_model()

class_weight = compute_class_weight("balanced", numpy.unique(y_flat), y_flat)

#------------------------------
#checking point'
checkpoint = ModelCheckpoint(config.model_path,
                             monitor = "validation accuracy",
                             verbose = 1,
                             mode = "max",
                             save_best_only = True,
                             save_weights_only = False,
                             period = 1)

#-------------------------------
#model parameters
history = model.fit(X, y,
          epochs = 1,
          batch_size = 128,
          shuffle = True,
          validation_split = 0.3, #30% of the data for training
          callbacks = [checkpoint],
          )

model.save(config.model_path)

# summarize history for accuracy
pyplot.plot(history.history['accuracy'])
pyplot.plot(history.history['val_accuracy'])
pyplot.title('model accuracy')
pyplot.ylabel('accuracy')
pyplot.xlabel('epoch')
pyplot.legend(['train', 'test'], loc='upper left')
pyplot.show()

# summarize history for loss
pyplot.plot(history.history['loss'])
pyplot.plot(history.history['val_loss'])
pyplot.title('model loss')
pyplot.ylabel('loss')
pyplot.xlabel('epoch')
pyplot.legend(['train', 'test'], loc='upper left')
pyplot.show()