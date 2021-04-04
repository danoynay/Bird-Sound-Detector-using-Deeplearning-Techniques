import os
import pickle
import numpy as numpy
import pandas as pandas
from tqdm import tqdm
from scipy.io import wavfile
from python_speech_features import mfcc
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score
from tkinter import messagebox

def build_predictions(audio_dir):
	y_true = []
	y_pred = []
	filename_probability = {}
	
	print("Extracting features from audio")
	for filename in tqdm(os.listdir(audio_dir)):
		rate, wav = wavfile.read(os.path.join(audio_dir, filename))
		label = filename2class[filename]
		c = classes.index(label)
		y_probability = []

		for i in range(0, wav.shape[0] - config.step, config.step):
			sample = wav[i:i + config.step]
			x = mfcc(sample, rate, numcep = config.nfeat, nfilt = config.nfilt, nfft = config.nfft)
			x = (x - config.min) / (config.max - config.min)

			if config.mode == "CNN":
				x = x.reshape(1, x.shape[0], x.shape[1], 1)
			elif config.mode == "LSTM":
				x = numpy.expand_dims(x, axis = 0)
			y_hat = model.predict(x)
			y_probability.append(y_hat)
			y_pred.append(numpy.argmax(y_hat))
			y_true.append(c)

		filename_probability[filename] = numpy.mean(y_probability, axis = 0).flatten()

	return y_true, y_pred, filename_probability

#------------------------------
#OPEN FILE
dataframe = pandas.read_csv("TEST_BIRDS.csv")
classes = list(numpy.unique(dataframe.label))
filename2class = dict(zip(dataframe.filename, dataframe.label))
p_path = os.path.join("pickles", "LSTM.p")      #PICKLE PATH!!!

with open(p_path, "rb") as handle:
    config = pickle.load(handle)

model = load_model(config.model_path)

y_true, y_pred, filename_probability = build_predictions("clean_test")
acc_score = accuracy_score(y_true = y_true, y_pred = y_pred)

#configure the predict file i think
y_probs = []
for i, row in dataframe.iterrows():
	y_probability = filename_probability[row.filename]
	y_probs.append(y_probability)
	for c, p in zip(classes, y_probability):
		dataframe.at[i, c] = p

#create new column in the data frame
y_pred = [classes[numpy.argmax(y)] for y in y_probs]
dataframe["PREDICTION"] = y_pred
messagebox.showinfo("PREDICTION", y_pred)

dataframe.to_csv("TEST_PREDICT_LSTM.csv", index = False)