import os
import pandas as pandas
import matplotlib.pyplot as pyplot
import numpy as numpy
import librosa
from tqdm import tqdm
from scipy.io import wavfile
from python_speech_features import mfcc

#CREATE SUBPLOT
def plot_signals(signals):
	figure, axes = pyplot.subplots( nrows = 2, ncols = 5, sharex = False, sharey = True, figsize = (20,5) )
	figure.suptitle('TIME SERIES', size = 16)
	i = 0
	for x in range(2):
		for y in range(5):
			axes[x,y].set_title(list(signals.keys())[i])
			axes[x,y].plot(list(signals.values())[i])
			axes[x,y].get_xaxis().set_visible(False)
			axes[x,y].get_yaxis().set_visible(False)
			i += 1

def plot_mfccs(mfccs):
    figure, axes = pyplot.subplots(nrows = 2, ncols = 5, sharex = False, sharey = True, figsize = (20,5))
    figure.suptitle('MEL FREQUENCY CEPSTRAL COEFFICIENTS', size = 16)
    i = 0
    for x in range(2):
        for y in range(5):
            axes[x,y].set_title(list(mfccs.keys())[i])
            axes[x,y].imshow(list(mfccs.values())[i],
                    cmap='hot', interpolation='nearest')
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1

#-----------------------
#SIGNAL ENVELOPE
def envelope(y, rate, threshold):
	mask = []
	y = pandas.Series(y).apply(numpy.abs)
	y_mean = y.rolling(window = int(rate / 10), min_periods = 1, center = True).mean()
	for mean in y_mean:
		if mean > threshold:
			mask.append(True)
		else:
			mask.append(False)
	return mask

#----------------------
#Read audio data
dataframe = pandas.read_csv("BIRDS.csv")
dataframe.set_index("filename", inplace = True)
#read individual file
for file in dataframe.index:
	rate, signal = wavfile.read("wavfiles/" + file)
	dataframe.at[file, "length"] = signal.shape[0] / rate #get length of each file
#group different labels together to create classes and access the lenth
classes = list(numpy.unique(dataframe.label))
class_distribution = dataframe.groupby(["label"])["length"].mean()	

#----------------------
#Create PIE CHART
figure, ax = pyplot.subplots()
ax.set_title("CLASS DISTRIBUTION", y = 1.08)
ax.pie(class_distribution, labels = class_distribution.index, autopct = '%.2f%%', shadow = False, startangle = 90)
ax.axis("equal")
pyplot.show()
dataframe.reset_index(inplace = True)

#DICTIONARIES
signals = {}
mfccs = {}

#VISUALIZATION OF SIGNAL from a 1 random example per class
for c in classes:
	wav_file = dataframe[dataframe.label == c].iloc[0, 0]
	signal, rate = librosa.load("wavfiles/" + wav_file, sr = 44100)
	mask = envelope(signal, rate, 0.0005)
	signal = signal[mask]

	signals[c] = signal 
	mel = mfcc(signal[:rate], rate, numcep = 13, nfilt = 26, nfft = 1103).T
	mfccs[c] = mel

plot_signals(signals)
pyplot.show()

plot_mfccs(mfccs)
pyplot.show()

#------------------------
#Transfer WAV to clean folder
if len(os.listdir("clean")) == 0:
	for file in tqdm(dataframe.filename):
		signal, rate = librosa.load("wavfiles/" + file, sr = 16000)
		mask = envelope(signal, rate, 0.0005)
		wavfile.write(filename = "clean/" + file, rate = rate, data = signal[mask])
