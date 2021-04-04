import os

class Config:
    def __init__(self, mode = "LSTM", nfilt = 26, nfeat = 13, nfft = 512, rate = 16000): #MODE CHANGE!!!
        self.mode = mode
        self.nfilt = nfilt
        self.nfeat = nfeat
        self.nfft = nfft
        self.rate = rate
        self.step = int(rate/10)
        self.model_path = os.path.join("models", mode + ".model") # saving path of models
        self.p_path = os.path.join("pickles", mode + ".p") #saving path of pickels