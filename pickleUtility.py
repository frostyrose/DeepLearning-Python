import pickle
import sys

def saveInstance(model,filename):
    pickle.dump(model, open(filename,"wb"), -1)

def loadInstance(filename):
    model = pickle.load(open(filename, "rb" ))
    return model