import pickle
import sys

def saveInstance(model,filename):
    pickle.dump(model, open("Models/"+filename,"wb"), -1)

def loadInstance(filename):
    model = pickle.load(open("Models/"+filename, "rb" ))
    return model
