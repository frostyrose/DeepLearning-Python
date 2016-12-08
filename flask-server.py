#import urllib3
import RNN
from mathFunctions import *
from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/TEST/<pickleFile>/<dataFile>')
def runTest(pickleFile, dataFile):
    dataFile2 = "Dataset/" + dataFile
    pickleFile2 = pickleFile + ".pickle"
    print dataFile2
    print pickleFile
    if(pickleFile == "RNN"):
        RNN.evaluate_model(dataFile2, pickleFile2, 200, 1, 20, step_size=0.005, balance_model=False, scale_output=True, variant="GRU")
    return ("Testing " + dataFile + " on the " + pickleFile + " model.")

@app.route('/TRAIN/<testType>/<filename>')
def runTraining(testType,filename):
    if(testType == "RNN"):
        RNN.train_and_evaluate_model(200, 1, 20, step_size=0.005, balance_model=False, scale_output=True, variant="GRU")
    return "Training on %s" % filename

@app.route('/MATH/<testType>/<dataFile>')
def runMath(testType, dataFile):
    if str(testType) == "chisquared":
        chiSquaredTest(dataFile)
    # ADD ADDITIONAL CASES FOR ADDITIONAL TESTS
    else:
        return ("No Valid Test Selected")
    return ("Running " + testType + " on the " + dataFile + " data.")
