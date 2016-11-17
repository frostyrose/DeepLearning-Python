#import urllib3
from RNN import *
from mathFuctions import *
from flask import Flask
from RNN import RNN_GRU
app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'

# @app.route('/RNN/<filename>')
# def setFile(filename):
#     test("resources/"+filename,"RNN.pickle")
#     return "Accessing File: %s" % filename


@app.route('/TEST/<pickleFile>/<dataFile>')
def runTest(pickleFile, dataFile):
    dataFile = "resources/" + dataFile
    print dataFile
    print pickleFile
    test(dataFile, pickleFile)
    return ("Testing " + dataFile + " on the " + pickleFile + " model.")

@app.route('/TRAIN/<filename>')
def runTraining(filename):
    train_and_test("resources/"+filename)
    return "Training on %s" % filename

@app.route('/MATH/<testType>/<dataFile>')
def runMath(testType, dataFile):
    if str(testType) == "chisquared":
        chiSquaredTest(dataFile)
    # ADD ADDITIONAL CASES FOR ADDITIONAL TESTS
    else:
        return ("No Valid Test Selected")
    return ("Running " + testType + " on the " + dataFile + " data.")