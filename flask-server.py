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
    dataFile2 = "/home/ubuntu/DeepLearning-Python/Dataset/" + dataFile
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
    if str(testType) == "chisquared1":
        chiSquaredTest(dataFile)
    if str(testType) == "chisquared2":
        chi2(dataFile)
    if str(testType) == "anova":
        anova(dataFile)
    if str(testType) == "ttestInd":
        ttest(dataFile, 1)
    if str(testType) == "ttestIndWelch":
        ttest(dataFile, 2)
    if str(testType) == "ttestRel":
        ttest(dataFile, 3)

    else:
        return ("No Valid Test Selected")
    return ("Running " + testType + " on the " + dataFile + " data.")
