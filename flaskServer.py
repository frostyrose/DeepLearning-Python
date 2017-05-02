#import urllib3
import RNN
import ApplyDetector
from mathFunctions import *
from flask import Flask

app = Flask(__name__)
hostname = "ali-aws.cs.wpi.edu"

@app.route('/')
def root_menu():
    return 'Welcome to Flask - Server is Running\n----------Settings----------\n  Hostname: ' + hostname


@app.route('/TEST/<pickleFile>/<dataFile>')
def runTest(pickleFile, dataFile):
    local_file = "/home/ubuntu/DeepLearning-Python/Dataset/" + dataFile
    pickleFile2 = pickleFile + ".pickle"
    print dataFile2
    print pickleFile
    if(pickleFile == "AFFECT"):
        confidence_table = ApplyDetector.apply_model(pickleFile2, local_file, "/home/ali/Results/"+filename)
        ApplyDetector.aggregate_estimates(confidence_table, "/home/ali/Results/" + dataFile)
    return ("Testing " + dataFile + " on the " + pickleFile + " model.")


@app.route('/TRAIN/<testType>/<filename>')
def runTraining(testType, filename):
    local_file = downloadFileFromJavaFTPServer(filename, hostname)
    if(testType == "AFFECT"):
        model = ApplyDetector.train_and_save_model("AFFECT.pickle", local_file, "/home/ali/Results/"+filename)
    return "Training on %s" % filename


@app.route('/UPDATE/<variableName>/<value>')
def updateVariableWithValue(variableName, value):
    s1 = ""
    s2 = ""
    if(variableName == "HOSTNAME"):
        s1 = "hostname"
        s2 = str(value)
        hostname = value
    else:
        return ("No Valid Variable Selected")
    return ("Variable associated with " + s1 + " updated to the value" + s2)


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
