#import urllib3
from RNN import *
from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/RNN/<filename>')
def setFile(filename):
    test("resources/"+filename,"RNN.pickle")
    return "Accessing File: %s" % filename