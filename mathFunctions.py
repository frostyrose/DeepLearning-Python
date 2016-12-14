import scipy.stats as Stats
import DataUtility as du
import urllib3
import urllib
import numpy as np


def downloadFileFromJavaFTPServer(dataFile):
    # Should Open and Read in the Data File
    url = "ftp://Cameron@127.0.0.1:21/" + dataFile #'D:/Users/Cameron/Documents/GitHub/ALI-DataDumper/' + dataFile
    print "Grabbing File at: " + url
    urllib.urlretrieve(url, 'Resources/'+dataFile)


    return "Resources/" + dataFile #return location of locally saved file

def chiSquaredTest(dataFile):
    filename = "Resources/Test-Data-Set-1480976936524.csv"#downloadFileFromJavaFTPServer(dataFile) #where datafile will be the location of the data on the external machine
    f_obs, headers = du.loadFloatCSVwithHeaders(filename)
    print f_obs
    result_statistic, pvals = Stats.chisquare(f_obs)

    def writeOut(rStat, pVal, filename,headers=[]):

        with open(filename, 'w') as f:
            if len(headers)!=0:
                for i in range(len(headers)-1):
                    f.write(str(headers[i]) + ',')
                f.write(str(headers[-1])+'\n')
                for j in range(len(rStat)-1):
                    f.write(str(rStat[j]) + ',')
                f.write(str(rStat[-1]) + '\n')
                for k in range(len(pVal)-1):
                    f.write(str(pVal[k]) + ',')
                f.write(str(pVal[-1]) + '\n')
        f.close()

    newFileName = "/home/ali/Results/" + dataFile #using dataFile again as a convenient filename
    writeOut(result_statistic, pvals, newFileName, headers)


    return newFileName #passing the file name back up so that the main Flask code can handle sending the file back to Java

def anova(dataFile):
    '''
    Takes in data from a CSV as such:
    HEADER1 | HEADER2 | HEADER3 | ... | HEADERN
    Val1a   | Val2a   | Val3a   | ... | ValNa
    Val1b   | Val2b   | Val3b   | ... | ValNb
    .....   | .....   | .....   | ... | ...
    Val1x   | Val2x   | Val3x   | ... | ValNx

    ANOVA will compare the means of all headers in the data file.
    '''
    filename = "Resources/Test-Data-Set-1480976936524.csv"#downloadFileFromJavaFTPServer(dataFile) #where datafile will be the location of the data on the external machine
    dataValues, headers = du.loadFloatCSVwithHeaders(filename)

    #transpose data here, such that each row will be the data points for each heading
    #this simplifies the passing of the data in to Stats.f_oneway()
    np.transpose(dataValues)
    fStat, pValue = Stats.f_oneway(*dataValues)

    def writeOut(f, p, filename):

        with open(filename, 'w') as f:
            f.write("FValue" + ',')
            f.write("PValue" + '\n')
            f.write(str(f) + ',')
            f.write(str(p) + '\n' )
        f.close()

    newFileName = "/home/ali/Results/" + dataFile #using dataFile again as a convenient filename
    writeOut(fStat, pValue, newFileName)
    #output will be in the form:
    # FValue | pValue
    # Fstat  | pval

    return newFileName #passing the file name back up so that the main Flask code can handle sending the file back to Java

def ttest(dataFile, testVer):
    filename = "Resources/Test-Data-Set-1480976936524.csv"#downloadFileFromJavaFTPServer(dataFile) #where datafile will be the location of the data on the external machine
    dataValues, headers = du.loadFloatCSVwithHeaders(filename)

    #transpose data here, such that each row will be the data points for each heading
    np.transpose(dataValues)
    if not len(dataValues) == 2:
        print "ERROR-Input is invalid"
        return filename
    sample1 = dataValues[0]
    sample2 = dataValues[1]

    if testVer == 1:
        tStat, pValue = Stats.ttest_ind(sample1, sample2)
    if testVer == 2:
        tStat, pValue = Stats.ttest_ind(sample1, sample2, equal_var=False)
    if testVer == 3:
        tStat, pValue = Stats.ttest_rel(sample1, sample2)


    def writeOut(t, p, filename):

        with open(filename, 'w') as f:
            f.write("TStat" + ',')
            f.write("PValue" + '\n')
            f.write(str(t) + ',')
            f.write(str(p) + '\n' )
        f.close()

    newFileName = "/home/ali/Results/" + dataFile #using dataFile again as a convenient filename
    writeOut(tStat, pValue, newFileName)

    return newFileName #passing the file name back up so that the main Flask code can handle sending the file back to Java
