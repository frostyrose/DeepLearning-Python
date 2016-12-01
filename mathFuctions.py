import scipy.stats as Stats
import DataUtility as du
import urllib3
import urllib


def downloadFileFromJavaFTPServer(dataFile):
    # Should Open and Read in the Data File
    url = "ftp://Cameron@127.0.0.1:21/" + dataFile #'D:/Users/Cameron/Documents/GitHub/ALI-DataDumper/' + dataFile
    print "Grabbing File at: " + url
    urllib.urlretrieve(url, 'resources/'+dataFile)


    return "resources/" + dataFile #return location of locally saved file

def chiSquaredTest(dataFile):
    filename = downloadFileFromJavaFTPServer(dataFile) #where datafile will be the location of the data on the external machine
    f_obs, headers = du.loadCSVwithHeaders(filename)

    result_statistic, pvals = Stats.chisquare(f_obs)

    newFileName = "results/" + "filename-placeholder" #we can come up with a more formal scheme later
    du.writetoCSV(result_statistic, newFileName, headers)

    return newFileName #passing the file name back up so that the main Flask code can handle sending the file back to Java
