import scipy.stats as Stats
import DataUtility as du
import urllib3
import urllib


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
