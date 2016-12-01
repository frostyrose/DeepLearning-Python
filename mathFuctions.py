import scipy.stats as Stats
import DataUtility as du
import urllib3


def readInFile(dataFile):
    # Should Open and Read in the Data File
    #url = 'D:/Users/Cameron/Documents/GitHub/ALI-DataDumper/' + dataFile
    url = "https://www.google.com/"
    connection_pool = urllib3.PoolManager()
    resp = connection_pool.request('GET', url)
    f = open("resources/" + dataFile, 'wb')
    f.write(resp.data)
    f.close()
    resp.release_conn()

    return "resources/" + dataFile #return location of locally saved file

def chiSquaredTest(dataFile):
    filename = readInFile(dataFile) #where datafile will be the location of the data on the external machine
    f_obs, headers = du.loadCSVwithHeaders(filename)

    result_statistic, pvals = Stats.chisquare(f_obs)

    newFileName = "Placeholder" #we can come up with a more formal scheme later
    du.writetoCSV(result_statistic, newFileName, headers)


    return newFileName #passing the file name back up so that the main Flask code can handle sending the file back to Java
