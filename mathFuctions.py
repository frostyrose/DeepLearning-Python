#import Scipy.stats as Stats
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

    # data,headers = du.loadCSVwithHeaders(dataFile)
    return

def chiSquaredTest(dataFile):
    f_obs = readInFile(dataFile)
    return #Stats.chisquare(f_obs)