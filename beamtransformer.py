import pickle
from collections import defaultdict
import sys

myfolder = '/mnt/datadrive/Dropbox/Data/spine360/by-Beam/'
#myfolder = '/home/wilmer/Dropbox/Data/spine360/by-Beam/'
#data = pickle.load( open( myfolder + "358.pickle", "rb" ) )

class beamletdose:
    def __init__(self, a):
        self.Index = a.Index
        self.Dose = a.Dose

def load(filename):
    with open(filename, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break

for mynum in range(0,359,2):
    PIK = myfolder + str(mynum) + ".pickle"
    items = load(PIK)
    di = defaultdict(list) #indices
    dd = defaultdict(list) #doses
    for itemlist in items:
        for dp in itemlist:
            if dp[1].Dose > 0:
                di[dp[0]].append(dp[1].Index)
                dd[dp[0]].append(dp[1].Dose)
    output = open(myfolder + 'twolists' + str(mynum) + '.pickle', 'ab')
    pickle.dump([di,dd], output)
    output.close()
    d = None

sys.exit()

for mynum in range(0,359,2):
    PIK = myfolder + str(mynum) + ".pickle"
    items = load(PIK)
    d = defaultdict(list)
    for itemlist in items:
        for dp in itemlist:
            d[dp[0]].append(beamletdose(dp[1]))
    output = open(myfolder + 'defaultdict' + str(mynum) + '.pickle', 'ab')
    pickle.dump(d, output)
    output.close()
    d = None
#myitem = next(items)
