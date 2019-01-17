import pickle
from collections import defaultdict
import sys

readfolder = '/mnt/datadrive/Dropbox/Data/spine360/numbernames/'
resultfolder = '/mnt/datadrive/Dropbox/Data/spine360/by-Beam/'

readfolder = '/mnt/datadrive/Dropbox/Data/lung360/numbernames/'
resultfolder = '/mnt/datadrive/Dropbox/Data/lung360/by-Data/'

readfolder = '/mnt/datadrive/Dropbox/Data/brain360/numbernames/'
resultfolder = '/mnt/datadrive/Dropbox/Data/brain360/by-Beam/'
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
voxels = set()
voxelsnoNull = set()
beamletset = set()
for mynum in range(0,359,2):
    PIK = readfolder + str(mynum) + ".pickle"
    items = load(PIK)
    di = defaultdict(list) #indices
    dd = defaultdict(list) #doses
    for itemlist in items:
        for dp in itemlist:
            voxels.add(dp[0])
            beamletset.add(dp[1].Index)
            if dp[1].Dose > 0:
                voxelsnoNull.add(dp[0])
                di[dp[0]].append(dp[1].Index)
                dd[dp[0]].append(dp[1].Dose)
    output = open(resultfolder + 'twolists' + str(mynum) + '.pickle', 'ab')
    pickle.dump([di,dd], output)
    output.close()
    d = None

sys.exit()

