import pickle
import numpy as np
import socket
import os
import dose_to_points_data_pb2
import matplotlib.pyplot as plt
import sys
from sklearn.linear_model import LinearRegression
import pandas as pd
from pandas.plotting import table

class timedata(object):
    def __init__(self):
        self.initialtime = time.time()
        self.lasttime = time.time()
        self.looptimes = list()
        self.readtime = np.inf

    def newloop(self):
        self.looptimes.append(time.time() - self.lasttime)
        self.lasttime = time.time()

    def readingtime(self):
        self.readtime  = time.time() - self.lasttime
        self.lasttime = time.time()

## This class contains a structure (region)
class structure(object):
    ## Static variable that keeps a tally of the total number of structures
    numStructures = 0
    numTargets = 0
    numOARs = 0
    def __init__(self, sthing):
        self.Id = sthing.Id
        self.pointsDistanceCM = sthing.pointsDistanceCM
        self.StartPointIndex = sthing.StartPointIndex
        self.EndPointIndex = sthing.EndPointIndex
        self.Size = self.EndPointIndex - self.StartPointIndex
        self.isTarget = False
        self.isKilled = False
        # Identify targets
        alb = "PTV" in self.Id;
        ala = "GTV" in self.Id;
        alc = "CTV" in self.Id;
        if ( alb | ala  | alc):
            self.isTarget = True
        if self.isTarget:
            structure.numTargets = structure.numTargets + 1
            # NEVER CHANGE THIS! YOU MIGHT HAVE TROUBLE LATER
            self.threshold = 0.0
            self.overdoseCoeff = 0.0
            self.underdoseCoeff = 0.0
        else:
            structure.numOARs += 1
            self.threshold = 0.0
            self.overdoseCoeff = 0.0
            self.underdoseCoeff = 0.0
        structure.numStructures += 1

class beam(object):
    numBeams = 0
    M = None
    N = None
    # Initialize left and right leaf limits for all
    leftEdge = -1
    rightEdge = None
    leftEdgeFract = None
    rightEdgeFract = None
    JawX1 = None
    JawX2 = None
    JawY1 = None
    JawY2 = None
    def __init__(self, sthing):
        # Update the static counter and other variables
        self.location = int(int(sthing.Id)/2)
        self.Id = self.location
        self.angle = 2 * self.Id   # Notice that this one changes per beam
        self.JawX1 = sthing.JawX1
        self.JawX2 = sthing.JawX2
        self.JawY1 = sthing.JawY1
        self.JawY2 = sthing.JawY2
        # Update local variables
        self.StartBeamletIndex = sthing.StartBeamletIndex
        self.EndBeamletIndex = sthing.EndBeamletIndex
        self.beamletsPerBeam = self.EndBeamletIndex - self.StartBeamletIndex
        self.beamletsInBeam = self.beamletsPerBeam
        self.llist = [self.leftEdge] * self.M # One position more or less after the edges given by XStart, YStart in beamlets
        self.rlist = [self.rightEdge] * self.M
        self.KellyMeasure = 0
        self.Perimeter = 0
        self.Area = 0
        beam.numBeams += 1

class voxel(object):
    numVoxels = 0
    def __init__(self, sthing):
        voxel.numVoxels += 1
        self.Index = sthing.Index
        self.StructureId = sthing.StructureId
        self.X = sthing.X
        self.Y = sthing.Y
        self.Z = sthing.Z

class beamlet(object):
    numBeamlets = 0
    XSize = None
    YSize = None
    def __init__(self, sthing):
        beamlet.numBeamlets += 1
        self.Index = sthing.Index
        self.XStart = sthing.XStart + 0 # Make everything start at zero
        self.YStart = sthing.YStart
        beamlet.XSize = sthing.XSize
        beamlet.YSize = sthing.YSize
        self.belongsToBeam = None

## apertureList is a class definition of locs and angles that is always sorted.
# Its attributes are loc which is the numeric location; It has range 0 to 180 for
# the spine case; Angle is the numeric angle in degrees; It ranges from 0 to 360 degrees
# apertureList should be sorted in ascending order everytime you add a new element; User CAN make this safe assumption
class apertureList:
    ## constructor initializes empty lists
    def __init__(self):
        ## Location in index range(0,numbeams)
        self.loc = []
        ## Angles ranges from 0 to 360
        self.angle = []
    ## Insert a new angle in the list of angles to analyse.
    # Gets angle information and inserts location and angle
    # In the end it sorts the list in increasing order
    def insertAngle(self, i, aperangle):
        self.angle.append(aperangle)
        self.loc.append(i)
        # Sort the angle list in ascending order
        self.loc.sort()
        self.angle.sort()
    ## Removes the index and its corresponding angle from the list.
    # Notice that it only removes the first occurrence; but if you have done everything correctly this should never
    # be a problem
    def removeIndex(self, index):
        toremove = [i for i,x in enumerate(self.loc) if x == index]
        self.loc.pop(toremove[0]) # Notice that it removes the first entry
        self.angle.pop(toremove[0])
    ## Looks for the angle and removes the index and the angle corresponding to it from the list
    def removeAngle(self, tangl):
        toremove = [i for i,x in enumerate(self.angle) if x == tangl]
        self.loc.pop(toremove[0])
        self.angle.pop(toremove[0])
    ## Overloads parenthesis operator in order to fetch the ANGLE given an index.
    # Returns the angle at the ith location given by the index.
    # First Find the location of that index in the series of loc
    # Notice that this function overloads the parenthesis operator for elements of this class.
    def __call__(self, index):
        toreturn = [i for i,x in enumerate(self.loc) if x == index]
        return(self.angle[toreturn[0]])
    ## Overload the bracket operator to achieve the index from the angle
    def __getitem__(self, tangl):
        toreturn = [i for i,x in enumerate(self.angle) if x == tangl]
        return(self.loc[toreturn[0]])
    ## Returns the length of this instantiation without the need to pass parameters.
    def len(self):
        return(len(self.loc))
    ## Returns True if the list is empty; otherwise returns False.
    def isEmpty(self):
        if 0 == len(self.loc):
            return(True)
        else:
            return(False)

def calcObjValue(currentDose, quadHelperThresh, quadHelperOver, quadHelperUnder):
    oDoseObj = currentDose - quadHelperThresh
    oDoseObj = (oDoseObj > 0) * oDoseObj
    oDoseObj = oDoseObj * oDoseObj * quadHelperOver
    uDoseObj = quadHelperThresh - currentDose
    uDoseObj = (uDoseObj > 0) * uDoseObj
    uDoseObj = uDoseObj * uDoseObj * quadHelperUnder
    objectiveValue = sum(oDoseObj + uDoseObj)
    return(objectiveValue)

def calcObjValueByStructure(currentDose, quadHelperThresh, quadHelperOver, quadHelperUnder, mask, structures):
    objs = []
    maxdosesbrain = [50,55,55,60,62,54,54,56,56,54,56,40,40,10,10,60,40,40]
    fractionAboveThreshold = np.zeros(18)
    for structure in np.unique(mask):
        thisStructureDose = currentDose[np.where(structure == mask)]
        objs.append(calcObjValue(thisStructureDose, quadHelperThresh[np.where(structure == mask)], quadHelperOver[np.where(structure == mask)], quadHelperUnder[np.where(structure == mask)]))
        fractionAboveThreshold[int(np.log2(structure))] = sum(thisStructureDose > maxdosesbrain[int(np.log2(structure))])/len(thisStructureDose)
    print(objs)
    names = [structures[i].Id for i in range(len(structures))]
    d = dict()
    for i in range(len(names)):
        d[names[i]] = objs[i]
    print('dictionary of structures:', d)
    print('names:', names)
    print('contributions:', objs)
    print('proportion of voxels violating the threshold:', fractionAboveThreshold)
    return(sum(objs))

def functionAndPenalties (beamList, ds):
    averageNW = 0.0
    averageW = 0.0
    for i in range(len(beamList)):
        averageNW += beamList[i].KellyMeasure
        averageW += beamList[i].KellyMeasure * ds[1][i]
    print('function value and penalty measures', ds[17], averageNW, averageW)
    # Running times
    if len(ds) > 21:
        #print('total running time:' - (int(ds[21].initialtime) - int(ds[21].lasttime))/3600)
        print('looptimes list in hours:', [int(i)/3600 for i in ds[21].looptimes ])
        print('reading time:', int(ds[21].readtime) / 3600)

def get_files_by_file_size(dirname, reverse = False):
    """ Return list of file paths in directory sorted by file size """
    # Get list of files
    filepaths = []
    for basename in os.listdir(dirname):
        filename = os.path.join(dirname, basename)
        if os.path.isfile(filename):
            filepaths.append(filename)
    # Re-populate list with filename, size tuples
    for i in range(len(filepaths)):
        filepaths[i] = (filepaths[i], os.path.getsize(filepaths[i]))
    # Sort list by file size
    # If reverse=True sort from largest to smallest
    # If reverse=False sort from smallest to largest
    filepaths.sort(key = lambda filename: filename[1], reverse = reverse)
    # Re-populate list with just filenames
    for i in range(len(filepaths)):
        filepaths[i] = filepaths[i][0]
    return(filepaths)

# Fetch the information about beamlet edges:

def individualAnalizer(caseis):
    # What computer am I reading from?
    datalocation = '~'
    if 'radiation-math' == socket.gethostname(): # LAB
        datalocation = "/mnt/fastdata/Data/" + caseis + "/by-Beam/"
        dropbox = "/mnt/datadrive/Dropbox"
        cutter = 44
        if "lung360" == caseis:
            cutter = 43
    elif 'sharkpool' == socket.gethostname(): # MY HOUSE
        datalocation = "/home/wilmer/Dropbox/Data/spine360/by-Beam/"
        dropbox = "/home/wilmer/Dropbox"
        cutter = 51
    elif 'DESKTOP-EA1PG8V' == socket.gethostname(): # MY HOUSE
        datalocation = "C:/Users/wihen/Data/"+ caseis + "/by-Beam/"
        dropbox = "D:/Dropbox"
        cutter = 45
        numcores = 11
        if "lung360" ==  caseis:
            cutter = 44
    elif ('arc-ts.umich.edu' == socket.gethostname().split('.', 1)[-1]): # FLUX
        datalocation = "/scratch/engin_flux/wilmer/" + caseis + "/by-Beam/"
        dropbox = "/home/wilmer/Dropbox"
        cutter = 52
        if "lung360" ==  caseis:
            cutter = 51
    else:
        datalocation = "/home/wilmer/Dropbox/Data/spine360/by-Beam/" # MY LAPTOP
        dropbox = "/home/wilmer/Dropbox"

    datafiles = get_files_by_file_size(datalocation)
    # The first file will contain all the structure data, the rest will contain pointodoses.
    dpdata = dose_to_points_data_pb2.DoseToPointsData()
    print('datafiles:', datafiles)
    f = open(datafiles.pop(0), "rb")
    dpdata.ParseFromString(f.read())
    f.close()
    datafiles.sort()

    XSize = dpdata.Beamlets[0].XSize
    YSize = dpdata.Beamlets[0].YSize
    XSize = 1.0
    YSize = 1.0
    # Read all relevant files for a particular case
    datafiles = get_files_by_file_size('./outputGraphics/')
    matching = [s for s in datafiles if 'outputGraphics/allbeamshapesbefore-save-' +  caseis + '-' in s]
    print('Areas will be calculated in squared milimeters. Same as perimeter in milimeters')
    KellyMeasures = dict()
    MyMeasures = dict()
    if caseis == 'lung360':
        cutter = 50
    else:
        cutter = 51
    mylegends = list()
    for PIK in matching:
        mylegends.append(float(PIK[cutter:-7]))

    print(matching)
    matching = [x for _,x in sorted(zip(mylegends, matching))]
    print(matching)

    mylegends = list()
    pendients = list()
    for myColor, PIK in enumerate(matching):
        print(PIK)
        Cval = float(PIK[cutter:-7])
        mylegends.append(Cval)
        MLCtrip = pickle.load(open(PIK, "rb"))
        KellyMeasures[PIK] = np.zeros(len(MLCtrip))
        MyMeasures[PIK] = np.zeros(len(MLCtrip))
        Perimeters = np.zeros(len(MLCtrip))
        Areas = np.zeros(len(MLCtrip))
        for i, thisbeam in enumerate(MLCtrip):
            l = thisbeam[0]
            r = thisbeam[1]
            Perimeter = (r[0] - l[0])
            Area = 0.0
            for n in range(1, len(l)):
                Area += (r[n] - l[n]) * YSize
                Perimeter += (np.abs(l[n] - l[n-1]) + np.abs(r[n] - r[n-1]) - 2 * np.maximum(0, l[n-1] - r[n]) - 2 * np.maximum(0, l[n] - r[n - 1])) * XSize
            Perimeter += (r[len(r)-1] - l[len(l)-1]) * XSize + np.sign(r[len(r)-1] - l[len(l)-1])
            Kellymeasure = Perimeter / Area
            Mymeasure = 1.0 * Perimeter - 0.55 * Area
            KellyMeasures[PIK][i] = Kellymeasure
            MyMeasures[PIK][i] = Mymeasure
            Perimeters[i] = Perimeter
            Areas[i] = Area
        lm = LinearRegression(fit_intercept=False)
        y = KellyMeasures[PIK] - Perimeters
        y = np.reshape(y, (len(y), 1))
        Areas = np.reshape(Areas, (len(y), 1))
        reg = lm.fit(-Areas, y)
        print(reg, reg.coef_)
        pendients.append((Cval, reg.coef_[0]))
        #print(reg.coef_)

        if mylegends[-1] == 1.0 or mylegends[-1] == 0.001:
            plt.scatter(MyMeasures[PIK], KellyMeasures[PIK], s = 13.0, marker = 'v')
        elif mylegends[-1] == 0.0:
            plt.scatter(MyMeasures[PIK], KellyMeasures[PIK], s=10.0, marker='o')
        else:
            plt.scatter(MyMeasures[PIK], KellyMeasures[PIK], s=1.0)
        plt.title('Comparison of measures for case: ' + caseis)
        plt.xlabel('Our Measure')
        plt.ylabel("Younge's Measure")
        plt.legend(mylegends)
    plt.savefig('outputGraphics/ComparisonOfMeasuresForCase' + caseis + '.png')
    #plt.show()
    return(pendients)

#caseis = "spine360"
#caseis = "lung360"
#caseis = "brain360"
#caseis = "braiF360"
cases = ("spine360", "lung360", "brain360", "braiF360")
coefficients = list()
for thiscase in cases:
    coefficients.append((thiscase, individualAnalizer(thiscase)))

for i in coefficients:
    df = pd.DataFrame(columns=['C_Value', 'xi'])
    for j in range(len(i[1])):
        df.loc[j] = [i[1][j][0], i[1][j][1]]
    print('generate latex code to paste in the document:')
    print("\\begin{table}")
    print('\\begin{tabular}{ll}')
    print('C & \\xi \\\\ ')
    print('\hline')
    for j in range(len(i[1])):
        print(i[1][j][0], '&', round(i[1][j][1], 2), ' \\\\ ')
    print('\end{tabular}')
    print('\end{table}')

print(coefficients)

sys.exit()

print('Analysis of Brain with an emphasis in OAR preservation')
ds = pickle.load(open("outputGraphics/pickle-C-brain360-0.0-save.dat", "rb"))
beamList = pickle.load(open("outputGraphics/beamList-save-brain3600.0.pickle", "rb"))
functionAndPenalties(beamList, ds)
print('Value calculated anew:', calcObjValueByStructure(ds[13], ds[18], ds[19], ds[20], ds[12], ds[16]))
ds = pickle.load(open("outputGraphics/pickle-C-brain360-5.0-save.dat", "rb"))
beamList = pickle.load(open("outputGraphics/beamList-save-brain3605.0.pickle", "rb"))
functionAndPenalties(beamList, ds)
print('Value calculated anew:', calcObjValueByStructure(ds[13], ds[18], ds[19], ds[20], ds[12], ds[16]))

print('Analysis of Brain with an emphasis in PTV destruction')
ds = pickle.load(open("outputGraphics/pickle-C-brain360-1e-05-save.dat", "rb"))
beamList = pickle.load(open("outputGraphics/beamList-save-brain3601e-05.pickle", "rb"))
functionAndPenalties(beamList, ds)

ds = pickle.load(open("outputGraphics/pickle-C-brain360-0.001-save.dat", "rb"))
beamList = pickle.load(open("outputGraphics/beamList-save-brain3600.001.pickle", "rb"))
functionAndPenalties(beamList, ds)

ds = pickle.load(open("outputGraphics/pickle-C-brain360-6-save.dat", "rb"))
beamList = pickle.load(open("outputGraphics/beamList-save-brain3606.pickle", "rb"))
functionAndPenalties(beamList, ds)
print('Detailed Analysis:', calcObjValueByStructure(ds[13], ds[18], ds[19], ds[20], ds[12], ds[16]))


print('Analysis of lung360')
ds = pickle.load(open("outputGraphics/pickle-C-lung360-0.0-save.dat", "rb"))
beamList = pickle.load(open("outputGraphics/beamList-save-lung3600.0.pickle", "rb"))
functionAndPenalties(beamList, ds)

ds = pickle.load(open("outputGraphics/pickle-C-lung360-1e-05-save.dat", "rb"))
beamList = pickle.load(open("outputGraphics/beamList-save-lung3601e-05.pickle", "rb"))
functionAndPenalties(beamList, ds)

ds = pickle.load(open("outputGraphics/pickle-C-lung360-0.0001-save.dat", "rb"))
beamList = pickle.load(open("outputGraphics/beamList-save-lung3600.0001.pickle", "rb"))
functionAndPenalties(beamList, ds)

ds = pickle.load(open("outputGraphics/pickle-C-lung360-0.001-save.dat", "rb"))
beamList = pickle.load(open("outputGraphics/beamList-save-lung3600.001.pickle", "rb"))
functionAndPenalties(beamList, ds)

ds = pickle.load(open("outputGraphics/pickle-C-lung360-1.0-save.dat", "rb"))
beamList = pickle.load(open("outputGraphics/beamList-save-lung3601.0.pickle", "rb"))
functionAndPenalties(beamList, ds)

print('Analysis of spine360')
print(0.0)
ds = pickle.load(open("outputGraphics/pickle-C-spine360-0.0-save.dat", "rb"))
beamList = pickle.load(open("outputGraphics/beamList-save-spine3600.0.pickle", "rb"))
functionAndPenalties(beamList, ds)

print(0.01)
ds = pickle.load(open("outputGraphics/pickle-C-spine360-0.01-save.dat", "rb"))
beamList = pickle.load(open("outputGraphics/beamList-save-spine3600.01.pickle", "rb"))
functionAndPenalties(beamList, ds)

print(0.1)
ds = pickle.load(open("outputGraphics/pickle-C-spine360-0.1-save.dat", "rb"))
beamList = pickle.load(open("outputGraphics/beamList-save-spine3600.1.pickle", "rb"))
functionAndPenalties(beamList, ds)

print(0.05)
ds = pickle.load(open("outputGraphics/pickle-C-spine360-0.05-save.dat", "rb"))
beamList = pickle.load(open("outputGraphics/beamList-save-spine3600.05.pickle", "rb"))
functionAndPenalties(beamList, ds)

print(0.5)
ds = pickle.load(open("outputGraphics/pickle-C-spine360-0.5-save.dat", "rb"))
beamList = pickle.load(open("outputGraphics/beamList-save-spine3600.5.pickle", "rb"))
functionAndPenalties(beamList, ds)

print(0.15)
ds = pickle.load(open("outputGraphics/pickle-C-spine360-0.15-save.dat", "rb"))
beamList = pickle.load(open("outputGraphics/beamList-save-spine3600.15.pickle", "rb"))
functionAndPenalties(beamList, ds)

print(0.25)
ds = pickle.load(open("outputGraphics/pickle-C-spine360-0.25-save.dat", "rb"))
beamList = pickle.load(open("outputGraphics/beamList-save-spine3600.25.pickle", "rb"))
functionAndPenalties(beamList, ds)

print(0.75)
ds = pickle.load(open("outputGraphics/pickle-C-spine360-0.75-save.dat", "rb"))
beamList = pickle.load(open("outputGraphics/beamList-save-spine3600.75.pickle", "rb"))
functionAndPenalties(beamList, ds)

print(1.0)
ds = pickle.load(open("outputGraphics/pickle-C-spine360-1.0-save.dat", "rb"))
beamList = pickle.load(open("outputGraphics/beamList-save-spine3601.0.pickle", "rb"))
functionAndPenalties(beamList, ds)

print(2.0)
ds = pickle.load(open("outputGraphics/pickle-C-spine360-2.0-save.dat", "rb"))
beamList = pickle.load(open("outputGraphics/beamList-save-spine3602.0.pickle", "rb"))
functionAndPenalties(beamList, ds)

print(2.5)
ds = pickle.load(open("outputGraphics/pickle-C-spine360-2.5-save.dat", "rb"))
beamList = pickle.load(open("outputGraphics/beamList-save-spine3602.5.pickle", "rb"))
functionAndPenalties(beamList, ds)

print(3.5)
ds = pickle.load(open("outputGraphics/pickle-C-spine360-3.5-save.dat", "rb"))
beamList = pickle.load(open("outputGraphics/beamList-save-spine3603.5.pickle", "rb"))
functionAndPenalties(beamList, ds)

print(10.0)
ds = pickle.load(open("outputGraphics/pickle-C-spine360-10.0-save.dat", "rb"))
beamList = pickle.load(open("outputGraphics/beamList-save-spine36010.0.pickle", "rb"))
functionAndPenalties(beamList, ds)

## Dan Polan's file:
import json
leaf = pickle.load(open("outputGraphics/allbeamshapesbefore-save-brain360-6.pickle", "rb"))
ds = pickle.load(open("outputGraphics/pickle-C-spine360-10.0-save.dat", "rb"))
mylist = list()
angle = 0.0
meterset = 0.0
for i in range(len(leaf)):
    mybeam = leaf[i]
    mbeam = [str(i) for i in mybeam[0]]
    nbeam = [str(i) for i in mybeam[1]]
    meterset += ds[13][i]
    mylist.append((str(2 * i), str(meterset), mbeam, nbeam))
print(mylist)

with open('outputGraphics/data.json', 'w') as outfile:
    json.dump(mylist, outfile)
