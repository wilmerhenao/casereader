import pickle
import numpy as np

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
