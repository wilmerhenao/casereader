import dose_to_points_data_pb2
import os
import gc
import numpy as np
from scipy import sparse
import socket
import pickle
import matplotlib.pyplot as plt

dropbox = "/mnt/datadrive/Dropbox"

# List of organs that will be used
structureListRestricted = [4,      8,    1,   7,     0   ]
#limits                    27,     30,    24,   36-47,  22
#names                     esof,   trach, prv2, tumor, chord
threshold  =              [0,      0,      0,      41,     0, ]
undercoeff =              [0.0,    0.0,   0.0,  10E-5,   0.0, ]
overcoeff  =              [10E-6, 10E-9, 10E-7,  10E-5,  10E-6]
numcores   = 8
testcase   = [i for i in range(0, 180, 18)]
fullcase   = [i for i in range(180)]
## If you activate this option. I will only analyze numcores apertures at a time
debugmode = False
easyread = False
refinementloops = False #This loop supercedes the eliminationPhase
eliminationPhase = False # Whether you want to eliminate redundant apertures at the end

gc.enable()
## Find out the variables according to the hostname
datalocation = '~'
if 'radiation-math' == socket.gethostname(): # LAB
    datalocation = "/mnt/fastdata/Data/spine360/by-Beam/"
    dropbox = "/mnt/datadrive/Dropbox"
elif 'sharkpool' == socket.gethostname(): # MY HOUSE
    datalocation = "/home/wilmer/Dropbox/Data/spine360/by-Beam/"
    dropbox = "/home/wilmer/Dropbox"
elif ('arc-ts.umich.edu' == socket.gethostname().split('.', 1)[1]): # FLUX
    datalocation = "/scratch/engin_flux/wilmer/spine360/by-Beam/"
    dropbox = "/home/wilmer/Dropbox"
else:
    datalocation = "/home/wilmer/Dropbox/Data/spine360/by-Beam/" # MY LAPTOP
    dropbox = "/home/wilmer/Dropbox"

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
            self.threshold = 42
            self.overdoseCoeff = 0.0000001
            self.underdoseCoeff = 1000.0
        else:
            structure.numOARs += 1
            self.threshold = 0.0
            self.overdoseCoeff = 0.000001
            self.underdoseCoeff = 0.0
        structure.numStructures += 1

class beam(object):
    numBeams = 0
    M = 8
    N = 14
    JawX1 = None
    JawX2 = None
    JawY1 = None
    JawY2 = None
    def __init__(self, sthing):
        # Update the static counter and other variables
        self.angle = 2 * beam.numBeams # Notice that this one changes per beam
        self.location = beam.numBeams
        beam.numBeams += 1
        self.JawX1 = sthing.JawX1
        self.JawX2 = sthing.JawX2
        self.JawY1 = sthing.JawY1
        self.JawY2 = sthing.JawY2
        # Update local variables
        self.StartBeamletIndex = sthing.StartBeamletIndex
        self.EndBeamletIndex = sthing.EndBeamletIndex
        self.beamletsPerBeam = self.EndBeamletIndex - self.StartBeamletIndex
        self.beamletsInBeam = self.beamletsPerBeam
        # Initialize left and right leaf limits for all
        self.leftEdge = -1
        self.rightEdge = 14
        self.llist = [self.leftEdge] * self.M # One position more or less after the edges given by XStart, YStart in beamlets
        self.rlist = [self.rightEdge] * self.M
        self.KellyMeasure = 0

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
        self.XStart = sthing.XStart + 7 # Make everything start at zero
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

class problemData():
    def __init__(self):
        self.kappa = []
        self.notinC = apertureList()
        self.caligraphicC = apertureList()
        self.currentDose = np.zeros(voxel.numVoxels, dtype = float)
        self.quadHelperThresh = np.empty(voxel.numVoxels, dtype = float)
        self.quadHelperOver = np.empty(voxel.numVoxels, dtype = float)
        self.quadHelperUnder = np.empty(voxel.numVoxels, dtype = float)
        self.maskValue = np.empty(voxel.numVoxels, dtype = float)
        self.setQuadHelpers(structureList, voxelList)
        self.openApertureMaps = [[] for i in range(beam.numBeams)]
        self.diagmakers = [[] for i in range(beam.numBeams)]
        self.strengths = [[] for i in range(beam.numBeams)]
        self.DlistT = None
        self.currentIntensities = np.zeros(beam.numBeams, dtype=float)
        self.voxelsUsed = None
        self.structuresUsed = None
        self.structureIndexUsed = None
        self.YU = None
        self.RU = None
        self.speedlim = None
        self.rmpres = None
        self.listIndexofAperturesRemovedEachStep = []
        self.listIndexofAperturesAddedEachStep = []

    def setQuadHelpers(self, sList, vList):
        for i in range(voxel.numVoxels):
            sid = structureDict[vList[i].StructureId] # Find structure of this particular voxel
            self.quadHelperThresh[i] = sList[sid].threshold
            self.quadHelperOver[i] = sList[sid].overdoseCoeff
            self.quadHelperUnder[i] = sList[sid].underdoseCoeff
            self.maskValue[i] = 2**sid

    def calcDose(self):
        self.currentDose = np.zeros(voxel.numVoxels, dtype = float)
        self.dZdK = np.matrix(np.zeros((voxel.numVoxels, beam.numBeams)))
        if self.caligraphicC.len() != 0:
            for i in self.caligraphicC.loc:
                self.currentDose += self.DlistT[i][:,self.openApertureMaps[i]] * sparse.diags(self.strengths[i]) * np.repeat(self.currentIntensities[i], len(self.openApertureMaps[i]), axis = 0)
                self.dZdK[:,i] = (self.DlistT[i] * sparse.diags(self.diagmakers[i], 0)).sum(axis=1)


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

#----------------------------------------------------------------------------------------
## Get the point to dose data in a sparse matrix
def getDmatrixPieces():
## Initialize vectors for voxel component, beamlet component and dose
    thiscase = fullcase
    if debugmode:
        thiscase = testcase

    # Get the ranges of the voxels that I am going to use and eliminate the rest
    myranges = []
    for i in structureListRestricted:
        myranges.append(range(structureList[i].StartPointIndex, structureList[i].EndPointIndex))
    ## Read the beams now.
    counter = 0
    d = dict()
    input = open('/mnt/fastdata/Data/spine360/by-Beam/twolists200.pickle', 'rb')
    indices, doses = pickle.load(input)
    input.close()

    input = open('/mnt/fastdata/Data/spine360/by-Beam/twolists350.pickle', 'rb')
    indicesb, dosesb = pickle.load(input)
    input.close()
    dlist = np.zeros(voxel.numVoxels, dtype = float)
    dlistb = np.zeros(voxel.numVoxels, dtype = float)
    alist = np.zeros(voxel.numVoxels, dtype = float)
    for k in indices.keys(): # Cycle in the voxels.
        for m in myranges:
            if k in m:
                dlist[k] = sum(doses[k])
                dlistb[k] = sum(dosesb[k])
                alist[k] = dlist[k] - dlistb[k]
    gc.collect()
    print('maxdiff: ', max(alist))
    PIK = 'd.dat'
    with open(PIK, "wb") as f:
        pickle.dump(d, f, pickle.HIGHEST_PROTOCOL)
    f.close()
    return(d)
#------------------------------------------------------------------------------------------------------------------

datafiles = get_files_by_file_size(datalocation)

# The first file will contain all the structure data, the rest will contain pointodoses.
dpdata = dose_to_points_data_pb2.DoseToPointsData()

# Start with reading structures, numvoxels and all that.
f = open(datafiles.pop(0), "rb")
dpdata.ParseFromString(f.read())
f.close()
datafiles.sort()
#----------------------------------------------------------------------------------------
# Get the data about the structures
numstructs = len(dpdata.Structures)
structureList = []
structureDict = {} # keys are the names of the structure and the value is the corresponding index location (integer)
print("Reading in structures")
for s in range(numstructs):
    print('Reading:', dpdata.Structures[s].Id)
    structureList.append(structure(dpdata.Structures[s]))
    structureDict[structureList[s].Id] = s
    print('This structure goes between voxels ', structureList[s].StartPointIndex, ' and ', structureList[s].EndPointIndex)
print('Number of structures:', structure.numStructures, '\nNumber of Targets:', structure.numTargets,
      '\nNumber of OARs', structure.numOARs)
# Manual modifications of targets
print('modifying penalization function according to the 5 structure case')
strcounter = 0
# Assign the values for the penalization function F(z)
for s in structureListRestricted:
    structureList[s].underdoseCoeff = undercoeff[strcounter]
    structureList[s].overdoseCoeff = overcoeff[strcounter]
    structureList[s].threshold = threshold[strcounter]
    strcounter += 1
#----------------------------------------------------------------------------------------
## Get the data about beamlets
numbeamlets = len(dpdata.Beamlets)
beamletList = []
print('Reading in beamlet data:')
for blt in range(numbeamlets):
    a = beamlet(dpdata.Beamlets[blt])
    beamletList.append(a)
    #print(beamletList[blt].XSize, beamletList[blt].YSize, beamletList[blt].XStart, beamletList[blt].YStart)
print('total number of beamlets read:', beamlet.numBeamlets)
#----------------------------------------------------------------------------------------
# Get the data about beams
numbeams = len(dpdata.Beams)
beamList = []
print('Reading in Beam Data:')
for b in range(numbeams):
    beamList.append(beam(dpdata.Beams[b]))
    for blt in range(dpdata.Beams[b].StartBeamletIndex, dpdata.Beams[b].EndBeamletIndex):
        beamletList[blt].belongsToBeam = b
print('There are a total of beams:', beam.numBeams)
print('beamlet data was updated so they point to their owner')
#----------------------------------------------------------------------------------------
## Get data about voxels.
numvoxels = len(dpdata.Points)
voxelList = []
print('Reading in Voxel data:')
for v in range(numvoxels):
    voxelList.append(voxel(dpdata.Points[v]))
print('total number of voxels read:', voxel.numVoxels)
## Free the memory
dpdata = None

/home/wilmer/Dropbox/Data/spine360/by-Structure/PsVM4m_182_270_2/ad0f4fc3-6f05-4f1a-83d9-3648721ce1d5

sys.exit()
d = getDmatrixPieces()
def treatDictionary(d):
    m = [[0 for x in range(180)] for y in range(180)]
    for i in d.keys():
        im = int(int(i[44:].split('.')[0])/2)
        for j in d.keys():
            jm = int(int(j[44:].split('.')[0])/2)
            m[im][jm] = np.intersect1d(d[i][:200], d[j][:200]).size
    return(np.array(m))

m = treatDictionary(d)
plt.imshow(m, cmap='hot', interpolation='nearest')
plt.savefig('heatmap180beams200.png')
PIK = 'm.dat'
with open(PIK, "wb") as f:
    pickle.dump(m, f, pickle.HIGHEST_PROTOCOL)
f.close()
gc.collect()
print(m)