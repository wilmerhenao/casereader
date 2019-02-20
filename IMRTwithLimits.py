#! /opt/intel/intelpython35/bin/python3.5

import dose_to_points_data_pb2
import os
import time
import gc
import numpy as np
from scipy import sparse
from scipy.optimize import minimize
import socket
import pylab
import matplotlib.pyplot as plt
import pickle
import sys

# List of organs that will be used
caseis = "spine360"
#caseis = "lung360"
#caseis = "brain360"
strengthThreshold = 0.22 # This threshold only works for the spine case. It gets overwritten by all other cases and by
                        # the input arguments if provided

# let's not read arguments and overwrite the case
if len(sys.argv) == 3:
    strengthThreshold = float(sys.argv[2])
    caseis = sys.argv[1]
    print('Case Analysed is:', caseis, 'with strength Threshold:', float(sys.argv[2]))

structureListRestricted = [          4,            8,               1,           7,           0 ]
#limits                   [         27,           30,              24,       36-47,          22 ]
                         #[      esoph,      trachea,         cordprv,         ptv,        cord ]
threshold  =              [         10,           10,               5,          39,           5 ]
undercoeff =              [        0.0,          0.0,             0.0, 2891 * 1E2 ,         0.0 ]
overcoeff  =              [ 624 * 1E-4,  1209 * 1E-4,     6735 * 2E-3, 2891 * 6E-1, 3015 * 2E-3 ]

if "lung360" == caseis:
    structureListRestricted = [          0,            1,               2,           3,           4,            5 ]
    #limits                   [      60-66,      Mean<20,                ,       max45,mean20-max63, mean34-max63 ]
                           #[PTV Composite,    LUNGS-GTV,   CTEX_EXTERNAL,        CORD,       HEART,    ESOPHAGUS ]
    threshold  =              [         63,           10,               5,          20,          10,           10 ]
    undercoeff =              [        100,          0.0,             0.0,         0.0,         0.0,          0.0]
    overcoeff  =              [         50,         5E-1,             0.0,      2.2E-2,        1E-8,         5E-7]
    if 1 == len(sys.argv):
        strengthThreshold = 0.5 #this is the default threshold value for this case

if "brain360" == caseis:
    structureListRestricted = [          1,            2,               3,           5,           6,            9,  11,     12,     15,           16]
    #limits                   [        PTV,          PTV,       Brainstem,       ONRVL     ONRVR,      chiasm,    eyeL,   eyeR,  BRAIN,      COCHLEA]
    #                                                                  60           54           54         54      40      40      10            40]
    threshold  =              [         58,           58,            10.0,        10.0,        10.0,         10.0, 30.0,  30.0,   10.0,         10.0]
    undercoeff =              [        100,         5E+3,             0.0,         0.0,         0.0,          0.0,  0.0,   0.0,    0.0,          0.0]
    overcoeff  =              [         50,          150,            5E-5,          5,          5.5,           5, 5E-7,  5E-1,    0.5,         2E-1]
    if 1 == len(sys.argv):
        strengthThreshold = 0.45 #this is the default threshold value for this case
    # above this threshold and the beamlet will be considered for insertion into the problem

numcores = 8
testcase = [i for i in range(0, 180, 45)]
fullcase = [i for i in range(180)]
debugmode = False
easyread = False
eliminateWrapper = False #True if you want to do the wrap of all hard limits. False, if you want individually.
numsubdivisions = 1

gc.enable()
## Find out the variables according to the hostname
datalocation = '~'
if 'radiation-math' == socket.gethostname():  # LAB
    datalocation = "/mnt/fastdata/Data/brain360/by-Beam/"
    dropbox = "/mnt/datadrive/Dropbox"
    cutter = 44
elif 'sharkpool' == socket.gethostname():  # MY HOUSE
    datalocation = "/home/wilmer/Dropbox/Data/brain360/by-Beam/"
    dropbox = "/home/wilmer/Dropbox"
elif ('DESKTOP-EA1PG8V' == socket.gethostname()):  # FLUX
    datalocation = "C:/Users/wihen/Data/" + caseis + "/by-Beam/"
    dropbox = "D:/Dropbox"
    cutter = 45
    numcores = 11
    if "lung360" == caseis:
        cutter = 44
else:
    datalocation = "/home/wilmer/Dropbox/Data/brain360/by-Beam/"  # MY LAPTOP
    dropbox = "/home/wilmer/Dropbox"

## This class contains a structure (region)
# It could be a PTV or an organ. It also keeps track of the associated goals such
# as the function coefficients as they pertain to the objective function
class structure(object):
    ## Static variable that keeps a tally of the total number of structures
    numStructures = 0
    numTargets = 0
    numOARs = 0
    listTargets = []
    def __init__(self, sthing, s):
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
            structure.listTargets.append(s)
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

## Contains the beam data
# It keeps the location, the beamlet members, different measures of the aperture
class beam(object):
    numBeams = 0
    M = None
    N = None
    JawX1 = None
    JawX2 = None
    JawY1 = None
    JawY2 = None
    def __init__(self, sthing):
        # Initialize left and right leaf limits for all
        self.leftEdge = self.N // 2
        self.rightEdge = self.N // 2 + 1/numsubdivisions
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
        self.xilist = [self.leftEdge] * self.M
        self.psilist = [self.rightEdge] * self.M
        # The two hard limits below exist only to test whether the hard limits changed or not
        self.initialxilist = [self.leftEdge] * self.M
        self.initialpsilist = [self.rightEdge] * self.M
        self.KellyMeasure = 0
        self.Perimeter = 0
        self.Area = 0
        self.leftEdgeFract = None
        self.rightEdgeFract = None
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
        self.XStart = sthing.XStart + 7  # Make everything start at zero
        self.YStart = sthing.YStart
        beamlet.XSize = sthing.XSize
        beamlet.YSize = sthing.YSize
        self.belongsToBeam = None

## apertureList is a class definition of locs and angles that is always sorted.
# Its attributes are loc which is the numeric location; It has range 0 to 36 for
# the brain case; Angle is the numeric angle in degrees; It ranges from 0 to 360 degrees
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
        toremove = [i for i, x in enumerate(self.loc) if x == index]
        self.loc.pop(toremove[0])  # Notice that it removes the first entry
        self.angle.pop(toremove[0])

    ## Looks for the angle and removes the index and the angle corresponding to it from the list
    def removeAngle(self, tangl):
        toremove = [i for i, x in enumerate(self.angle) if x == tangl]
        self.loc.pop(toremove[0])
        self.angle.pop(toremove[0])

    ## Overloads parenthesis operator in order to fetch the ANGLE given an index.
    # Returns the angle at the ith location given by the index.
    # First Find the location of that index in the series of loc
    # Notice that this function overloads the parenthesis operator for elements of this class.
    def __call__(self, index):
        toreturn = [i for i, x in enumerate(self.loc) if x == index]
        return (self.angle[toreturn[0]])

    ## Returns the length of this instantiation without the need to pass parameters.
    def len(self):
        return (len(self.loc))

    ## Returns True if the list is empty; otherwise returns False.
    def isEmpty(self):
        if 0 == len(self.loc):
            return (True)
        else:
            return (False)


def get_files_by_file_size(dirname, reverse=False):
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
    filepaths.sort(key=lambda filename: filename[1], reverse=reverse)
    # Re-populate list with just filenames
    for i in range(len(filepaths)):
        filepaths[i] = filepaths[i][0]
    return (filepaths)

datafiles = get_files_by_file_size(datalocation)

# The first file will contain all the structure data, the rest will contain pointodoses.
dpdata = dose_to_points_data_pb2.DoseToPointsData()

# Start with reading structures, numvoxels and all that.
stfile = datafiles.pop(0)
print('structure file: ', stfile)
f = open(stfile, "rb")
dpdata.ParseFromString(f.read())
f.close()
datafiles.sort()
# ----------------------------------------------------------------------------------------
# Get the data about the structures
numstructs = len(dpdata.Structures)
structureList = []
structureDict = {}
print("Reading in structures")
for s in range(numstructs):
    print('Reading:', dpdata.Structures[s].Id)
    structureList.append(structure(dpdata.Structures[s], s))
    structureDict[structureList[s].Id] = s
    print('This structure goes between voxels ', structureList[s].StartPointIndex, ' and ',
          structureList[s].EndPointIndex)
print('Number of structures:', structure.numStructures, '\nNumber of Targets:', structure.numTargets,
      '\nNumber of OARs', structure.numOARs)
# Manual modifications of targets
if not debugmode:
    print('modifying penalization function according to the 5 structure case')
    strcounter = 0
    for s in structureListRestricted:
        structureList[s].underdoseCoeff = undercoeff[strcounter]
        structureList[s].overdoseCoeff = overcoeff[strcounter]
        structureList[s].threshold = threshold[strcounter]
        strcounter += 1
# ----------------------------------------------------------------------------------------
## Get the data about beamlets
numbeamlets = len(dpdata.Beamlets)
beamletList = [None] * numbeamlets
print('Reading in beamlet data:')
xstartlist = []
ystartlist = []
for blt in range(numbeamlets):
    a = beamlet(dpdata.Beamlets[blt])
    beamletList[dpdata.Beamlets[blt].Index] = a
    xstartlist.append(beamletList[blt].XStart)
    ystartlist.append(beamletList[blt].YStart)

beam.M = len(np.unique(ystartlist))
beam.N = len(np.unique(xstartlist))
beam.rightEdge = beam.N
print('total number of beamlets read:', beamlet.numBeamlets)
# ----------------------------------------------------------------------------------------
# Get the data about beams
numbeams = len(dpdata.Beams)
beamList = [None] * numbeams
print('Reading in Beam Data:')
for b in range(numbeams):
    mybeam = beam(dpdata.Beams[b])
    beamList[int(int(mybeam.Id))] = mybeam
    for blt in range(mybeam.StartBeamletIndex, (mybeam.EndBeamletIndex+1)):
        beamletList[blt].belongsToBeam = int(mybeam.Id)
print('There are a total of beams:', beam.numBeams)
print('beamlet data was updated so they point to their owner')
# for b in range(numbeams):
#    print(beamList[b].angle)
# ----------------------------------------------------------------------------------------
## Get data about voxels.
numvoxels = len(dpdata.Points)
voxelList = []
print('Reading in Voxel data:')
for v in range(numvoxels):
    voxelList.append(voxel(dpdata.Points[v]))
print('total number of voxels read:', voxel.numVoxels)
## Free the memory
dpdata = None
gc.collect()
# ----------------------------------------------------------------------------------------
## Get the point to dose data in a sparse matrix
start = time.time()

# ----------------------------------------------------------------------------------------
def getDmatrixPiecesCutLimits():
    #threshold = 0.4 # above this threshold and the beamlet will be considered for insertion into the problem
    if easyread:
        print('doing an easyread')
        if debugmode:
            PIK = 'IMRTEasyReadCases/testdump-' + caseis + 'threshold' + str(strengthThreshold) + '.dat'
        else:
            PIK = 'IMRTEasyReadCases/fullcasedump-' + caseis + 'threshold' + str(strengthThreshold) + '.dat'
        with open(PIK, "rb") as f:
            datasave = pickle.load(f)
        f.close()

        blist = datasave[0]
        vlist = datasave[1]
        dlist = datasave[2]
    else:
        ## Initialize vectors for voxel component, beamlet component and dose
        thiscase = fullcase
        if debugmode:
            thiscase = testcase

        # Get the ranges of the voxels that I am going to use and eliminate the rest
        myranges = []
        for i in structureListRestricted:
            myranges.append(range(structureList[i].StartPointIndex, structureList[i].EndPointIndex))
        ## Read the beams now.
        counter = 0 # will count the control points
        #Beamlets per beam
        bpb = beam.N * beam.M
        uniquev = set()
        dlist = []
        vlist = []
        blist = []
        # xilowest and psihighest contains minimum and maximum values for each leaf along the path
        xilowest = [beamList[0].leftEdge] * beam.M
        psihighest = [beamList[0].rightEdge] * beam.M
        for fl in [datalocation + 'twolists'+ str(2*x) + '.pickle' for x in thiscase]:
            newvcps = []
            newbcps = []
            newdcps = []
            counter += 1
            print('reading datafile Boundary Creation:', counter, ' out of ', len(thiscase), fl)
            input = open(fl, 'rb')
            indices, doses = pickle.load(input)
            input.close()
            for k in indices.keys(): # k here represents the voxel that we are analyzing
                for m in myranges:
                    if k in m:
                        newvcps += [k] * len(indices[k]) # This is the voxel we're dealing with
                        newbcps += indices[k]
                        newdcps += doses[k]
            # Create the matrix that goes in the list
            print(cutter, fl[cutter:])
            thisbeam = int(int(fl[cutter:].split('.')[0])/2)  #Find the beamlet in its coordinate space (not in Angle)
            initialBeamletThisBeam = beamList[thisbeam].StartBeamletIndex
            # Transform the space of beamlets to the new coordinate system starting from zero
            newbcps = [i - initialBeamletThisBeam for i in newbcps]
            # Wilmer Changed this part here
            #------------------------------
            bcps = []
            beamshape = np.reshape(np.zeros(bpb), (beam.M, beam.N))
            for i, v in enumerate(newvcps):
                if np.log2(data.maskValue[v]) in structure.listTargets:
                    if newdcps[i] > strengthThreshold:
                        bcps.append(newbcps[i])
            for i in np.unique(bcps):
                j = i % beam.N
                k = i // beam.N
                beamshape[k, j] = 1

            xi = np.empty(beam.M)
            psi = np.empty(beam.M)
            for i in range(beam.M):
                myones = [i for i, x in enumerate(beamshape[i,]) if x == 1]
                if 0 == len(myones):
                    xi[i], psi[i] = beamList[thisbeam].leftEdge, beamList[thisbeam].rightEdge
                else:
                    xi[i], psi[i] = myones[0] - 1, myones[-1] + 1 # this is where I assign the hard boundaries.
                    # one to the left and one to the right of the beamlets that actually produce the dose.
                if xi[i] < xilowest[i]:
                    xilowest[i] = xi[i]
                if psi[i] > psihighest[i]:
                    psihighest[i] = psi[i]
            beamList[thisbeam].xilist = xi
            beamList[thisbeam].psilist = psi
            beamList[thisbeam].initialxilist = xi
            beamList[thisbeam].initialpsilist = psi
            del newdcps
            del newbcps
            del newvcps
            del bcps
        # Identify positions to eliminate, this is the envolvent case. The individual case is below
        if eliminateWrapper:
            position = int(0)
            eliminateThesePositions = []
            for i in range(beam.M):
                for j in range(beam.N):
                    if j <= xilowest[i] or j >= psihighest[i]:
                        eliminateThesePositions.append(position)
                    position += 1
                ## Do the real creation of matrices now
            keepThesePositions = [i for i in range(bpb) if i not in eliminateThesePositions]
        counter = 0
        for fl in [datalocation + 'twolists' + str(2 * x) + '.pickle' for x in thiscase]:

            newvcps = []
            newbcps = []
            newdcps = []
            counter += 1
            print('reading datafile:', counter, ' out of ', len(thiscase), fl)
            input = open(fl, 'rb')
            indices, doses = pickle.load(input)
            input.close()
            for k in indices.keys():  # k here represents the voxel that we are analyzing
                for m in myranges:
                    if k in m:
                        newvcps += [k] * len(indices[k])  # This is the voxel we're dealing with
                        newbcps += indices[k]
                        newdcps += doses[k]
                        uniquev.add(k)
            # Create the matrix that goes in the list
            print(cutter, fl[cutter:])
            thisbeam = int(int(fl[cutter:].split('.')[0]) / 2)  # Find the beamlet in its coordinate space (not in Angle)

            if not eliminateWrapper:
                position = int(0)
                eliminateThesePositions = []
                for i in range(beam.M):
                    for j in range(beam.N):
                        if j <= beamList[thisbeam].xilist[i] or j >= beamList[thisbeam].psilist[i]:
                            eliminateThesePositions.append(position)
                        position += 1
                    ## Do the real creation of matrices now
                keepThesePositions = [i for i in range(bpb) if i not in eliminateThesePositions]

            initialBeamletThisBeam = beamList[thisbeam].StartBeamletIndex
            # Transform the space of beamlets to the new coordinate system starting from zero
            newbcps = [i - initialBeamletThisBeam for i in newbcps]
            # Remove all positions that for sure will not be used in the pricing problem to save memory
            mylocs = np.isin(newbcps, keepThesePositions)
            numgoodguys = sum(mylocs * 1)
            shortnewbcps = [None] * numgoodguys
            shortnewdcps = [None] * numgoodguys
            shortnewvcps = [None] * numgoodguys
            newgoodguy = 0
            for i, l in enumerate(mylocs):
                if l:
                    shortnewbcps[newgoodguy] = newbcps[i]
                    shortnewdcps[newgoodguy] = newdcps[i]
                    shortnewvcps[newgoodguy] = newvcps[i]
                    newgoodguy += 1
            # Return bcps to the old coordinate system
            newbcpsreverted = [i + initialBeamletThisBeam for i in shortnewbcps]

            # print('Adding index beam:', thisbeam, 'Corresponding to angle:', 2 * thisbeam)
            #beamDs[thisbeam] = sparse.csr_matrix((shortnewdcps, (shortnewvcps, shortnewbcps)), shape=(voxel.numVoxels, bpb),
            #                                     dtype=float)
            del newdcps
            del newbcps
            del newvcps
            gc.collect()
            vlist += shortnewvcps
            dlist += shortnewdcps
            blist += newbcpsreverted
        datasave = [blist, vlist, dlist]
        if not easyread:
            if debugmode:
                PIK = 'IMRTEasyReadCases/testdump-' + caseis + 'threshold' + str(strengthThreshold) + '.dat'
            else:
                PIK = 'IMRTEasyReadCases/fullcasedump-' + caseis + 'threshold' + str(strengthThreshold) + '.dat'
            with open(PIK, "wb") as f:
                pickle.dump(datasave, f, pickle.HIGHEST_PROTOCOL)
            f.close()
    return(vlist, blist, dlist)

print('total time reading dose to points:', time.time() - start)

# ------------------------------------------------------------------------------------------------------------------

class problemData():
    def __init__(self):
        self.kappa = []
        self.notinC = apertureList()
        self.caligraphicC = apertureList()
        self.currentDose = np.zeros(voxel.numVoxels, dtype=float)
        self.quadHelperThresh = np.empty(voxel.numVoxels, dtype=float)
        self.quadHelperOver = np.empty(voxel.numVoxels, dtype=float)
        self.quadHelperUnder = np.empty(voxel.numVoxels, dtype=float)
        self.maskValue = np.empty(voxel.numVoxels, dtype=float)
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

    def setQuadHelpers(self, sList, vList):
        for i in range(voxel.numVoxels):
            sid = structureDict[vList[i].StructureId]  # Find structure of this particular voxel
            self.quadHelperThresh[i] = sList[sid].threshold
            self.quadHelperOver[i] = sList[sid].overdoseCoeff
            self.quadHelperUnder[i] = sList[sid].underdoseCoeff
            self.maskValue[i] = 2 ** sid

    # data class function
    def calcDose(self, newIntensities):
        self.currentIntensities = newIntensities
        self.currentDose = DmatBig.transpose() * newIntensities


def calcObjGrad(x):
    data.calcDose(x)
    oDoseObj = data.currentDose - data.quadHelperThresh
    oDoseObjCl = (oDoseObj > 0) * oDoseObj
    oDoseObjGl = oDoseObjCl * oDoseObjCl * data.quadHelperOver
    uDoseObj = data.quadHelperThresh - data.currentDose
    uDoseObjCl = (uDoseObj > 0) * uDoseObj
    uDoseObjGl = uDoseObjCl * uDoseObjCl * data.quadHelperUnder
    objectiveValue = sum(oDoseObjGl + uDoseObjGl)
    oDoseObjGl = oDoseObjCl * data.quadHelperOver
    uDoseObjGl = uDoseObjCl * data.quadHelperUnder
    mygradient = DmatBig * 2 * (oDoseObjGl - uDoseObjGl)
    return (objectiveValue, mygradient)


# The next function prints DVH values
def printresults(myfolder):
    data.maskValue = np.array([int(i) for i in data.maskValue])
    print('Starting to Print Result DVHs brain case')
    zvalues = data.currentDose
    maxDose = max([float(i) for i in zvalues])
    dose_resln = 0.1
    dose_ub = maxDose + 10
    bin_center = np.arange(0, dose_ub, dose_resln)
    # Generate holder matrix
    dvh_matrix = np.zeros((structure.numStructures, len(bin_center)))
    # iterate through each structure
    for s in data.structureIndexUsed:
        doseHolder = sorted(zvalues[[i for i, v in enumerate(data.maskValue & 2 ** s) if v > 0]])
        if 0 == len(doseHolder):
            continue
        histHolder, garbage = np.histogram(doseHolder, bin_center)
        histHolder = np.append(histHolder, 0)
        histHolder = np.cumsum(histHolder)
        dvhHolder = 1 - (np.matrix(histHolder) / max(histHolder))
        dvh_matrix[s,] = dvhHolder
    print('matrix shape:', dvh_matrix.shape)
    dvh_matrix = dvh_matrix[data.structureIndexUsed,]
    print(dvh_matrix.shape)

    myfig = pylab.plot(bin_center, dvh_matrix.T, linewidth=2.0)
    plt.grid(True)
    plt.xlabel('Dose Gray')
    plt.ylabel('Fractional Volume')
    plt.title('IMRT Solution')
    plt.legend(structureNames, prop={'size': 9})
    plt.savefig(myfolder + 'DVH-for-debugging-IMRT-' + caseis + '-Threshold' + str(strengthThreshold) + '.png')
    plt.close()

data = problemData()
vlist, blist, dlist = getDmatrixPiecesCutLimits()

print('max of dlist:', max(dlist))
data.voxelsUsed = np.unique(vlist)
strsUsd = set([])
strsIdxUsd = set([])
for v in data.voxelsUsed:
    print(v)
    strsUsd.add(voxelList[v].StructureId)
    strsIdxUsd.add(structureDict[voxelList[v].StructureId])
data.structuresUsed = list(strsUsd)
data.structureIndexUsed = list(strsIdxUsd)
print('voxels used:', data.voxelsUsed)
print('structures used in no particular order:', data.structureIndexUsed)
structureNames = []
for s in data.structureIndexUsed:
    structureNames.append(structureList[s].Id)  # Names have to be organized in this order or it doesn't work
print(structureNames)
DmatBig = sparse.csr_matrix((dlist, (blist, vlist)), shape=(beamlet.numBeamlets, voxel.numVoxels), dtype=float)
del vlist
del blist
del dlist
CValue = 1.0
# find initial location
data.currentIntensities = np.zeros(beamlet.numBeamlets)
data.calcDose(data.currentIntensities)
before = time.time()
data.res = minimize(calcObjGrad, data.currentIntensities, method='L-BFGS-B', jac=True,
                    bounds=[(0, None) for i in range(0, len(data.currentIntensities))],
                    options={'ftol': 1e-4, 'disp': 5})
print('result:', data.res.x)

PIK = dropbox + '/Research/VMAT/casereader/datares.pickle'
print(PIK)
with open(PIK, "wb") as f:
    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
f.close()

PIK = dropbox + '/Research/VMAT/casereader/beamletlist.pickle'
print(PIK)
with open(PIK, "wb") as f:
    pickle.dump(beamletList, f, pickle.HIGHEST_PROTOCOL)
f.close()

printresults(dropbox + '/Research/VMAT/casereader/outputGraphics/')
after = time.time()
print('The whole program took: ' + str(time.time() - start) + ' seconds to finish')
print('intensities', data.currentIntensities)
print('Lets try to print all of the other structures:')
# Free ressources
del DmatBig
del data.DlistT
# Read the data necessary
with open("dvhdump.dat", "rb") as f:
    datasave = pickle.load(f)
f.close()
DmatBig = sparse.csr_matrix((datasave[2], (datasave[0], datasave[1])), shape=(beamlet.numBeamlets, voxel.numVoxels),
                            dtype=float)
#data.currentDose += DmatBig * data.currentIntensities
printresults(dropbox + '/Research/VMAT/casereader/outputGraphics/')
print('done!')
