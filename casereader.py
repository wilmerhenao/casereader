
import numpy as np
import dose_to_points_data_pb2
import sys
import os
import time
import gc
from scipy import sparse
from scipy.optimize import minimize
from multiprocessing import Pool
from functools import partial
import socket
import math
import pylab
import matplotlib.pyplot as plt
import pickle
from matplotlib.backends.backend_pdf import PdfPages
import traceback

# List of organs that will be used# List of organs that will be used
# The last one is the case to be analysed. They all overwrite
strengthThreshold = 0.97
Nbeamlets = 1
#caseis = "spine360"
#caseis = "lung360"
caseis = "brain360"
#caseis = "braiF360"

CValue = 0.0
if len(sys.argv) > 1:
    CValue = float(sys.argv[1])
    print('new CValue is:', CValue)
    if len(sys.argv) > 2:
        caseis = str(sys.argv[2])
        print('new case is:', caseis)

# structureListRestricted = [          4,            8,               1,           7,           0 ]
# #limits                   [         27,           30,              24,       36-47,          22 ]
#                          #[      esoph,      trachea,         cordprv,         ptv,        cord ]
# threshold  =              [         10,           10,               5,          39,           5 ]
# undercoeff =              [        0.0,          0.0,             0.0, 2891 * 9E-2,         0.0 ]
# overcoeff  =              [ 624 * 1E-4,  1209 * 1E-4,     6735 * 2E-3, 2891 * 6E-2, 3015 * 2E-3 ]

structureListRestricted = [          4,            8,               1,           7,           0 , 6          ]
#limits                   [         27,           30,              24,       36-47,          22 , other target???]
                         #[      esoph,      trachea,         cordprv,         ptv,        cord , ptv]
threshold  =              [         10,           10,               5,          39,           5 , 39         ]
undercoeff =              [        0.0,          0.0,             0.0, 2891 * 9E2 ,         0.0 , 2891 * 1E2 ]
overcoeff  =              [ 624 * 1E-4,  1209 * 1E-4,     6735 * 2E-3, 2891 * 6E-1, 3015 * 2E-3 , 2891 * 6E-1]

if "lung360" == caseis:
    structureListRestricted = [          0,            1,               2,           3,           4,            5 ]
    #limits                   [      60-66,      Mean<20,                ,       max45,mean20-max63, mean34-max63 ]
                           #[PTV Composite,    LUNGS-GTV,   CTEX_EXTERNAL,        CORD,       HEART,    ESOPHAGUS ]
    threshold  =              [         63,           10,               5,          20,          10,           10 ]
    undercoeff =              [      300000,          0.0,             0.0,         0.0,         0.0,          0.0]
    overcoeff  =              [       5000,           50,           100.0,         100,         300,          300]
    strengthThreshold = 0.05
    Nbeamlets = 1

ptvpriority = False

if not ptvpriority:
    # Make sure that the optic nerve is preserved
    if "brain360" == caseis:
        structureListRestricted = [          1,            2,               3,           5,           6,            9,  11,     12,     15,           16]
        #  limits                 [        PTV,          PTV,       Brainstem,       ONRVL     ONRVR,      chiasm,    eyeL,   eyeR,  BRAIN,      COCHLEA]
        #                                                                  60           54           54         54      40      40      10            40]
        threshold  =              [         58,           58,            10.0,        10.0,        10.0,         10.0, 30.0,  30.0,   10.0,         10.0]
        undercoeff =              [        100,         5E+4,             0.0,         0.0,         0.0,          0.0,  0.0,   0.0,    0.0,          0.0]
        overcoeff  =              [         50,          150,            5E-5,          5,          5.5,           5, 5E-7,  5E-1,    0.5,         2E-1]
        strengthThreshold = 0.662
        Nbeamlets = 2

    if "braiF360" == caseis:
        structureListRestricted = [1, 2, 3, 5, 6, 9, 11, 12, 15, 16]
        # limits                   [        PTV,          PTV,       Brainstem,       ONRVL     ONRVR,      chiasm,    eyeL,   eyeR,  BRAIN,      COCHLEA]
        #                                                                  60           54           54         54      40      40      10            40]
        threshold = [33, 33, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
        undercoeff = [100, 5E+3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        overcoeff = [50, 150, 5E-5, 50, 55, 50, 5E-7, 5E-1, 5.0, 2E-1]
        strengthThreshold = 0.662
        Nbeamlets = 2

else:
    # Make sure that the PTV dies
    if "brain360" == caseis:
        structureListRestricted = [          1,            2,               3,           5,           6,            9,  11,     12,     15,           16]
        #limits                   [        PTV,          PTV,       Brainstem,       ONRVL     ONRVR,      chiasm,    eyeL,   eyeR,  BRAIN,      COCHLEA]
        #                                                                  60           54           54         54      40      40      10            40]
        threshold  =              [         58,           58,            10.0,        10.0,        10.0,         10.0, 30.0,  30.0,   10.0,         10.0]
        undercoeff =              [        100,         5E+4,             0.0,         0.0,         0.0,          0.0,  0.0,   0.0,    0.0,          0.0]
        overcoeff  =              [         50,          150,            5E-5,          50,          55,           50, 5E-7,  5E-1,    5.0,         2E-1]
        strengthThreshold = 0.662
        Nbeamlets = 2

    if "braiF360" == caseis:
        structureListRestricted = [1, 2, 3, 5, 6, 9, 11, 12, 15, 16]
        # limits                   [        PTV,          PTV,       Brainstem,       ONRVL     ONRVR,      chiasm,    eyeL,   eyeR,  BRAIN,      COCHLEA]
        #                                                                  60           54           54         54      40      40      10            40]
        threshold = [33, 33, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
        undercoeff = [300, 5E+4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        overcoeff = [50, 150, 5E-5, 50, 55, 50, 5E-7, 5E-1, 5.0, 2E-1]
        strengthThreshold = 0.662
        Nbeamlets = 2

numcores   = 1
testcase   = [i for i in range(0, 180, 45)]
fullcase   = [i for i in range(180)]
## If you activate this option. I will only analyze numcores apertures at a time
debugmode = False
easyread = False
refinementloops = True #This loop supercedes the eliminationPhase
eliminationPhase = False # Whether you want to eliminate redundant apertures at the end
memorySaving = True
numsubdivisions = 1

gc.enable()
## Find out the variables according to the hostname
datalocation = '~'
print(socket.gethostname())
if 'radiation-math' == socket.gethostname(): # LAB
    datalocation = "/mnt/fastdata/Data/" + caseis + "/by-Beam/"
    dropbox = "/mnt/datadrive/Dropbox"
    cutter = 44
    if "lung360" == caseis:
        cutter = 43
elif 'IOE-Starchief' == socket.gethostname(): # MY HOUSE
    datalocation = "/home/wilmer/Dropbox/Data/spine360/by-Beam/"
    dropbox = "/home/wilmer/Dropbox"
    cutter = 51
elif 'DESKTOP-EA1PG8V' == socket.gethostname(): # MY HOUSE
    datalocation = "C:/Users/wihen/Data/"+ caseis + "/by-Beam/"
    dropbox = "D:/Dropbox"
    cutter = 45
    numcores = 11
    #memorySaving = False
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

exec(open('VMATClasses.py').read())

mytime = timedata()

datafiles = get_files_by_file_size(datalocation)
# The first file will contain all the structure data, the rest will contain pointodoses
dpdata = dose_to_points_data_pb2.DoseToPointsData()

# Start with reading structures, numvoxels and all that.
print('datafiles:', datafiles)
f = open(datafiles.pop(0), "rb")
dpdata.ParseFromString(f.read())
f.close()
datafiles.sort()
#----------------------------------------------------------------------------------------
# Get the data about the structures
numstructs = len(dpdata.Structures)
structureList = []
structureDict = {}    # keys are the names of the structure and the value is the corresponding index location (integer)
print("Reading in structures")
for s in range(numstructs):
    print('Reading:', dpdata.Structures[s].Id)
    structureList.append(structure(dpdata.Structures[s], s))
    structureDict[structureList[s].Id] = s
    print('This structure goes between voxels ', structureList[s].StartPointIndex, ' and ', structureList[s].EndPointIndex)
    
print('Number of structures:', structure.numStructures, '\nNumber of Targets:', structure.numTargets,
      '\nNumber of OARs', structure.numOARs)
# Manual modifications of targets
print('modifying penalization function for the structures that we want to study')
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
beamletList = [None] * numbeamlets
print('Reading in beamlet data:')
xstartlist = []
ystartlist = []
for blt in range(numbeamlets):
    a = beamlet(dpdata.Beamlets[blt])
    beamletList[dpdata.Beamlets[blt].Index] = a
    xstartlist.append(beamletList[blt].XStart)
    ystartlist.append(beamletList[blt].YStart)
    #print(beamletList[blt].XSize, beamletList[blt].YSize, beamletList[blt].XStart, beamletList[blt].YStart)

beam.M = len(np.unique(ystartlist))
beam.N = len(np.unique(xstartlist))
beam.rightEdge = beam.N
print('total number of beamlets read:', beamlet.numBeamlets)
#----------------------------------------------------------------------------------------
# Get the data about beams. They will be ordered. But the Id will be in half the range
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
#----------------------------------------------------------------------------------------
## Get data about voxels.
numvoxels = len(dpdata.Points)
voxelList = [None] * numvoxels
print('Reading in Voxel data:')
for v in range(numvoxels):
    voxelList[dpdata.Points[v].Index] = voxel(dpdata.Points[v])
print('total number of voxels read:', voxel.numVoxels)
## Free the memory
dpdata = None
gc.collect()
#----------------------------------------------------------------------------------------
## Get the point to dose data in a sparse matrix
def getDmatrixPieces():
    if easyread:
        # The analyse vectors are not being changed here because they will be read AFTER the optimization is done
        print('doing an easyread')
        if debugmode:
            PIK = datalocation + '../testdump.dat'
        else:
            PIK = datalocation + '../fullcasedump.dat'
        with open(PIK, "rb") as f:
            datasave = pickle.load(f)
        f.close()

        newbcps = datasave[0]
        newvcps = datasave[1]
        newdcps = datasave[2]
    else:
        ## Initialize vectors for voxel component, beamlet component and dose
        newvcps = []
        newbcps = []
        newdcps = []

        thiscase = fullcase
        if debugmode:
            thiscase = testcase

        # Get the ranges of the voxels that I am going to use and eliminate the rest
        myranges = []
        for i in structureListRestricted:
            myranges.append(range(structureList[i].StartPointIndex, structureList[i].EndPointIndex))
        ## Read the beams now.
        counter = 0
        for fl in [datafiles[x] for x in thiscase]:
            print(fl)
            counter += 1
            print('reading datafile:', counter,fl)
            input = open(fl, 'rb')
            indices, doses = pickle.load(input)
            input.close()
            for k in indices.keys():
                for m in myranges:
                    if k in m:
                        newvcps += [k] * len(indices[k]) # This is the voxel we're dealing with
                        newbcps += indices[k]
                        newdcps += doses[k]
            gc.collect()
            del indices
            del doses
            del input
        print('voxels seen:', np.unique(newvcps))
        datasave = [newbcps, newvcps, newdcps]
        if debugmode:
            PIK = datalocation + '../testdump.dat'
        else:
            PIK = datalocation + '../fullcasedump.dat'
        with open(PIK, "wb") as f:
            pickle.dump(datasave, f, pickle.HIGHEST_PROTOCOL)
        f.close()
    return(newvcps, newbcps, newdcps)

#----------------------------------------------------------------------------------------

def fillbeamshape(beamshape, Nbeamlets):
    newbeamshape = np.reshape(np.zeros(beam.M * beam.N), (beam.M, beam.N))
    for i in range(beam.M):
        for j in range(beam.N):
            newbeamshape[i,j] = 1 * (beamshape[max(0, i - Nbeamlets): min(beam.M, i + Nbeamlets + 1),
            max(0, j - Nbeamlets): min(beam.N, j + Nbeamlets + 1)].sum() > 0)
    return(newbeamshape)

## Get the point to dose data in a list of sparse matrices
def getDmatrixPiecesMemorySaving():
    ## Initialize vectors for voxel component, beamlet component and dose
    thiscase = fullcase
    if debugmode:
        thiscase = testcase

    # Get the ranges of the voxels that I am going to use and eliminate the rest
    myranges = []
    sickranges = [] # sickranges contains the ranges of the target structures only
    for i in structureListRestricted:
        myranges.append(range(structureList[i].StartPointIndex, structureList[i].EndPointIndex))
        if i in structure.listTargets:
            sickranges.append(range(structureList[i].StartPointIndex, structureList[i].EndPointIndex))
    ## Read the beams now.
    counter = 0 # will count the control points
    beamDs = [None] * beamlet.numBeamlets
    #Beamlets per beam
    bpb = beam.N * beam.M
    uniquev = set()
    # xilowest and psihighest contains minimum and maximum values for each leaf along the path
    xilowest = [beamList[0].leftEdge] * beam.M
    psihighest = [beamList[0].rightEdge] * beam.M
    for fl in [datalocation + 'twolists'+ str(2*x) + '.pickle' for x in thiscase]:
        newvcps = []
        newbcps = []
        newdcps = []
        maxsickdose = 0.0 #Maxsickdose will contain the maximum dose to any target from any beamlet
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
                    for sickrange in sickranges:  # Keep track of doses to targets only
                        if k in sickrange:
                            maxcandidate = np.max(doses[k])
                            maxsickdose = max(maxcandidate, maxsickdose)
        # Create the matrix that goes in the list
        print(cutter, fl[cutter:])
        thisbeam = int(int(fl[cutter:].split('.')[0])/2)  #Find the beamlet in its coordinate space (not in Angle)
        initialBeamletThisBeam = beamList[thisbeam].StartBeamletIndex
        # Transform the space of beamlets to the new coordinate system starting from zero
        newbcps = [i - initialBeamletThisBeam for i in newbcps]
        # Wilmer Changed this part here
        #-------------------------------------

        bcps = []
        vcps = []
        dcps = []
        beamshape = np.reshape(np.zeros(bpb), (beam.M, beam.N))
        #maxDoseThisAngle = np.max(newdcps) #Max of all doses (Not used, but here just in case)
        maxDoseThisAngle = maxsickdose #Max of all doses to targets
        for i, v in enumerate(newvcps):
            if np.log2(data.maskValue[v]) in structure.listTargets:
                if newdcps[i] > maxDoseThisAngle * strengthThreshold: # above this threshold and the beamlet will be considered for insertion into the problem
                    bcps.append(newbcps[i])
                    #vcps.append(newvcps[i]) # No need to use these vectors (waste of time calculating them)
                    #dcps.append(newdcps[i])
        for i in np.unique(bcps):
            j = i % beam.N
            k = i // beam.N
            beamshape[k, j] = 1

        beamshape = fillbeamshape(beamshape, Nbeamlets)

        xi = np.empty(beam.M)
        psi = np.empty(beam.M)

        for i in range(beam.M):
            myones = [i for i, x in enumerate(beamshape[i,]) if x == 1]
            if 0 == len(myones):
                xi[i], psi[i] = beamList[thisbeam].leftEdge, beamList[thisbeam].rightEdge
            else:
                xi[i], psi[i] = myones[0] - 1, myones[-1] + 1 # this is where I assign the hard boundaries.
                # one to the left and one to the right of the beamlets that actually produce the dose.

        #xi, psi = xipsiexpansion(xi, psi, Nbeamlets)
        #print(xi, psi)
        # The for loop below is here to assign the global hull of boundaries
        for i in range(beam.M):
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
    # Identify positions to eliminate
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

        # print('Adding index beam:', thisbeam, 'Corresponding to angle:', 2 * thisbeam)
        beamDs[thisbeam] = sparse.csr_matrix((shortnewdcps, (shortnewvcps, shortnewbcps)), shape=(voxel.numVoxels, bpb),
                                             dtype=float)
        del newdcps
        del newbcps
        del newvcps
        gc.collect()
    return(beamDs, uniquev)

#------------------------------------------------------------------------------------------------------------------
## This class keeps the core of the function parameters. Regularly calculates doses, and the values of obj. fn. and gradients
class problemData(object):
    def __init__(self, numberbeams):
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
        self.voxelsUsed = None # This is going to be a set
        self.structuresUsed = None
        self.structureIndexUsed = None
        self.YU = None
        self.RU = None
        self.speedlim = None
        self.rmpres = None
        self.listIndexofAperturesRemovedEachStep = []
        self.listIndexofAperturesAddedEachStep = []
        self.distancebetweenbeams = int(360 / numberbeams)  # Assumes beam regularly discretized on the circle.

    ## Organizes parameters of the objective function
    def setQuadHelpers(self, sList, vList):
        for i in range(voxel.numVoxels):
            sid = structureDict[vList[i].StructureId] # Find structure of this particular voxel
            self.quadHelperThresh[i] = sList[sid].threshold
            self.quadHelperOver[i] = sList[sid].overdoseCoeff / sList[sid].Size
            self.quadHelperUnder[i] = sList[sid].underdoseCoeff / sList[sid].Size
            self.maskValue[i] = 2**sid

    ## This function regularly enters the optimization engine to calculate doses
    def calcDose(self):
        self.currentDose = np.zeros(voxel.numVoxels, dtype = float)
        self.dZdK = np.matrix(np.zeros((voxel.numVoxels, beam.numBeams)))
        if self.caligraphicC.len() != 0:
            for i in self.caligraphicC.loc:
                self.currentDose += self.DlistT[i][:,self.openApertureMaps[i]] * sparse.diags(self.strengths[i]) * np.repeat(self.currentIntensities[i], len(self.openApertureMaps[i]), axis = 0)
                self.dZdK[:,i] = (self.DlistT[i] * sparse.diags(self.diagmakers[i], 0)).sum(axis=1)

    ## This function regularly enters the optimization engine to calculate objective function and gradients
    def calcGradientandObjValue(self):
        oDoseObj = self.currentDose - self.quadHelperThresh
        oDoseObjCl = (oDoseObj > 0) * oDoseObj
        oDoseObj = (oDoseObj > 0) * oDoseObj
        oDoseObj = oDoseObj * oDoseObj * self.quadHelperOver

        uDoseObj = self.quadHelperThresh - self.currentDose
        uDoseObjCl = (uDoseObj > 0) * uDoseObj
        uDoseObj = (uDoseObj > 0) * uDoseObj
        uDoseObj = uDoseObj * uDoseObj * self.quadHelperUnder
        oDoseObjGl = 2 * oDoseObjCl * self.quadHelperOver
        uDoseObjGl = 2 * uDoseObjCl * self.quadHelperUnder
        # Notice that I use two types of gradients. One for voxels and one for apertures
        self.voxelgradient = 2 * (oDoseObjGl - uDoseObjGl)
        self.aperturegradient = (np.asmatrix(self.voxelgradient) * self.dZdK).transpose()
        self.objectiveValue = sum(oDoseObj + uDoseObj)


## Find geographical location of the ith row in aperture index given by index. This is really only a problem for the
# HN case from the CORT database
# Input:    i:     Row
#           index: Index of this aperture
# Output:   validbeamlets ONLY contains those beamlet INDICES for which we have available data in this beam angle
#           validbeamletspecialrange is the same as validbeamlets but appending the endpoints
def fvalidbeamlets(index):
    validbeamlets = np.array(range(beamList[index].N )) # Wilmer changed this line
    validbeamletspecialrange = np.append(np.append(min(validbeamlets) - 1, validbeamlets), max(validbeamlets) + 1)
    # That last line appends the endpoints.
    return (validbeamlets, validbeamletspecialrange)

## C, C2, C3 are constants in the penalization function
# angdistancem = $\delta_{c^-c}$
# angdistancep = $\delta_{cc^+}$
# vmax = maximum leaf speed
# speedlim = s
# predec = predecesor index, either an index or an empty list
# succ = succesor index, either an index or an empty list
# lcm = vector of left limits in the previous aperture
# lcp = vector of left limits in the next aperture
# rcm = vector of right limits in the pPPrevious aperture
# rcp = vector of right limits in the previous aperture
# N = Number of beamlets per row
# M = Number of rows in an aperture
# thisApertureIndex = index location in the set of apertures that I have saved.
# K = Number of times that I will artificially split each interval
def PPsubroutine(C, C2, C3, angdistancem, angdistancep, vmax, speedlim, predec, succ, thisApertureIndex, bw, K):
    print('this aperture index:', thisApertureIndex)
    # Get the slice of the matrix that I need
    if memorySaving:
        D = data.DlistT[thisApertureIndex].transpose()
    else:
        D = DmatBig[beamList[thisApertureIndex].StartBeamletIndex:(beamList[thisApertureIndex].EndBeamletIndex + 1),]
    M = beamList[thisApertureIndex].M
    leftEdge = beamList[thisApertureIndex].leftEdge
    rightEdge = beamList[thisApertureIndex].rightEdge
    #beamList[thisApertureIndex].leftEdgeFract = leftEdge + (K - 1) / K
    #beamList[thisApertureIndex].rightEdgeFract = rightEdge - (K - 1) / K
    b = bw
    # vmaxm and vmaxp describe the speeds that are possible for the leaves from the predecessor and to the successor
    vmaxm = vmax
    vmaxp = vmax
    # Arranging the predecessors and the succesors
    #Predecessor left and right indices
    if type(predec) is list:
        hardlcm = [leftEdge] * M
        hardrcm = [rightEdge] * M
        lcm = [leftEdge] * M
        rcm = [rightEdge] * M
        # If there is no predecessor is as if the pred. speed was infinite
        vmaxm = np.inf
    else:
        hardlcm = beamList[predec].xilist
        hardrcm = beamList[predec].psilist
        lcm = beamList[predec].llist
        rcm = beamList[predec].rlist

    #Succesors left and right indices
    if type(succ) is list:
        hardlcp = [leftEdge] * M
        hardrcp = [rightEdge] * M
        lcp = [leftEdge] * M
        rcp = [rightEdge] * M
        # If there is no successor is as if the succ. speed was infinite.
        vmaxp = np.inf
    else:
        hardlcp = beamList[succ].xilist
        hardrcp = beamList[succ].psilist
        lcp = beamList[succ].llist
        rcp = beamList[succ].rlist

    # Handle the calculations for the first row
    beamGrad = D * data.voxelgradient

    nodesinpreviouslevel = 0
    posBeginningOfRow = 1
    thisnode = 0
    # Max beamlets per row
    bpr = K * (beamList[thisApertureIndex].N + 2) # the ones inside plus two edges
    networkNodesNumber = bpr * bpr + M * bpr * bpr + bpr * bpr # An overestimate of the network nodes in this network
    # Initialization of network vectors. This used to be a list before
    lnetwork = np.zeros(networkNodesNumber, dtype = np.float32) #left limit vector
    rnetwork = np.zeros(networkNodesNumber, dtype = np.float32) #right limit vector
    mnetwork = np.ones(networkNodesNumber, dtype = np.int) #Only to save some time in the first loop
    wnetwork = np.empty(networkNodesNumber, dtype = np.float) # Weight Vector initialized with +\infty
    wnetwork[:] = np.inf
    dadnetwork = np.zeros(networkNodesNumber, dtype = np.int) # Dad Vector. Where Dad is the combination of (l,r) in previous row
    # These subfunctions help round to the nearest multiple of 1/K
    def roundceil(x):
        return(math.ceil(x * K) / K)
    def roundfloor(x):
        return(math.floor(x * K) / K)

    # Predecessor left and right hard limits
    for i in range(M):
        beamList[thisApertureIndex].xilist[i] = roundfloor(min([beamList[thisApertureIndex].xilist[i], hardlcm[i] + vmaxm * (angdistancem/speedlim)/bw, hardlcp[i] + vmaxm * (angdistancem/speedlim)/bw]) )
        beamList[thisApertureIndex].psilist[i] = roundceil(max([beamList[thisApertureIndex].psilist[i], hardrcm[i] - vmaxm * (angdistancem/speedlim)/bw, hardrcp[i] - vmaxm * (angdistancem/speedlim)/bw]) )

    # Work on the first row perimeter and area values
    leftrange = np.arange(roundceil(max(lcm[0] - vmaxm * (angdistancem/speedlim)/bw, lcp[0] - vmaxp * (angdistancep/speedlim)/bw, beamList[thisApertureIndex].xilist[0] + (K-1)/K)), (1/K) + roundfloor(min(lcm[0] + vmaxm * (angdistancem/speedlim)/bw , lcp[0] + vmaxp * (angdistancep/speedlim)/bw, beamList[thisApertureIndex].psilist[0] - (K-1) / K - 1/K )), 1/K)
    # Check if unfeasible. If it is then assign one value
    if (0 == len(leftrange)):
        midpoint = (angdistancep * lcm[0] + angdistancem * lcp[0])/(angdistancep + angdistancem)
        leftrange = np.arange(midpoint, midpoint + 1)
    for l in leftrange:
        rightrange = np.arange(roundceil(max(l + (1 / K), rcm[0] - vmaxm * (angdistancem/speedlim)/bw , rcp[0] - vmaxp * (angdistancep/speedlim)/bw, beamList[thisApertureIndex].xilist[0] + (K-1)/K)), (1/K) + roundfloor(min(rcm[0] + vmaxm * (angdistancem/speedlim)/bw , rcp[0] + vmaxp * (angdistancep/speedlim)/bw, beamList[thisApertureIndex].psilist[0] - (K-1) / K )), 1/K)
        if (0 == len(rightrange)):
            midpoint = (angdistancep * rcm[0] + angdistancem * rcp[0])/(angdistancep + angdistancem)
            rightrange = np.arange(midpoint, midpoint + 1)
            ##print('constraint rightrange at level ' + str(0) + ' aperture ' + str(thisApertureIndex) + ' could not be met', 'ERROR Report: lcm[0], angdistancem, lcp[0], angdistancep', lcm[0], angdistancem, lcp[0], angdistancep, '\nFull left limits, rcp, rcm:', rcp, rcm, 'm: ', 0, 'predecesor: ', predec, 'succesor: ', succ)
        for r in rightrange:
            thisnode += 1
            nodesinpreviouslevel += 1
            # First I have to make sure to add the beamlets that I am interested in
            if(l + (1 / K) < r): # prints r numbers starting from l + 1. So range(3,4) = 3
                ## Take integral pieces of the dose component
                possiblebeamletsthisrow = range(int(np.ceil(l)),int(np.floor(r)))
                ## Calculate dose on the sides, the fractional component
                DoseSide = -((np.ceil(l) - (l)) * beamGrad[int(np.floor(l))] + (r - np.floor(r)) * beamGrad[int(np.ceil(r))])
                if (len(possiblebeamletsthisrow) > 0):
                    Dose = -beamGrad[ possiblebeamletsthisrow ].sum()
                    weight = C * ( C2 * (r - l) - C3 * b * (r - l)) - Dose + 10E-10 * (r-l) + DoseSide# The last term in order to prefer apertures opening in the center
                else:
                    weight = C * ( C2 * (r - l) - C3 * b * (r - l)) + 10E-10 * (r-l) + DoseSide
            else:
                weight = 0.0 # it is turned off
            # Create node (1,l,r) in array of existing nodes and update the counter
            # Replace the following expression
            lnetwork[thisnode] = l
            rnetwork[thisnode] = r
            wnetwork[thisnode] = weight
            # dadnetwork and mnetwork don't need to be changed here for obvious reasons
    posBeginningOfRow += nodesinpreviouslevel
    leftmostleaf = beam.N - 1 # Position in python position(-1) of the leftmost leaf
    # Then handle the calculations for the m rows. Nodes that are neither source nor sink.
    for m in range(1,M):
        oldflag = nodesinpreviouslevel
        nodesinpreviouslevel = 0
        # And now process normally checking against valid beamlets
        leftrange = np.arange(roundceil(max(lcm[m] - vmaxm * (angdistancem/speedlim)/bw, lcp[m] - vmaxp * (angdistancep/speedlim)/bw, beamList[thisApertureIndex].xilist[m] + (K-1) / K)), (1/K) + roundfloor(min(lcm[m] + vmaxm * (angdistancem/speedlim)/bw , lcp[m] + vmaxp * (angdistancep/speedlim)/bw, beamList[thisApertureIndex].psilist[m] - (K-1) / K - 1/K)), 1/K)
        # Check if unfeasible. If it is then assign one value but tell the result to the person running this
        #print('N', beam.N, 'xilist', beamList[thisApertureIndex].xilist[m], 'psilist', beamList[thisApertureIndex].psilist[m])
        #print('leftrange', leftrange)
        if(0 == len(leftrange)):
            midpoint = (angdistancep * lcm[m] + angdistancem * lcp[m])/(angdistancep + angdistancem)
            leftrange = np.arange(midpoint, midpoint + 1)
        for l in leftrange:
            rightrange = np.arange(roundceil(max(l + (1/K), rcm[m] - vmaxm * (angdistancem/speedlim)/bw, rcp[m] - vmaxp * (angdistancep/speedlim)/bw, beamList[thisApertureIndex].xilist[m] + (K-1) / K)), (1/K) + roundfloor(min(rcm[m] + vmaxm * (angdistancem/speedlim)/bw , rcp[m] + vmaxp * (angdistancep/speedlim)/bw, beamList[thisApertureIndex].psilist[m] - (K-1) / K)), 1 / K)
            #print('left and right range', l, rightrange)
            #print('components', roundceil(max(l + (1/K), rcm[m] - vmaxm * (angdistancem/speedlim)/bw, rcp[m] - vmaxp * (angdistancep/speedlim)/bw, beamList[thisApertureIndex].xilist[m] + (K-1) / K)), (1/K) + roundfloor(min(rcm[m] + vmaxm * (angdistancem/speedlim)/bw , rcp[m] + vmaxp * (angdistancep/speedlim)/bw, beamList[thisApertureIndex].psilist[m] - (K-1) / K)))
            #print('components left', l + (1/K), rcm[m] - vmaxm * (angdistancem/speedlim)/bw, rcp[m] - vmaxp * (angdistancep/speedlim)/bw, beamList[thisApertureIndex].xilist[m] + (K-1) / K)
            if (0 == len(rightrange)):
                midpoint = (angdistancep * rcm[m] + angdistancem * rcp[m])/(angdistancep + angdistancem)
                rightrange = np.arange(midpoint, midpoint + 1)
            for r in rightrange:
                nodesinpreviouslevel += 1
                thisnode += 1
                # Create node (m, l, r) and update the level counter
                lnetwork[thisnode] = l
                rnetwork[thisnode] = r
                mnetwork[thisnode] = m
                wnetwork[thisnode] = np.inf
                # Select only those beamlets that are possible in between the (l,r) limits.
                possiblebeamletsthisrow = range(int(np.ceil(l)) + leftmostleaf, int(np.floor(r) + leftmostleaf))#
                DoseSide = -((np.ceil(l) - (l)) * beamGrad[int(np.floor(l)) + leftmostleaf] + (r - np.floor(r)) * beamGrad[int(np.ceil(r)) + leftmostleaf])
                if(len(possiblebeamletsthisrow) > 0):
                    #print(possiblebeamletsthisrow)
                    Dose = -beamGrad[possiblebeamletsthisrow].sum()
                    C3simplifier = C3 * b * (r - l)
                else:
                    Dose = 0.0
                    C3simplifier = 0.0
                lambdaletter = np.absolute(lnetwork[(posBeginningOfRow - oldflag): posBeginningOfRow] - l) + np.absolute(rnetwork[(posBeginningOfRow - oldflag): posBeginningOfRow] - r) - 2 * np.maximum(0, lnetwork[(posBeginningOfRow - oldflag): posBeginningOfRow] - r) - 2 * np.maximum(0, l - np.absolute(rnetwork[(posBeginningOfRow - oldflag): posBeginningOfRow]))
                weight = C * (C2 * lambdaletter - C3simplifier) - Dose  + 10E-10 * (r-l) + DoseSide # The last term in order to prefer apertures opening in the center
                # Add the weights that were just calculated
                newweights = wnetwork[(posBeginningOfRow - oldflag): posBeginningOfRow] + weight
                # Find the minimum and its position in the vector
                minloc = np.argmin(newweights)
                wnetwork[thisnode] = newweights[minloc]
                dadnetwork[thisnode] = minloc + posBeginningOfRow - oldflag

        posBeginningOfRow = nodesinpreviouslevel + posBeginningOfRow # This is the total number of network nodes
        # Keep the location of the leftmost leaf
        leftmostleaf = beam.N + leftmostleaf
    # thisnode gets augmented only 1 because only the sink node will be added
    thisnode += 1

    for mynode in (range(posBeginningOfRow - nodesinpreviouslevel, posBeginningOfRow )): # +1 because otherwise it could be empty
        weight = C * ( C2 * (rnetwork[mynode] - lnetwork[mynode] ))
        if(wnetwork[mynode] + weight <= wnetwork[thisnode]):
            wnetwork[thisnode] = wnetwork[mynode] + weight
            dadnetwork[thisnode] = mynode
            p = wnetwork[thisnode]
    thenode = thisnode
    l = []
    r = []
    while(1):
        # Find the predecessor data
        l.append(lnetwork[thenode])
        r.append(rnetwork[thenode])
        thenode = dadnetwork[thenode]
        if(0 == thenode): # If at the origin then break
            break
    l.reverse()
    r.reverse()
    #Pop the last elements because this is the direction of nonexistent sink field
    l.pop(); r.pop()
    return(p, l, r)

def parallelizationPricingProblem(i, C, C2, C3, vmax, speedlim, bw, K, refinementFlag = False):
    thisApertureIndex = i
    # print("analysing available aperture" , thisApertureIndex)
    # Find the successor and predecessor of this particular element
    try:
        #This could be done with angles instead of indices (reconsider this at some point)
        succs = [i for i in data.caligraphicC.loc if i > thisApertureIndex]
    except:
        succs = []
    try:
        predecs = [i for i in data.caligraphicC.loc if i < thisApertureIndex]
    except:
        predecs = []

    # If there are no predecessors or succesors just return an empty list. If there ARE, then return the indices
    if 0 == len(succs):
        succ = []
        angdistancep = np.inf
    else:
        succ = min(succs)
        angdistancep = data.caligraphicC(succ) - data.notinC(thisApertureIndex)
    if 0 == len(predecs):
        predec = []
        angdistancem = np.inf
    else:
        predec = max(predecs)
        angdistancem = data.notinC(thisApertureIndex) - data.caligraphicC(predec)
    # Find Numeric value of previous and next angle. Measuring the time taken and keeping it in the object
    thisInitialTime = time.clock()
    p, l, r = PPsubroutine(C, C2, C3, angdistancem, angdistancep, vmax, speedlim, predec, succ, thisApertureIndex, bw, K)
    #print('about to enter the function from outside')
    mytime.beamNetworkTime(thisInitialTime, thisApertureIndex, refinementFlag)
    #print('left the function beamNetworkTime')

    return(p,l,r,thisApertureIndex)

## The main difference between this pricing problem and the complete one is that this one analyses one aperture control
## point only.
def refinementPricingProblem(refaper, C, C2, C3, vmax, speedlim, beamletwidth, K):
    pstar, l, r, bestApertureIndex = parallelizationPricingProblem(refaper, C, C2, C3, vmax, speedlim, beamletwidth, K, True)
    # Calculate Kelly's aperture measure
    Area = 0.0
    Perimeter = (r[0] - l[0])/beamlet.XSize + np.sign(r[0] - l[0]) # First part of the perimeter plus first edge
    #for n in range(len(l)):
    #    Area += 0.5 * (r[n] - l[n]) * 0.5
    for n in range(1, len(l)):
        Area += 1.0 * (r[n] - l[n]) / beamlet.XSize
        Perimeter += np.sign(r[n] - l[n]) # Vertical part of the perimeter
        Perimeter += (np.abs(l[n] - l[n-1]) + np.abs(r[n] - r[n-1]) - 2 * np.maximum(0, l[n-1] - r[n]) - 2 * np.maximum(0, l[n] - r[n - 1]))/beamlet.XSize
    Perimeter += (r[len(r)-1] - l[len(l)-1]) / beamlet.XSize + np.sign(r[len(r)-1] - l[len(l)-1])
    Kellymeasure = Perimeter / Area
    return(pstar, l, r, bestApertureIndex, Kellymeasure, Perimeter, Area)

def chooseSmallest(locallocation, listinorder, degreesapart):
    # Choose the first one no matter what
    chosenlocs = [listinorder[0]]
    lllist = [locallocation[0]]
    for i in range(1, len(listinorder)):
        candidate = listinorder[i]
        # Makes sure that the new entry is far enough from the already included
        if min(np.absolute([min(abs(data.notinC(candidate) - data.notinC(apsin)), abs(360 - abs(data.notinC(candidate) - data.notinC(apsin)))) for apsin in chosenlocs])) > degreesapart:
            if not data.caligraphicC.isEmpty():
                if min(np.absolute([min(abs(data.notinC(candidate) - data.caligraphicC(apsin)), abs(360 - abs(data.notinC(candidate) - data.caligraphicC(apsin)))) for apsin in data.caligraphicC.loc])) > degreesapart:
                    chosenlocs.append(candidate)
                    lllist.append(locallocation[i])
                if degreesapart > 180:
                    return(lllist)
            else:
                chosenlocs.append(candidate)
                lllist.append(locallocation[i])
    return(lllist)

def PricingProblem(C, C2, C3, vmax, speedlim, bw, K):
    print("Choosing one aperture amongst the ones that are available")
    # Allocate empty list with enough size for all l, r combinations
    global structureList
    try:
        partialparsubpp = partial(parallelizationPricingProblem, C=C, C2=C2, C3=C3, vmax=vmax, speedlim=speedlim, bw=bw, K=K, refinementFlag=False)
        if __name__ == '__main__':
            pool = Pool(processes = numcores)              # process per MP
            locstotest = data.notinC.loc
            if debugmode:
                locstotest = data.notinC.loc[0:numcores]
            respool = pool.map(partialparsubpp, locstotest)
        pool.close()
        pool.join()
    except:
        traceback.print_exc()
        raise

    # Get only the pvalues
    pvalues = np.array([result[0] for result in respool])
    # Save all positions for the regression
    allApertures = ([result[1] for result in respool], [result[2] for result in respool])
    # Order according to pvalues
    respoolinorder = np.argsort(pvalues)
    listinorder = [respool[i][3] for i in respoolinorder]
    # Choose only pvalues that are negative to be selected to enter
    negpvalues = max(1, sum([1 for i in pvalues if i < 0]))
    ## Choose entering candidates making sure that there are at least 10 degrees of separation
    indstars = chooseSmallest(respoolinorder[:negpvalues], listinorder[:negpvalues], 180) #This 10 is the degrees of separation
    # Initialize the lists that I'm going to return
    pstarlist = []
    llist = []
    rlist = []
    bestApertureIndexlist = []
    Kellymeasurelist = []
    PerimeterList = []
    AreaList = []
    goodaperturessent = 0
    for indstar in indstars:
        bestgroup = respool[indstar]
        pstar = bestgroup[0]
        if pstar > 0:
            # Make sure that I report at least the pstar of the first one in case no one works
            pstarlist.append(pstar)
            bestApertureIndexlist.append('None Added')
            break #Break the for because it cannot get any better now.
        l = bestgroup[1]
        r = bestgroup[2]
        bestApertureIndex = bestgroup[3]
        # Change the leaf positions for this particular beam
        print("One of the best apertures was: ", bestApertureIndex)
        # Calculate Kelly's aperture measure
        Area = 0.0
        Perimeter = (r[0] - l[0]) + np.sign(r[0] - l[0]) # First part of the perimeter plus first edge
        #for n in range(len(l)):
        #    Area += 0.5 * (r[n] - l[n]) * 0.5
        for n in range(1, len(l)):
            Area += 1.0 * (r[n] - l[n])
            Perimeter += np.sign(r[n] - l[n]) # Vertical part of the perimeter
            Perimeter += (np.abs(l[n] - l[n-1]) + np.abs(r[n] - r[n-1]) - 2 * np.maximum(0, l[n-1] - r[n]) - 2 * np.maximum(0, l[n] - r[n - 1]))
        Perimeter += (r[len(r)-1] - l[len(l)-1]) + np.sign(r[len(r)-1] - l[len(l)-1])
        print(Perimeter, Area)
        Kellymeasure = Perimeter / Area
        pstarlist.append(pstar)
        llist.append(l)
        rlist.append(r)
        bestApertureIndexlist.append(bestApertureIndex)
        Kellymeasurelist.append(Kellymeasure)
        PerimeterList.append(Perimeter)
        AreaList.append(Area)
        goodaperturessent += 1
    return(pstarlist, llist, rlist, bestApertureIndexlist, Kellymeasurelist, goodaperturessent, PerimeterList, AreaList, allApertures)

## This function returns the set of available AND open beamlets for the selected aperture (i).
# The idea is to have everything ready and pre-calculated for the evaluation of the objective function in
# calcDose
# input: i is the index number of the aperture that I'm working on
# output: openaperturenp. the set of available AND open beamlets for the selected aperture. Doesn't contain fractional values
#         diagmaker. A vector that has a 1 in each position where an openaperturebeamlet is available.
# openaperturenp is read as openapertureMaps. A member of the VMAT_CLASS.
def updateOpenAperture(i, K):
    leftlimits = 0
    openaperture = []
    ## While openaperturenp contains positions, openapertureStrength contains proportion of the beamlets that's open.
    openapertureStrength = []
    diagmaker = np.zeros(beam.N * beam.M, dtype = float)
    for m in range(beam.M):
        # Find geographical values of llist and rlist.
        # Find geographical location of the first row.
        validbeamlets, validbeamletspecialrange = fvalidbeamlets(i)
        # First index in this row
        beginningOfLeftAperture = beamList[i].llist[m] + 1/K

        ## Notice that indleft and indright below may be floats instead of just integers
        if (beginningOfLeftAperture >= min(validbeamlets) - 1/K):
            ## I subtract min validbeamlets bec. I want to find coordinates in available space
            ## indleft is where the edge of the left leaf ends. From there on to the right, there are photons.
            indleft = beginningOfLeftAperture + leftlimits - min(validbeamlets)
        else:
            # if the left limit is too far away to the left, just take what's available
            indleft = 0

        if (beamList[i].rlist[m] > max(validbeamlets) + 1/K):
            # If the right limit is too far to the left, just grab the whole thing.
            indright = len(validbeamlets) + leftlimits
        else:
            if(beamList[i].rlist[m] >= min(validbeamlets)):
                ## indright is where the edge of the right leaf ends.
                indright = beamList[i].rlist[m] + leftlimits - min(validbeamlets)
            else:
                # Right limit is to the left of validbeamlets (This situation is weird)
                indright = 0
        if (indleft < indright): ## Make sure the thing opened at all.
            first = True
            #Everything should be included
            for thisbeamlet in range(int(np.floor(indleft)), int(np.ceil(indright))):
                strength = 1.0
                if first:
                    first = False
                    # Fix the proportion of the left beamlet that is open
                    strength = np.ceil(indleft) - indleft
                openapertureStrength.append(strength)
                diagmaker[thisbeamlet] = strength
                openaperture.append(thisbeamlet)
            ## Fix the proportion of the right beamlet that is open.
            strength = indright - np.floor(indright)
            if strength > 0.001: # this is here to make sure that right - floor(right) is not zero when right is integer
                ## Important: There is no need to check if there exists a last element because after all, you already
                # checked whe you entered the if loop above this one
                openapertureStrength[-1] = strength
                diagmaker[int(np.ceil(indright)) - 1] = strength

            ## One last scenario. If only a little bit of the aperture is open (less than a beamlet and within one beamlet
            if 1 == int(np.ceil(indright)) - int(np.floor(indleft)):
                strength = indright - indleft
                openapertureStrength[-1] = strength
                diagmaker[int(np.floor(indright))] = strength
        # Keep the location of the leftmost leaf
        leftlimits += len(validbeamlets)
    openaperturenp = np.array(openaperture, dtype=int) #Contains indices of open beamlets in the aperture
    return(openaperturenp, diagmaker, openapertureStrength)

def calcObjGrad(x, user_data = None):
    data.currentIntensities = x
    data.calcDose()
    data.calcGradientandObjValue()
    return(data.objectiveValue, data.aperturegradient)

def solveRMC(YU, mymaxiter = 20):
    start = time.clock()
    numbe = data.caligraphicC.len()
    calcObjGrad(data.currentIntensities)
    # Create the boundaries making sure that the only free variables are the ones with perfectly defined apertures.
    boundschoice = []
    for thisindex in range(0, beam.numBeams):
        if thisindex in data.caligraphicC.loc: #Only activate what is an aperture
            boundschoice.append((0, YU))
        else:
            boundschoice.append((0, 0))
    res = minimize(calcObjGrad, data.currentIntensities, method='L-BFGS-B', jac = True, bounds = boundschoice, tol = 0.1)#, options={'ftol':10e-1, 'disp':5,'maxfun':mymaxiter})
    print('Restricted Master Problem solved in ' + str(time.clock() - start) + ' seconds')
    return(res)

def contributionofBeam(refaper, oldobj,  C, C2, C3, vmax, beamletwidth, K):
    # Remove aperture from the set temporarily
    # Check if it's in caligraphicC or not
    itwasinCaligraphicC = False
    if refaper in data.caligraphicC.loc:
        data.notinC.insertAngle(beamList[refaper].location, beamList[refaper].angle)
        data.caligraphicC.removeIndex(refaper)
        itwasinCaligraphicC = True
        # I NEED to recalculate this in order to update the vector variable beamGrad to be used in the PP Problem
        data.openApertureMaps[refaper], data.diagmakers[refaper], data.strengths[refaper] = updateOpenAperture(refaper, K)
    # Do as if the dose was zero coming from that aperture
    data.calcDose()
    data.calcGradientandObjValue()
    # Select a new aperture for that particular location
    pstar, lm, rm, bestApertureIndex, kmeasure, perimeter, area = refinementPricingProblem(refaper, C, C2, C3, vmax, data.speedlim, beamletwidth, K)
    # Put the aperture back in
    if itwasinCaligraphicC:
        data.caligraphicC.insertAngle(bestApertureIndex, data.notinC(bestApertureIndex))
        data.notinC.removeIndex(bestApertureIndex)
        data.calcDose()
        data.calcGradientandObjValue()
    # If the new aperture is not good, don't waste more time and return 0
    if pstar > 0:
        print('p star > 0')
        return(0)
    else:
        print('p star < 0')
        # Solve the instance of the RMP associated with caligraphicC. But make sure to put everything back to the values
        # where it was before
        print('1')
        lmsave = beamList[bestApertureIndex].llist
        rmsave = beamList[bestApertureIndex].rlist
        beamList[bestApertureIndex].llist = lm
        beamList[bestApertureIndex].rlist = rm
        print('2')
        # Precalculate the aperture map to save times.
        data.openApertureMaps[bestApertureIndex], data.diagmakers[bestApertureIndex], data.strengths[bestApertureIndex] = updateOpenAperture(bestApertureIndex, K)
        print('3')
        data.rmpres = solveRMC(data.YU, 5)
        print('4')
        beamList[bestApertureIndex].llist = lmsave
        beamList[bestApertureIndex].rlist = rmsave
        print('5')
        data.openApertureMaps[bestApertureIndex], data.diagmakers[bestApertureIndex], data.strengths[bestApertureIndex] = updateOpenAperture(bestApertureIndex, K)
        print('6')
    return(data.rmpres.fun - oldobj)

def column_generation(C, K, mytime):
    C2 = 1/3
    C3 = 1/4
    CSave = C
    #C = 0.0
    #eliminationThreshold = 0.1 Wilmer:This one worked really Well
    eliminationThreshold = 0.3
    ## Maximum leaf speed
    vmaxincmspersecond = 4.25 # 3.25 cms per second
    data.speedlim = 0.8 # Values are in the VMATc paper page 2955. 0.85 < s < 6. This is in degrees per second
    secondsbetweenbeams = data.distancebetweenbeams / data.speedlim # seconds per beam interval
    cushion = 0.1
    vmax = vmaxincmspersecond * secondsbetweenbeams + cushion
    ## Maximum Dose Rate
    data.RU = 20.0
    ## Maximum intensity
    data.YU = data.RU / data.speedlim
    beamletwidth = beamlet.XSize / 10.0
    #beam.leftEdgeFract = beam.leftEdge + (K - 1) / K
    #beam.rightEdgeFract = beam.rightEdge - (K - 1) / K
    bigListofApertures = []

    #Step 0 on Fei's paper. Set C = empty and zbar = 0. The gradient of numbeams dimensions generated here will not
    # be used, and therefore is nothing to worry about.
    # At the beginning no apertures are selected, and those who are not selected are all in notinC
    if debugmode:
        rangenumbeams = testcase
    else:
        rangenumbeams = range(beam.numBeams)
    for j in rangenumbeams:
        data.notinC.insertAngle(beamList[j].location, beamList[j].angle)

    plotcounter = 0
    optimalvalues = []
    while (data.notinC.len() > 0):
        # Step 1 on Fei's paper. Use the information on the current treatment plan to formulate and solve an instance of the PP
        data.calcDose()
        data.calcGradientandObjValue()
        pstarlist, lmlist, rmlist, bestApertureIndexlist, kmeasurelist, goodaperturesreceived, PerimeterList, AreaList, listAper = PricingProblem(C, C2, C3, vmax, data.speedlim, beamletwidth, K)
        bigListofApertures.append(listAper)
        # Step 2. If the optimal value of the PP is nonnegative**, go to step 5. Otherwise, denote the optimal solution to the
        # PP by c and Ac and replace caligraphic C and A = Abar, k \in caligraphicC
        print('pstar0', pstarlist[0], bestApertureIndexlist[0])
        if pstarlist[0] >= 0:
            #This choice includes the case when no aperture was selected
            print('Program finishes because no aperture was selected to enter')
            break
        else:
            for acounter in range(goodaperturesreceived):
                lm = lmlist.pop(0)
                rm = rmlist.pop(0)
                bestApertureIndex = bestApertureIndexlist.pop(0)
                kmeasure = kmeasurelist.pop(0)
                Perimeter = PerimeterList.pop(0)
                Area = AreaList.pop(0)
                # Update caligraphic C.
                data.caligraphicC.insertAngle(bestApertureIndex, data.notinC(bestApertureIndex))
                data.notinC.removeIndex(bestApertureIndex)
                # Solve the instance of the RMP associated with caligraphicC and Ak = A_k^bar, k \in
                beamList[bestApertureIndex].llist = lm
                beamList[bestApertureIndex].rlist = rm
                beamList[bestApertureIndex].KellyMeasure = kmeasure
                beamList[bestApertureIndex].Perimeter = Perimeter
                beamList[bestApertureIndex].Area = Area
                allbeamshapes.append((lm, rm, kmeasure, Perimeter, Area, bestApertureIndex))
                # Precalculate the aperture map to save times.
                data.openApertureMaps[bestApertureIndex], data.diagmakers[bestApertureIndex], data.strengths[bestApertureIndex] = updateOpenAperture(bestApertureIndex, K)
            data.rmpres = solveRMC(data.YU, 10)
            print('Solved Restricted Master Problem')
            ## List of apertures that was removed in this iteration
            IndApRemovedThisStep = []
            entryCounter = 0
            for thisindex in rangenumbeams:
                if thisindex in data.caligraphicC.loc: #Only activate what is an aperture
                    ## THIS PART IS DEACTIVATED RIGHT NOW BECAUSE ELIMINATIONPHASE = FALSE
                    if (data.rmpres.x[thisindex] < eliminationThreshold) & (eliminationPhase) & (not refinementloops):
                        ## Maintain a tally of apertures that are being removed
                        entryCounter += 1
                        IndApRemovedThisStep.append(thisindex)
                        # Remove from caligraphicC and add to notinC
                        data.notinC.insertAngle(beamList[thisindex].location, beamList[thisindex].angle)
                        data.caligraphicC.removeIndex(thisindex)
            print('Indapremoved this step:', IndApRemovedThisStep)
            ## Save all apertures that were removed in this step
            data.listIndexofAperturesAddedEachStep.append(bestApertureIndexlist)
            data.listIndexofAperturesRemovedEachStep.append(IndApRemovedThisStep)
            print('All apertures added in each step:', data.listIndexofAperturesAddedEachStep)
            print('All apertures removed in each step:', data.listIndexofAperturesRemovedEachStep)
            optimalvalues.append(data.rmpres.fun)
            plotcounter = plotcounter + 1
            if eliminationPhase | refinementloops:
                printresults(len(data.caligraphicC.loc), dropbox + '/Research/VMAT/casereader/outputGraphics/' + caseis, C)
            else:
                printresults(len(data.caligraphicC.loc), dropbox + '/Research/VMAT/casereader/outputGraphics/NOELIMINATIONPHASE'  + caseis, C)
        print('caligraphicC:', data.caligraphicC.angle)
        print('notinC: ', data.notinC.angle)
    # Set up an order to go refining one by one.
    PIK = "outputGraphics/allbeamshapesbefore-save-" +  caseis + '-' + str(C) + ".pickle"
    with open(PIK, "wb") as f:
        pickle.dump(allbeamshapes, f, pickle.HIGHEST_PROTOCOL)
    f.close()
    mytime.newloop()
    PIK = "outputGraphics/pickleApertures-1stPass-C-" + caseis + '-' + str(C) + "-save.dat"
    with open(PIK, "wb") as f:
        pickle.dump(bigListofApertures, f, pickle.HIGHEST_PROTOCOL)
    f.close()
    if refinementloops:
        phi = 0
        refinementLoopCounter = 0
        C = CSave
        while refinementLoopCounter < 5:
            refinementLoopCounter += 1
            # Create the list of contributions
            contributionsPengList = []
            oldObjectiveValue = data.rmpres.fun  #Base to compare against
            for mynumbeam in rangenumbeams:
                #print('gonna call contributionofBeam with:', mynumbeam, 'oo', oldObjectiveValue, 'C', C, 'C2', C2, 'C3', C3, 'vmax', vmax, 'bw', beamletwidth, 'K', K)
                contribs = contributionofBeam(mynumbeam, oldObjectiveValue, C, C2, C3, vmax, beamletwidth, K)
                print('here')
                contributionsPengList.append(contribs)
                print('there')
                #bigListofApertures.append(listAper)
            # pengList will contain a list of the different apertures in decreasing order of contribution
            print('Contributions PengList. This one is not ordered:', contributionsPengList)
            pengList = [x for _, x in sorted(zip(contributionsPengList, rangenumbeams), key=lambda pair: pair[0], reverse=False)]
            print('pengList ordenada: ', pengList)
            for refaper in pengList:
                print('rechecking aperture:', refaper)
                # Remove aperture from the set temporarily
                if refaper in data.caligraphicC.loc:
                    data.notinC.insertAngle(beamList[refaper].location, beamList[refaper].angle)
                    data.caligraphicC.removeIndex(refaper)
                    data.openApertureMaps[refaper], data.diagmakers[refaper], data.strengths[refaper] = updateOpenAperture(refaper, K)
                # Calculate dose and gradients
                data.calcDose()
                data.calcGradientandObjValue()
                # Select a new aperture for that particular location
                pstar, lm, rm, bestApertureIndex, kmeasure, perimeter, area = refinementPricingProblem(refaper, C, C2, C3, vmax, data.speedlim, beamletwidth, K)
                # Update caligraphic C. why?
                if pstar >= 0:
                    phi = phi + 1
                    if 5 == phi:
                        break
                    else:
                        continue #No aperture can make things better (I'm doing nothing though)
                else:
                    phi = 0
                data.caligraphicC.insertAngle(bestApertureIndex, data.notinC(bestApertureIndex))
                data.notinC.removeIndex(bestApertureIndex)
                # Solve the instance of the RMP associated with caligraphicC and Ak = A_k^bar, k \in
                beamList[bestApertureIndex].llist = lm
                beamList[bestApertureIndex].rlist = rm
                allbeamshapes.append((lm, rm, kmeasure, perimeter, area, bestApertureIndex))
                beamList[bestApertureIndex].KellyMeasure = kmeasure
                beamList[bestApertureIndex].Perimeter = perimeter
                beamList[bestApertureIndex].Area = area
                # Precalculate the aperture map to save times.j
                data.openApertureMaps[bestApertureIndex], data.diagmakers[bestApertureIndex], data.strengths[bestApertureIndex] = updateOpenAperture(bestApertureIndex, K)
                data.rmpres = solveRMC(data.YU, 100)
                printresults(len(data.caligraphicC.loc), dropbox + '/Research/VMAT/casereader/outputGraphics/' + caseis, C)
            print("Let's see round of refinement", refinementLoopCounter)
            print('oldObjectiveValue Comparison', oldObjectiveValue, data.rmpres.fun)
            print('caligraphicC:', data.caligraphicC.angle)
            print('notinC: ', data.notinC.angle)
            print('new objective value', data.rmpres.fun)
            #This is only here because I want to save time
            PIK = "outputGraphics/allbeamshapes-save-" + caseis + str(C) + ".pickle"
            with open(PIK, "wb") as f:
                pickle.dump(allbeamshapes, f, pickle.HIGHEST_PROTOCOL)
            f.close()
            PIK = "outputGraphics/beamList-save-" + caseis + str(C) + ".pickle"
            with open(PIK, "wb") as f:
                pickle.dump(beamList, f, pickle.HIGHEST_PROTOCOL)
            f.close()

            mynumbeams = beam.numBeams
            M = beam.M
            N = beam.N
            datasave = [mynumbeams, data.rmpres.x, C, C2, C3, vmax, data.speedlim, data.RU, data.YU, M, N, beamList,
                        data.maskValue, data.currentDose, data.currentIntensities, structure.numStructures,
                        structureList, data.rmpres.fun, data.quadHelperThresh, data.quadHelperOver, data.quadHelperUnder,
                        mytime, socket.gethostname(), beamlet.XSize, beamlet.YSize]
            PIK = "outputGraphics/pickle-C-" + caseis + '-' + str(C) + "-save.dat"
            with open(PIK, "wb") as f:
                pickle.dump(datasave, f, pickle.HIGHEST_PROTOCOL)
            f.close()
            PIK = "outputGraphics/pickleApertures-C-" + caseis + '-' + str(C) + "-save.dat"
            with open(PIK, "wb") as f:
                pickle.dump(bigListofApertures, f, pickle.HIGHEST_PROTOCOL)
            f.close()
            if np.abs((oldObjectiveValue - data.rmpres.fun)/oldObjectiveValue) < 0.1 and refinementLoopCounter > 1: #epsilon
                print('refinement produced less than 0.1 percent improvement in the last iteration')
                print('caligraphicC:', data.caligraphicC.angle)
                print('notinC: ', data.notinC.angle)
                mytime.newloop()
                break
        if eliminationPhase | refinementloops:
            printresults(len(data.caligraphicC.loc), dropbox + '/Research/VMAT/casereader/outputGraphics/' + caseis, C)
        else:
            printresults(len(data.caligraphicC.loc), dropbox + '/Research/VMAT/casereader/outputGraphics/NOELIMINATIONPHASE' + caseis,
                                 C)
        mytime.newloop()
    plotApertures(C)
    return(data.rmpres.x)

# The next function prints DVH values
def printresults(iterationNumber, myfolder, Cvalue):
    if ('arc-ts.umich.edu' == socket.gethostname().split('.', 1)[-1]):
        myfolder = "/home/wilmer/Dropbox/Research/outputGraphics"
    data.maskValue = np.array([int(i) for i in data.maskValue])
    print('Starting to Print Result DVHs')
    zvalues = data.currentDose
    maxDose = max([float(i) for i in zvalues])
    dose_resln = 0.1
    dose_ub = maxDose + 10
    bin_center = np.arange(0,dose_ub,dose_resln)
    # Generate holder matrix
    dvh_matrix = np.zeros((structure.numStructures, len(bin_center)))
    # iterate through each structure
    for s in data.structureIndexUsed:
        doseHolder = sorted(zvalues[[i for i,v in enumerate(data.maskValue & 2**s) if v > 0]])
        if 0 == len(doseHolder):
            continue
        histHolder, garbage = np.histogram(doseHolder, bin_center)
        histHolder = np.append(histHolder, 0)
        histHolder = np.cumsum(histHolder)
        dvhHolder = 1-(np.matrix(histHolder)/max(histHolder))
        dvh_matrix[s,] = dvhHolder
    print('matrix shape:', dvh_matrix.shape)
    dvh_matrix = dvh_matrix[data.structureIndexUsed,]
    print(dvh_matrix.shape)

    myfig = pylab.plot(bin_center, dvh_matrix.T, linewidth = 2.0)
    if caseis == "spine360":
        pylab.xlim(0, 60)
    elif caseis == "brain360":
        pylab.xlim(0, 86)
    elif caseis == "braiF360":
        pylab.xlim(0, 44)
    else:
        pylab.xlim(0, 70)
    plt.grid(True)
    plt.xlabel('Dose Gray')
    plt.ylabel('Fractional Volume')
    plt.title('DVH-VMAT-C-' + str(Cvalue) + '-beams-' + str(iterationNumber) + caseis + '.png')
    plt.legend(structureNames, prop={'size':9})
    if not ptvpriority:
        plt.savefig(myfolder + 'DVH-VMAT-C-' + str(Cvalue) + '-beams-' + str(iterationNumber) + caseis + '.png')
    else:
        plt.savefig(myfolder + 'DVH-VMAT-PTVPRIORITY-C-' + str(Cvalue) + '-beams-' + str(iterationNumber) + caseis + '.png')
        plt.title(caseis + "-PTV-Priority")
    plt.close()

def plotApertures(C):
    magnifier = 50
    ## Plotting apertures
    xcoor = math.ceil(math.sqrt(beam.numBeams))
    ycoor = math.ceil(math.sqrt(beam.numBeams))
    nrows, ncols = beam.M, beam.N
    print('numbeams', beam.numBeams)
    for mynumbeam in range(0, beam.numBeams):
        lmag = [llim + 1 / numsubdivisions for llim in beamList[mynumbeam].llist]
        rmag = beamList[mynumbeam].rlist
        ## Convert the limits to hundreds.
        for posn in range(0, len(lmag)):
            lmag[posn] = int(magnifier * lmag[posn])
            rmag[posn] = int(magnifier * rmag[posn])
        image = -1 * np.ones(magnifier * nrows * ncols)
            # Reshape things into a 9x9 grid
        image = image.reshape((nrows, magnifier * ncols))
        for i in range(0, beam.M):
            image[i, lmag[i]:(rmag[i]-1)] = data.rmpres.x[mynumbeam] #intensity assignment
        image = np.repeat(image, 7*magnifier, axis = 0) # Repeat. Otherwise the figure will look flat like a pancake
        image[0,0] = data.YU # In order to get the right list of colors
        # Set up a location where to save the figure
        fig = plt.figure(1)
        plt.subplot(ycoor,xcoor, mynumbeam + 1)
        cmapper = plt.get_cmap("autumn_r")
        cmapper.set_under('black', 1.0)
        plt.imshow(image, cmap = cmapper, vmin = 0.0, vmax = data.YU)
        plt.axis('off')
    fig.savefig(dropbox + '/Research/VMAT/casereader/outputGraphics/plotofapertures' + caseis + str(C) + '.png')
    plt.close()
    
start = time.clock()
data = problemData(numbeams)
allbeamshapes = [] #This will be a list of tuples
print('Assigning problemData', time.clock() - start)
if memorySaving:
    data.DlistT, data.voxelsUsed = getDmatrixPiecesMemorySaving()
else: #Not memorysaving doesn't work anymore
    vlist, blist, dlist = getDmatrixPieces()
    data.voxelsUsed = np.unique(vlist)
    starttime = time.clock()
    DmatBig = sparse.csr_matrix((dlist, (blist, vlist)), shape=(beamlet.numBeamlets, voxel.numVoxels), dtype=float)
    del vlist
    del blist
    del dlist
    print('Assigned DmatBig in seconds: ', time.clock() - starttime)
    starttime = time.clock()
    data.DlistT = [DmatBig[beamList[i].StartBeamletIndex:(beamList[i].EndBeamletIndex+1),].transpose() for i in range(beam.numBeams)]
    print('Assigned DmatBig in seconds: ', time.clock() - starttime)
print('total time reading dose to points:', time.clock() - start)
start = time.clock()
strsUsd = set([])
strsIdxUsd = set([])
for v in data.voxelsUsed:
    strsUsd.add(voxelList[v].StructureId)
    strsIdxUsd.add(structureDict[voxelList[v].StructureId])
data.structuresUsed = list(strsUsd)
data.structureIndexUsed = list(strsIdxUsd)
print('structures used in no particular order:', data.structureIndexUsed)
structureNames = []
for s in data.structureIndexUsed:
    structureNames.append(structureList[s].Id) #Names have to be organized in this order or it doesn't work
print(structureNames)

mytime.readingtime()
if debugmode:
    finalintensities = column_generation(CValue, numsubdivisions, mytime)
else:
    finalintensities = column_generation(CValue, numsubdivisions, mytime) # Second argument here determines how many times I will cut each beamlet artificially
averageNW = 0.0
averageW = 0.0
for i in range(beam.numBeams):
    averageNW += beamList[i].KellyMeasure
    averageW += beamList[i].KellyMeasure * finalintensities[i]

print('averageW:', averageW/beam.numBeams)
print('averageNW:', averageNW/beam.numBeams)
PIK = "outputGraphics/" + caseis + "allbeamshapes-save-" + str(CValue) + ".pickle"
with open(PIK, "wb") as f:
    pickle.dump(allbeamshapes, f, pickle.HIGHEST_PROTOCOL)
f.close()
PIK = "outputGraphics/" + caseis + "beamList-save-" + str(CValue) + ".pickle"
with open(PIK, "wb") as f:
    pickle.dump(beamList, f, pickle.HIGHEST_PROTOCOL)
f.close()
sys.exit()
