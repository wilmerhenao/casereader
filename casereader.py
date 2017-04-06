#! /opt/intel/intelpython35/bin/python3.5

import dose_to_points_data_pb2
import sys
import os
import time
import gc
import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from multiprocessing import Pool
from functools import partial
import pickle

numcores = 6

gc.enable()
datalocation = "/mnt/fastdata/Data/spine"
datalocation = "/home/wilmer/Dropbox/Data/spine/"

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
            self.threshold = 70
            self.overdoseCoeff = 0.1
            self.underdoseCoeff = 10
        else:
            structure.numOARs += 1
            self.threshold = 10
            self.overdoseCoeff = 5
            self.underdoseCoeff = 0.0
        structure.numStructures += 1

class beam(object):
    numBeams = 0
    JawX1 = None
    JawX2 = None
    JawY1 = None
    JawY2 = None
    def __init__(self, sthing):
        # Update the static counter and other variables
        self.angle = 10 * beam.numBeams # Notice that this one changes per beam
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
        self.M = 8
        self.leftEdge = -36
        self.rightEdge = 35
        self.llist = [self.leftEdge] * self.M # One position more or less after the edges given by XStart, YStart in beamlets
        self.rlist = [self.rightEdge] * self.M


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
        self.XStart = sthing.XStart
        self.YStart = sthing.YStart
        beamlet.XSize = sthing.XSize
        beamlet.YSize = sthing.YSize
        self.belongsToBeam = None

## apertureList is a class definition of locs and angles that is always sorted.
# Its attributes are loc which is the numeric location; It has range 0 to 36 for
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
    ## Returns the length of this instantiation without the need to pass parameters.
    def len(self):
        return(len(self.loc))
    ## Returns True if the list is empty; otherwise returns False.
    def isEmpty(self):
        if 0 == len(self.loc):
            return(True)
        else:
            return(False)

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
    filepaths.sort(key = lambda filename: filename[1], reverse = reverse)
    # Re-populate list with just filenames
    for i in range(len(filepaths)):
        filepaths[i] = filepaths[i][0]
    return(filepaths)

datafiles = get_files_by_file_size(datalocation)
print(datafiles)

# The first file will contain all the structure data, the rest will contain pointodoses.
dpdata = dose_to_points_data_pb2.DoseToPointsData()

# Start with reading structures, numvoxels and all that.
f = open(datafiles.pop(0), "rb")
dpdata.ParseFromString(f.read())
f.close()
#----------------------------------------------------------------------------------------
# Get the data about the structures
numstructs = len(dpdata.Structures)
structureList = []
print("Reading in structures")
for s in range(numstructs):
    print('Reading:', dpdata.Structures[s].Id)
    structureList.append(structure(dpdata.Structures[s]))
print('Number of structures:', structure.numStructures, '\nNumber of Targets:', structure.numTargets,
      '\nNumber of OARs', structure.numOARs)
#----------------------------------------------------------------------------------------
## Get the data about beamlets
numbeamlets = len(dpdata.Beamlets)
beamletList = []
print('Reading in beamlet data:')
for blt in range(numbeamlets):
    beamletList.append(beamlet(dpdata.Beamlets[blt]))
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
    #print(beamList[b].JawX1, beamList[b].JawX2, beamList[b].JawY1, beamList[b].JawY2)
print('There are a total of beams:', beam.numBeams)
print('beamlet data was updated so they point to their owner')
#for b in range(numbeams):
#    print(beamList[b].angle)
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
gc.collect()
#----------------------------------------------------------------------------------------
## Get the point to dose data in a sparse matrix
start = time.time()

def getDmatrixPieces():
    ## Initialize vectors for voxel component, beamlet component and dose
    vcps = []
    bcps = []
    dcps = []

    ## Read the beams now.
    counter = 0
    for fl in datafiles[0:1]:
        counter+=1
        print('reading datafile:', counter,fl)
        dpdata = dose_to_points_data_pb2.DoseToPointsData()

        f = open(fl, "rb")
        dpdata.ParseFromString(f.read())
        f.close()

        for dp in dpdata.PointDoses:
            numbeamletDoses = len(dp.BeamletDoses)
            vcps += [dp.Index] * numbeamletDoses # This is the voxel we're dealing with
            bcps += list(map(lambda val: val.Index, dp.BeamletDoses))
            dcps += list(map(lambda val: val.Dose, dp.BeamletDoses))
        gc.collect()
    return(vcps, bcps, dcps)

vlist, blist, dlist = getDmatrixPieces()

killlist = range(17,23)

def eliminateListofStructures(killlist, vlist, blist, dlist):
    vkills = []
    for i in killlist:
        vkills += list(range(structureList[i].StartPointIndex, (structureList[i].EndPointIndex+1)))
        structureList[i].isKilled = True
    vlist = [vlist[ind] for ind, x in enumerate(blist) if x not in vkills]
    dlist = [dlist[ind] for ind, x in enumerate(blist) if x not in vkills]
    blist = [x for ind, x in enumerate(blist) if x not in vkills]
    return(vlist, blist, dlist)

#print(len(vlist), len(blist), len(dlist))
#vlist, blist, dlist = eliminateListofStructures(killlist, vlist, blist, dlist)
#print('after')
#print(len(vlist), len(blist), len(dlist))
DmatBig = sparse.csr_matrix((dlist, (blist, vlist)), shape=(beamlet.numBeamlets, voxel.numVoxels), dtype=float)
del vlist
del blist
del dlist

print('total time reading dose to points:', time.time() - start)
#------------------------------------------------------------------------------------------------------------------

class problemData():
    def __init__(self):
        self.kappa = []
        self.notinC = apertureList()
        self.caligraphicC = apertureList()
        self.currentDose = np.zeros(voxel.numVoxels, dtype = float)
        self.quadHelperThresh = np.empty(voxel.numVoxels, dtype = float)
        self.quadHelperOver = np.empty(voxel.numVoxels, dtype = float)
        self.quadHelperUnder = np.empty(voxel.numVoxels, dtype = float)
        self.setQuadHelpers(structureList, voxelList)
        self.openApertureMaps = [[] for i in range(beam.numBeams)]
        self.diagmakers = [[] for i in range(beam.numBeams)]
        self.strengths = [[] for i in range(beam.numBeams)]
        self.DlistT = [DmatBig[BeamList[i].StartBeamletIndex:BeamList[i].EndBeamletIndex,].transpose() for i in range(beam.numBeams)]

    def setQuadHelpers(self, sList, vList):
        for i in range(voxel.numVoxels):
            sid = vList[i].StructureId # Find structure of this particular voxel
            self.quadHelperThresh[i] = sList[sid].threshold
            self.quadHelperOver[i] = sList[sid].overdoseCoeff
            self.quadHelperUnder[i] = sList[sid].underdoseCoeff

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

        self.objectiveValue = sum(oDoseObj + uDoseObj)

        oDoseObjGl = 2 * oDoseObjCl * self.quadHelperOver
        uDoseObjGl = 2 * uDoseObjCl * self.quadHelperUnder
        # Notice that I use two types of gradients. One for voxels and one for apertures
        self.voxelgradient = 2 * (oDoseObjGl - uDoseObjGl)
        self.aperturegradient = (np.asmatrix(self.voxelgradient) * self.dZdK).transpose()

## Find geographical location of the ith row in aperture index given by index. This is really only a problem for the
# HN case from the CORT database
# Input:    i:     Row
#           index: Index of this aperture
# Output:   validbeamlets ONLY contains those beamlet INDICES for which we have available data in this beam angle
#           validbeamletspecialrange is the same as validbeamlets but appending the endpoints
def fvalidbeamlets(index):
    validbeamlets = np.array(range(beamList[index].leftEdge + 1, beamList[index].rightEdge - 1))[validbeamletlogic]
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
# rcm = vector of right limits in the previous aperture
# rcp = vector of right limits in the previous aperture
# N = Number of beamlets per row
# M = Number of rows in an aperture
# thisApertureIndex = index location in the set of apertures that I have saved.
def PPsubroutine(C, C2, C3, angdistancem, angdistancep, vmax, speedlim, predec, succ, thisApertureIndex, bw):
    # Get the slice of the matrix that I need
    D = DmatBig[beamList[thisApertureIndex].StartBeamletIndex:beamList[thisApertureIndex].EndBeamletIndex,]
    M = beamList[thisApertureIndex].M
    leftEdge = beamList[thisApertureIndex].leftEdge
    rightEdge = beamList[thisApertureIndex].rightEdge
    b = bw
    # vmaxm and vmaxp describe the speeds that are possible for the leaves from the predecessor and to the successor
    vmaxm = vmax
    vmaxp = vmax
    # Arranging the predecessors and the succesors.
    #Predecessor left and right indices
    if type(predec) is list:
        lcm = [leftEdge] * M
        rcm = [rightEdge] * M
        # If there is no predecessor is as if the pred. speed was infinite
        vmaxm = np.inf
    else:
        lcm = beamList[predec].llist
        rcm = beamList[predec].rlist

    #Succesors left and right indices
    if type(succ) is list:
        lcp = [leftEdge] * M
        rcp = [rightEdge] * M
        # If there is no successor is as if the succ. speed was infinite.
        vmaxp = np.inf
    else:
        lcp = beamList[succ].llist
        rcp = beamList[succ].rlist

    validbeamlets, validbeamletspecialrange = fvalidbeamlets(thisApertureIndex)
    # First handle the calculations for the first row
    beamGrad = D * data.voxelgradient
    # Keep the location of the most leaf

    nodesinpreviouslevel = 0
    oldflag = 0
    posBeginningOfRow = 1
    thisnode = 0
    # Max beamlets per row
    bpr = rightEdge - leftEdge + 2
    networkNodesNumber = bpr * bpr + M * bpr * bpr + bpr * bpr # An overestimate of the network nodes in this network
    # Initialization of network vectors. This used to be a list before
    lnetwork = np.zeros(networkNodesNumber, dtype = np.int) #left limit vector
    rnetwork = np.zeros(networkNodesNumber, dtype = np.int) #right limit vector
    mnetwork = np.ones(networkNodesNumber, dtype = np.int) #Only to save some time in the first loop
    wnetwork = np.empty(networkNodesNumber, dtype = np.float) # Weight Vector
    wnetwork[:] = np.inf
    dadnetwork = np.zeros(networkNodesNumber, dtype = np.int) # Dad Vector. Where Dad is the combination of (l,r) in previous row
    # Work on the first row perimeter and area values
    leftrange = range(math.ceil(max(leftEdge, lcm[0] - vmaxm * (angdistancem/speedlim)/bw , lcp[0] - vmaxp * (angdistancep/speedlim)/bw )), 1 + math.floor(min(rightEdge - 1, lcm[0] + vmaxm * (angdistancem/speedlim)/bw , lcp[0] + vmaxp * (angdistancep/speedlim)/bw )))
    # Check if unfeasible. If it is then assign one value but tell the result to the person running this
    if (0 == len(leftrange)):
        midpoint = (angdistancep * lcm[0] + angdistancem * lcp[0])/(angdistancep + angdistancem)
        leftrange = np.arange(midpoint, midpoint + 1)
        ##print('constraint leftrange at level ' + str(0) + ' aperture ' + str(thisApertureIndex) + ' could not be met', 'ERROR Report: lcm[0], angdistancem, lcp[0], angdistancep', lcm[0], angdistancem, lcp[0], angdistancep, '\nFull left limits, lcp, lcm:', lcp, lcm, 'm: ', 0, 'predecesor: ', predec, 'succesor: ', succ)
    for l in leftrange:
        rightrange = range(math.ceil(max(l + 1, rcm[0] - vmaxm * (angdistancem/speedlim)/bw , rcp[0] - vmaxp * (angdistancep/speedlim)/bw )), 1 + math.floor(min(N, rcm[0] + vmaxm * (angdistancem/speedlim)/bw , rcp[0] + vmaxp * (angdistancep/speedlim)/bw )))
        if (0 == len(rightrange)):
            midpoint = (angdistancep * rcm[0] + angdistancem * rcp[0])/(angdistancep + angdistancem)
            rightrange = np.arange(midpoint, midpoint + 1)
            ##print('constraint rightrange at level ' + str(0) + ' aperture ' + str(thisApertureIndex) + ' could not be met', 'ERROR Report: lcm[0], angdistancem, lcp[0], angdistancep', lcm[0], angdistancem, lcp[0], angdistancep, '\nFull left limits, rcp, rcm:', rcp, rcm, 'm: ', 0, 'predecesor: ', predec, 'succesor: ', succ)
        for r in rightrange:
            thisnode += 1
            nodesinpreviouslevel += 1
            # First I have to make sure to add the beamlets that I am interested in
            if(l + 1 < r): # prints r numbers starting from l + 1. So range(3,4) = 3
                ## Take integral pieces of the dose component
                possiblebeamletsthisrow = np.intersect1d(range(int(np.ceil(l+1)),int(np.floor(r))), validbeamlets) - min(validbeamlets)
                ## Calculate dose on the sides
                DoseSide = -((np.ceil(l+1) - (l+1)) * beamGrad[int(np.floor(l+1))] + (r - np.floor(r)) * beamGrad[int(np.ceil(r))])
                if (len(possiblebeamletsthisrow) > 0):
                    Dose = -beamGrad[ possiblebeamletsthisrow ].sum()
                    weight = C * ( C2 * (r - l) - C3 * b * (r - l)) - Dose + 10E-10 * (r-l) + DoseSide# The last term in order to prefer apertures opening in the center
                else:
                    weight = C * ( C2 * (r - l) - C3 * b * (r - l)) + 10E-10 * (r-l) + DoseSide
            else:
                weight = 0.0
            # Create node (1,l,r) in array of existing nodes and update the counter
            # Replace the following expression
            lnetwork[thisnode] = l
            rnetwork[thisnode] = r
            wnetwork[thisnode] = weight
            # dadnetwork and mnetwork don't need to be changed here for obvious reasons
    posBeginningOfRow += nodesinpreviouslevel
    leftmostleaf = len(validbeamlets) - 1 # Position in python position(-1) of the leftmost leaf
    # Then handle the calculations for the m rows. Nodes that are neither source nor sink.
    for m in range(1,M):
        # Get the beamlets that are valid in this row in particular (all others are still valid but are zero)
        validbeamlets, validbeamletspecialrange = fvalidbeamlets(thisApertureIndex)
        oldflag = nodesinpreviouslevel
        nodesinpreviouslevel = 0
        # And now process normally checking against valid beamlets
        leftrange = range(math.ceil(max(leftEdge, lcm[m] - vmaxm * (angdistancem/speedlim)/bw , lcp[m] - vmaxp * (angdistancep/speedlim)/bw )), 1 + math.floor(min(rightEdge - 1, lcm[m] + vmaxm * (angdistancem/speedlim)/bw , lcp[m] + vmaxp * (angdistancep/speedlim)/bw )))
        # Check if unfeasible. If it is then assign one value but tell the result to the person running this
        if(0 == len(leftrange)):
            midpoint = (angdistancep * lcm[m] + angdistancem * lcp[m])/(angdistancep + angdistancem)
            leftrange = np.arange(midpoint, midpoint + 1)
        for l in leftrange:
            rightrange = range(math.ceil(max(l + 1, rcm[m] - vmaxm * (angdistancem/speedlim)/bw , rcp[m] - vmaxp * (angdistancep/speedlim)/bw )), 1 + math.floor(min(N, rcm[m] + vmaxm * (angdistancem/speedlim)/bw , rcp[m] + vmaxp * (angdistancep/speedlim)/bw )))
            if (0 == len(rightrange)):
                midpoint = (angdistancep * rcm[m] + angdistancem * rcp[m])/(angdistancep + angdistancem)
                rightrange = np.arange(midpoint, midpoint + 1)
                ##print('constraint rightrange at level ' + str(m) + ' aperture ' + str(thisApertureIndex) + ' could not be met', 'ERROR Report: lcm[m], angdistancem, lcp[m], angdistancep', lcm[m], angdistancem, lcp[m], angdistancep, '\nFull left limits, rcp, rcm:', rcp, rcm, 'm: ', m, 'predecesor: ', predec, 'succesor: ', succ)
            for r in rightrange:
                nodesinpreviouslevel += 1
                thisnode += 1
                # Create node (m, l, r) and update the level counter
                lnetwork[thisnode] = l
                rnetwork[thisnode] = r
                mnetwork[thisnode] = m
                wnetwork[thisnode] = np.inf
                # Select only those beamlets that are possible in between the (l,r) limits.
                possiblebeamletsthisrow = np.intersect1d(range(int(np.ceil(l+1)),int(np.floor(r))), validbeamlets) + leftmostleaf# - min(validbeamlets)
                DoseSide = -((np.ceil(l+1) - (l+1)) * beamGrad[int(np.floor(l+1))] + (r - np.floor(r)) * beamGrad[int(np.ceil(r))])
                if(len(possiblebeamletsthisrow) > 0):
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
        leftmostleaf = len(validbeamlets) + leftmostleaf
    # thisnode gets augmented only 1 because only the sink node will be added
    thisnode += 1
    for mynode in (range(posBeginningOfRow - nodesinpreviouslevel, posBeginningOfRow )): # +1 because otherwise it could be empty
        weight = C * ( C2 * (rnetwork[mynode] - lnetwork[mynode] ))
        if(wnetwork[mynode] + weight <= wnetwork[thisnode]):
            wnetwork[thisnode] = wnetwork[mynode] + weight
            dadnetwork[thisnode] = mynode
            p = wnetwork[thisnode]
    thenode = thisnode # WILMER take a look at this
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


def parallelizationPricingProblem(i, C, C2, C3, vmax, speedlim, bw):
    thisApertureIndex = i
    print("analysing available aperture" , thisApertureIndex)
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

    # Find Numeric value of previous and next angle.
    p, l, r = PPsubroutine(C, C2, C3, angdistancem, angdistancep, vmax, speedlim, predec, succ, thisApertureIndex, bw)
    return(p,l,r,thisApertureIndex)

def PricingProblem(C, C2, C3, vmax, speedlim, bw):
    pstar = np.inf
    bestAperture = None
    print("Choosing one aperture amongst the ones that are available")
    # Allocate empty list with enough size for all l, r combinations
    global structureList

    partialparsubpp = partial(parallelizationPricingProblem, C=C, C2=C2, C3=C3, vmax=vmax, speedlim=speedlim, bw=bw)
    if __name__ == '__main__':
        pool = Pool(processes = numcores)              # process per MP
        respool = pool.map(partialparsubpp, data.notinC.loc)
    pool.close()
    pool.join()

    pvalues = np.array([result[0] for result in respool])
    indstar = np.argmin(pvalues)
    bestgroup = respool[indstar]
    pstar = bestgroup[0]
    l = bestgroup[1]
    r = bestgroup[2]
    bestApertureIndex = bestgroup[3]
    # Change the leaf positions for this particular beam
    structureList[bestApertureIndex].llist = l
    structureList[bestApertureIndex].rlist = r
    print("Best aperture was: ", bestApertureIndex)
    # Calculate Kelly's aperture measure
    Area = 0.0
    Perimeter = r[0] - l[0] + np.sign(r[0] - l[0]) # First part of the perimeter plus first edge
    for n in range(len(l)):
        Area += 0.5 * (r[n] - l[n]) * 0.5
    for n in range(1, len(l)):
        Area += 0.5 * (r[n] - l[n]) * 0.5
        Perimeter += np.sign(r[n] - l[n]) # Vertical part of the perimeter
        Perimeter += np.abs(l[n] - l[n-1]) + np.abs(r[n] - r[n-1]) - 2 * np.maximum(0, l[n-1] - r[n]) - 2 * np.maximum(0, l[n] - r[n - 1])
    Perimeter += r[len(r)-1] - l[len(l)-1] + np.sign(r[len(r)-1] - l[len(l)-1])
    Kellymeasure = Perimeter / Area
    return(pstar, l, r, bestApertureIndex)

## This function returns the set of available AND open beamlets for the selected aperture (i).
# The idea is to have everything ready and pre-calculated for the evaluation of the objective function in
# calcDose
# input: i is the index number of the aperture that I'm working on
# output: openaperturenp. the set of available AND open beamlets for the selected aperture. Doesn't contain fractional values
#         diagmaker. A vector that has a 1 in each position where an openaperturebeamlet is available.
# openaperturenp is read as openapertureMaps. A member of the VMAT_CLASS.
def updateOpenAperture(i):
    leftlimits = 0
    openaperture = []
    ## While openaperturenp contains positions, openapertureStrength contains proportion of the beamlets that's open.
    openapertureStrength = []
    D = DmatBig[BeamList[thisApertureIndex].StartBeamletIndex:BeamList[thisApertureIndex].EndBeamletIndex, ]
    diagmaker = np.zeros(D.shape[0], dtype = float)
    for m in range(0, len(beamList[i].llist)):
        # Find geographical values of llist and rlist.
        # Find geographical location of the first row.
        validbeamlets, validbeamletspecialrange = fvalidbeamlets(i)
        # First index in this row (only full beamlets included in this part

        ## Notice that indleft and indright below may be floats instead of just integers
        if (beamList[i].llist[m] >= min(validbeamlets) - 1):
            ## I subtract min validbeamlets bec. I want to find coordinates in available space
            ## indleft is where the edge of the left leaf ends. From there on there are photons.
            indleft = beamList[i].llist[m] + 1 + leftlimits - min(validbeamlets)
        else:
            # if the left limit is too far away to the left, just take what's available
            indleft = 0

        if (beamList[i].rlist[m] > max(validbeamlets)):
            # If the right limit is too far to the left, just grab the whole thing.
            indright = len(validbeamlets) + leftlimits
        else:
            if(beamList[i].rlist[m] >= min(validbeamlets)):
                ## indright is where the edgo of the right leaf ends. From there on there are photons
                indright = beamList[i].rlist[m] - 1 + leftlimits - min(validbeamlets)
            else:
                # Right limit is to the left of validbeamlets (This situation is weird)
                indright = 0

        # Keep the location of the letftmost leaf
        leftlimits += len(validbeamlets)
        if (np.floor(indleft) < np.ceil(indright)): ## Just a necessary logical check.
            first = True
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
            if strength > 0.01:
                ## Important: There is no need to check if there exists a last element because after all, you already
                # checked whe you entered the if loop above this one
                openapertureStrength[-1] = strength
                diagmaker[-1] = strength

            ## One last scenario. If only a little bit of the aperture is open (less than a beamlet and within one beamlet
            if 1 == int(np.ceil(indright)) - int(np.floor(indleft)):
                strength = indright - indleft
                openapertureStrength[-1] = strength
                diagmaker[-1] = strength
    openaperturenp = np.array(openaperture, dtype=int) #Contains indices of open beamlets in the aperture
    return(openaperturenp, diagmaker, openapertureStrength)

def calcObjGrad(x, user_data = None):
    data.currentIntensities = x
    data.calcDose()
    data.calcGradientandObjValue()
    return(data.objectiveValue, data.aperturegradient)

def solveRMC(YU):
    ## IPOPT SOLUTION
    start = time.time()
    numbe = data.caligraphicC.len()

    calcObjGrad(data.currentIntensities)
    # Create the boundaries making sure that the only free variables are the ones with perfectly defined apertures.
    boundschoice = []
    for thisindex in range(0, data.numbeams):
        if thisindex in data.caligraphicC.loc: #Only activate what is an aperture
            boundschoice.append((0, YU))
        else:
            boundschoice.append((0, 0))
    print(len(data.currentIntensities), len(boundschoice))
    res = minimize(calcObjGrad, data.currentIntensities, method='L-BFGS-B', jac = True, bounds = boundschoice, options={'ftol':1e-1, 'disp':5,'maxiter':200})

    print('Restricted Master Problem solved in ' + str(time.time() - start) + ' seconds')
    return(res)

def column_generation(C):
    C2 = 1.0
    C3 = 1.0
    eliminationPhase = False # Whether you want to eliminate redundant apertures at the end
    eliminationThreshold = 10E-3
    ## Maximum leaf speed
    vmax = 5 * 2.25 # 2.25 cms per second
    speedlim = 0.83  # Values are in the VMATc paper page 2968. 0.85 < s < 6
    ## Maximum Dose Rate
    RU = 10.0
    ## Maximum intensity
    YU = RU / speedlim
    beamletwidth = 0.2

    #Step 0 on Fei's paper. Set C = empty and zbar = 0. The gradient of numbeams dimensions generated here will not
    # be used, and therefore is nothing to worry about.
    # At the beginning no apertures are selected, and those who are not selected are all in notinC
    for j in range(beam.numBeams):
        data.notinC.insertAngle(beamList[j].location, beamList[j].angle)

    pstar = -np.inf
    plotcounter = 0
    optimalvalues = []
    while (pstar < 0) & (data.notinC.len() > 0):
        # Step 1 on Fei's paper. Use the information on the current treatment plan to formulate and solve an instance of the PP
        data.calcDose()
        data.calcGradientandObjValue()
        pstar, lm, rm, bestApertureIndex = PricingProblem(C, C2, C3, vmax, speedlim, beamletwidth)
        # Step 2. If the optimal value of the PP is nonnegative**, go to step 5. Otherwise, denote the optimal solution to the
        # PP by c and Ac and replace caligraphic C and A = Abar, k \in caligraphicC
        if pstar >= 0:
            #This choice includes the case when no aperture was selected
            print('Program finishes because no aperture was selected to enter')
            break
        else:
            # Update caligraphic C.
            data.caligraphicC.insertAngle(bestApertureIndex, data.notinC(bestApertureIndex))
            data.notinC.removeIndex(bestApertureIndex)
            # Solve the instance of the RMP associated with caligraphicC and Ak = A_k^bar, k \in
            data.llist[bestApertureIndex] = lm
            data.rlist[bestApertureIndex] = rm
            # Precalculate the aperture map to save times.
            data.openApertureMaps[bestApertureIndex], data.diagmakers[bestApertureIndex], data.strengths[bestApertureIndex] = updateOpenAperture(bestApertureIndex)
            rmpres = solveRMC(YU)
            ## List of apertures that was removed in this iteration
            IndApRemovedThisStep = []
            entryCounter = 0
            for thisindex in range(0, data.numbeams):
                if thisindex in data.caligraphicC.loc: #Only activate what is an aperture
                    if (rmpres.x[thisindex] < eliminationThreshold) & (eliminationPhase):
                        ## Maintain a tally of apertures that are being removed
                        entryCounter += 1
                        IndApRemovedThisStep.append(thisindex)
                        # Remove from caligraphicC and add to notinC
                        data.notinC.insertAngle(thisindex, data.pointtoAngle[thisindex])
                        data.caligraphicC.removeIndex(thisindex)
            print('Indapremoved this step:', IndApRemovedThisStep)
            ## Save all apertures that were removed in this step
            data.listIndexofAperturesRemovedEachStep.append(IndApRemovedThisStep)
            optimalvalues.append(rmpres.fun)
            plotcounter = plotcounter + 1
            printresults(plotcounter, '/home/wilmer/Dropbox/Research/VMAT/VMATwPenCode/outputGraphics/', C)
            #Step 5 on Fei's paper. If necessary complete the treatment plan by identifying feasible apertures at control points c
            #notinC and denote the final set of fluence rates by yk
    plotApertures(C)

# The next function prints DVH values
def printresults(iterationNumber, myfolder, Cvalue):
    numzvectors = 1
    maskValueFull = np.array([int(i) for i in data.fullMaskValue])
    print('Starting to Print Results')
    for i in range(0, numzvectors):
        zvalues = data.currentDose
        maxDose = max([float(i) for i in zvalues])
        dose_resln = 0.1
        dose_ub = maxDose + 10
        bin_center = np.arange(0,dose_ub,dose_resln)
        # Generate holder matrix
        dvh_matrix = np.zeros((data.numstructs, len(bin_center)))
        # iterate through each structure
        for s in range(0,data.numstructs):
            allNames[s] = allNames[s].replace("_VOILIST.mat", "")
            doseHolder = sorted(zvalues[[i for i,v in enumerate(maskValueFull & 2**s) if v > 0]])
            if 0 == len(doseHolder):
                continue
            histHolder = []
            carryinfo = 0
            histHolder, garbage = np.histogram(doseHolder, bin_center)
            histHolder = np.append(histHolder, 0)
            histHolder = np.cumsum(histHolder)
            dvhHolder = 1-(np.matrix(histHolder)/max(histHolder))
            dvh_matrix[s,] = dvhHolder

    myfig = pylab.plot(bin_center, dvh_matrix.T, linewidth = 2.0)
    plt.grid(True)
    plt.xlabel('Dose Gray')
    plt.ylabel('Fractional Volume')
    plt.title('Iteration: ' + str(iterationNumber))
    plt.legend(allNames,prop={'size':9})
    plt.savefig(myfolder + 'DVH-for-debugging-greedyVMAT.png')
    plt.close()

    voitoplot = [18, 7, 12, 13, 14, 15, 19, 20]
    dvhsub2 = dvh_matrix[voitoplot,]
    myfig2 = pylab.plot(bin_center, dvhsub2.T, linewidth = 1.0, linestyle = '--')
    pylab.xlim([0, 120])
    plt.grid(True)
    plt.xlabel('Dose Gray')
    plt.ylabel('Fractional Volume')
    plt.title('VMAT DVH Plot')
    #allNames.reverse()
    #plt.legend([allNames[i] for i in voitoplot])
    plt.legend(['PTV 1', 'PTV 2', 'Left Optic Nerve', 'Right Optic Nerve', 'Left Parotid', 'Right Parotid', 'Spinal Cord', 'Spinal Cord PRV'],prop={'size':9})
    plt.savefig(myfolder + 'DVHMAT' + str(Cvalue) + '.png')
    plt.close()

def plotApertures(C):
    magnifier = 100
    ## Plotting apertures
    xcoor = math.ceil(math.sqrt(data.numbeams))
    ycoor = math.ceil(math.sqrt(data.numbeams))
    nrows, ncols = M,N
    print('numbeams', data.numbeams)
    YU = (10.0 / 0.83)
    for mynumbeam in range(0, data.numbeams):
        lmag = data.llist[mynumbeam]
        rmag = data.rlist[mynumbeam]
        ## Convert the limits to hundreds.
        for posn in range(0, len(lmag)):
            lmag[posn] = int(magnifier * lmag[posn])
            rmag[posn] = int(magnifier * rmag[posn])
        image = -1 * np.ones(magnifier * nrows * ncols)
            # Reshape things into a 9x9 grid
        image = image.reshape((nrows, magnifier * ncols))
        for i in range(0, M):
            image[i, lmag[i]:(rmag[i]-1)] = data.rmpres.x[mynumbeam]
        image = np.repeat(image, magnifier, axis = 0) # Repeat. Otherwise the figure will look flat like a pancake
        image[0,0] = YU # In order to get the right list of colors
        # Set up a location where to save the figure
        fig = plt.figure(1)
        plt.subplot(ycoor,xcoor, mynumbeam + 1)
        cmapper = plt.get_cmap("autumn_r")
        cmapper.set_under('black', 1.0)
        plt.imshow(image, cmap = cmapper, vmin = 0.0, vmax = YU)
        plt.axis('off')
    fig.savefig('/home/wilmer/Dropbox/Research/VMAT/VMATwPenCode/outputGraphics/plotofapertures'+ str(C) + '.png')

data = problemData()
CValue = 1.0
column_generation(CValue)





sys.exit()
f = open(datafiles[0], "rb")
dpdata.ParseFromString(f.read())
f.close()



beamnumperbeam = [None] * numbeams
beamletsperbeam = [None] * numbeams
dijsPerBeam = [None] * numbeams

numX = 0
for b in range(numbeams):
    beamletsperbeam[b] = dpdata.Beams[b].EndBeamletIndex - dpdata.Beams[b].StartBeamletIndex
    numX = numX + beamletsperbeam[b]

# The variables below were not added in the original document by Carlos.

numpointdoses = len(dpdata.PointDoses)

print('control points:', len(dpdata.Beams))
print('Beamlets for the third beam:')
for bmlet in range(1120, 1680):
    print('whatisthis:',dpdata.Beamlets[bmlet].Index, dpdata.Beamlets[bmlet].BeamId, dpdata.Beamlets[bmlet].XSize, dpdata.Beamlets[bmlet].YSize, dpdata.Beamlets[bmlet].XStart, dpdata.Beamlets[bmlet].YStart)
beamletcounter = [None] * (numbeams + 1)

if (numpointdoses):
    print("This file has point doses!")
    print('total point doses so far:', len(dpdata.PointDoses))
    print("Generating Dose Matrix Dimensions")
    nDijs = 0
    for j in range(numvoxels):
        nDijs = nDijs + len(dpdata.PointDoses[j].BeamletDoses)
    print(nDijs)
    totaldijs = nDijs
    print('total non-zero Dijs:', totaldijs)
    print('Building Dose Matrix:')
