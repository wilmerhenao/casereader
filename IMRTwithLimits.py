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
import matplotlib.cm as cm

# List of organs that will be used
caseis = "spine360"
caseis = "lung360"
caseis = "brain360"
strengthThreshold = 0.05        # This threshold only works for the spine case. It gets overwritten by all other cases and by
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
        strengthThreshold = 0.05 #this is the default threshold value for this case

if "brain360" == caseis:
    structureListRestricted = [          1,            2,               3,           5,           6,            9,  11,     12,     15,           16]
    #limits                   [        PTV,          PTV,       Brainstem,       ONRVL     ONRVR,      chiasm,    eyeL,   eyeR,  BRAIN,      COCHLEA]
    #                                                                  60           54           54         54      40      40      10            40]
    threshold  =              [         58,           58,            10.0,        10.0,        10.0,         10.0, 30.0,  30.0,   10.0,         10.0]
    undercoeff =              [        100,         5E+3,             0.0,         0.0,         0.0,          0.0,  0.0,   0.0,    0.0,          0.0]
    overcoeff  =              [         50,          150,            5E-5,          5,          5.5,           5, 5E-7,  5E-1,    0.5,         2E-1]
    if 1 == len(sys.argv):
        strengthThreshold = 0.05 #this is the default threshold value for this case
    # above this threshold and the beamlet will be considered for insertion into the problem

numcores = 8
testcase = [i for i in range(0, 180, 1)]
fullcase = [i for i in range(180)]
debugmode = True
easyread = False
eliminateWrapper = False #True if you want to do the wrap of all hard limits. False, if you want individually.
numsubdivisions = 1

gc.enable()
## Find out the variables according to the hostname
datalocation = '~'
if 'radiation-math' == socket.gethostname():  # LAB
    datalocation = "/mnt/fastdata/Data/"+ caseis +"/by-Beam/"
    dropbox = "/mnt/datadrive/Dropbox"
    cutter = 44
    exec(open('VMATClasses.py').read())
elif 'sharkpool' == socket.gethostname():  # MY HOUSE
    datalocation = "/home/wilmer/Dropbox/Data/"+ caseis + "/by-Beam/"
    dropbox = "/home/wilmer/Dropbox"
elif ('DESKTOP-EA1PG8V' == socket.gethostname()):  # FLUX
    datalocation = "C:/Users/wihen/Data/" + caseis + "/by-Beam/"
    dropbox = "D:/Dropbox"
    cutter = 45
    numcores = 11
    if "lung360" == caseis:
        cutter = 44
    exec(open('VMATClasses.py').read())
    #execfile('VMATClasses.py')
else:
    datalocation = "C:/Users/wilmer/Dropbox/Research/mytempdata/by-Beam/"  # MY LAPTOP
    dropbox = "D:/Dropbox"
    cutter = 60
    execfile('VMATClasses.py')

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
## Get the dose to point data in a sparse matrix
start = time.time()

#----------------------------------------------------------------------------------------
def getDmatrixPiecesCutLimits():
    beamletMaxDoses = np.zeros(numbeamlets)
    beamletMaxVoxels = np.zeros(numbeamlets)
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
        sickranges = []
        for i in structureListRestricted:
            myranges.append(range(structureList[i].StartPointIndex, structureList[i].EndPointIndex))
            if i in structure.listTargets:
                sickranges.append(range(structureList[i].StartPointIndex, structureList[i].EndPointIndex))
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
            maxsickdose = 0.0
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
                        for sickrange in sickranges: # Keep track of doses to targets only
                            if k in sickrange:
                                maxcandidate = np.max(doses[k])
                                maxsickdose = max(maxcandidate, maxsickdose)

            # Create the matrix that goes in the list
            print(cutter, fl[cutter:])
            maxDoseThisAngle = np.max(newdcps)
            print('max all doses:', maxDoseThisAngle, maxsickdose)
            maxDoseThisAngle = maxsickdose
            print('maxDoseThisAngle:', maxDoseThisAngle)
            thisbeam = int(int(fl[cutter:].split('.')[0])/2)  #Find the beamlet in its coordinate space (not in Angle)
            initialBeamletThisBeam = beamList[thisbeam].StartBeamletIndex
            # Transform the space of beamlets to the new coordinate system starting from zero
            standardbcps = newbcps
            newbcps = [i - initialBeamletThisBeam for i in newbcps]
            # Wilmer Changed this part here
            #------------------------------
            bcps = []
            beamshape = np.reshape(np.zeros(bpb), (beam.M, beam.N))
            almacenv = []
            almacenb = []
            print('my target list:', structure.listTargets)
            time.sleep(1)
            for i, v in enumerate(newvcps):
                if beamletMaxDoses[standardbcps[i]] < newdcps[i]: #NOTICE: THIS IS OUTSIDE TARGETS
                    beamletMaxVoxels[standardbcps[i]] = v
                if np.log2(data.maskValue[v]) in structure.listTargets:
                    beamletMaxDoses[standardbcps[i]] = max(beamletMaxDoses[standardbcps[i]], newdcps[i])
                    if newdcps[i] > maxDoseThisAngle * strengthThreshold:
                        bcps.append(newbcps[i])
                        almacenv.append(v)
                        almacenb.append(standardbcps[i])
            hottestToTarget = np.reshape(beamletMaxDoses[beamList[thisbeam].StartBeamletIndex : (beamList[thisbeam].EndBeamletIndex + 1)], (beam.M, beam.N))
            plt.clf()
            plt.imshow(hottestToTarget, cmap=cm.plasma, origin = 'lower')
            #plt.imshow(hottestToTarget, cmap=cm.plasma)
            plt.title('Heatmap of contributions per beamlet to targets')
            plt.savefig('outputGraphics/brainview' + str(2 * thisbeam) + '.png')
            print(np.unique(almacenv))
            print(np.unique(almacenb))
            xloc = np.reshape(np.zeros(bpb), (beam.M, beam.N))
            yloc = np.reshape(np.zeros(bpb), (beam.M, beam.N))
            zloc = np.reshape(np.zeros(bpb), (beam.M, beam.N))
            np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
            # counterplus = 0
            # for i in range(beam.M):
            #     for j in range(beam.N):
            #         v = int(beamletMaxVoxels[ beamList[thisbeam].StartBeamletIndex + counterplus])
            #         xloc[i, j] = voxelList[v].X
            #         yloc[i, j] = voxelList[v].Y
            #         zloc[i, j] = voxelList[v].Z
            #         counterplus += 1
            # print('xloc', xloc)
            # print('yloc', yloc)
            # print('zloc', zloc)
            for i in np.unique(bcps):
                j = i % beam.N
                k = i // beam.N
                beamshape[k, j] = 1

            print(beamshape)

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
                PIK = 'IMRTEasyReadCases/testdump-' + caseis + 'proportionalThreshold' + str(strengthThreshold) + '.dat'
            else:
                PIK = 'IMRTEasyReadCases/fullcasedump-' + caseis + 'proportionalThreshold' + str(strengthThreshold) + '.dat'
            with open(PIK, "wb") as f:
                pickle.dump(datasave, f, pickle.HIGHEST_PROTOCOL)
            f.close()
    return(vlist, blist, dlist, beamletMaxDoses)

print('total time reading dose to points:', time.time() - start)

# ------------------------------------------------------------------------------------------------------------------

data = problemData()
print('done reading problemdata')
vlist, blist, dlist, maxDoses = getDmatrixPiecesCutLimits()

#plt.hist(maxDoses, bins = 20)
#plt.title('Maximum doses to target - ' + caseis)
#plt.savefig(dropbox + '/Research/VMAT/casereader/outputGraphics/' + 'histogram-' + caseis + '.png')
#plt.close()
import pandas as pd
llim = int(beamletList[0].XStart/10)
if debugmode:
    for angleposition in testcase:
        d = {'left': [int(i + 1) for i in beamList[angleposition].xilist], 'right': [int(i) for i in beamList[angleposition].psilist]}
        df = pd.DataFrame(data=d)
        df.to_csv('outputGraphics/Aperture' + str(2 * angleposition) + caseis + '-proportionalThreshold-' + str(strengthThreshold) +'.csv')
sys.exit()
with open('outputGraphics/DansCoordinatesAperture40' + caseis + '-proportionalThreshold-' + str(strengthThreshold) +'.csv','w') as resultFile:
    wr = csv.writer(resultFile)
    wr.writerows([int(i + 1 + llim) for i in beamList[20].xilist] + ["|"] + [int(i + llim) for i in beamList[20].psilist])
with open('outputGraphics/Aperture40' + caseis + '-proportionalThreshold-' + str(strengthThreshold) +'.csv','w') as resultFile:
    wr = csv.writer(resultFile)
    wr.writerows([str(int(i + 1 + llim)) for i in beamList[20].xilist] + ["|"] + [str(int(i + llim)) for i in beamList[20].psilist])
sys.exit()
with open('outputGraphics/Aperture0' + caseis + '.json', 'w') as outfile:
    json.dump([beamList[0].xilist, beamList[0].psilist], outfile)
sys.exit()

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
