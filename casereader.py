#! /opt/intel/intelpython35/bin/python3.5

import dose_to_points_data_pb2
import sys
import os

datalocation = "/mnt/fastdata/Data/spine"

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
        alb = "PTV" in self.Id;
        ala = "GTV" in self.Id;
        alc = "CTV" in self.Id;
        if ( alb | ala  | alc):
            self.isTarget = True
        if self.isTarget:
            structure.numTargets = structure.numTargets + 1
        else:
            structure.numOARs += 1
        structure.numStructures += 1

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
f = open(datafiles[0], "rb")
dpdata.ParseFromString(f.read())
f.close()

# Get the data about the structures
numstructs = len(dpdata.Structures)
structureList = []
print("Reading in structures")
for s in range(numstructs):
    print('Structure Data:', dpdata.Structures[s].Id)
    structureList.append(structure(dpdata.Structures[s]))
print('Number of structures:', structure.numStructures, 'Number of Targets:', structure.numTargets,
      'Number of OARs', structure.numOARs)

# Get the data about beams
numbeams = len(dpdata.Beams)
beamList = []
print('Reading in Beam Data:')
for b in range(numbeams):

sys.exit()

numvoxels = len(dpdata.Points)
beamnumperbeam = [None] * numbeams
beamletsperbeam = [None] * numbeams
dijsPerBeam = [None] * numbeams

numX = 0
for b in range(numbeams):
    beamletsperbeam[b] = dpdata.Beams[b].EndBeamletIndex - dpdata.Beams[b].StartBeamletIndex
    numX = numX + beamletsperbeam[b]

# The variables below were not added in the original document by Carlos.

numpointdoses = len(dpdata.PointDoses)
numbeamlets = len(dpdata.Beamlets)

print('control points:', len(dpdata.Beams))
for cp in range(len(dpdata.Beams)):
    print(dpdata.Beams[cp].JawX1, dpdata.Beams[cp].JawX2, dpdata.Beams[cp].JawY1, dpdata.Beams[cp].JawY2, dpdata.Beams[cp].StartBeamletIndex, dpdata.Beams[cp].EndBeamletIndex)
print('Beamlets for the third beam:')
for bmlet in range(1120, 1680):
    print(dpdata.Beamlets[bmlet].Index, dpdata.Beamlets[bmlet].BeamId, dpdata.Beamlets[bmlet].XSize, dpdata.Beamlets[bmlet].YSize, dpdata.Beamlets[bmlet].XStart, dpdata.Beamlets[bmlet].YStart)
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