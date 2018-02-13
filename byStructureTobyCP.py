#! /opt/intel/intelpython35/bin/python3.5

import DoseToPoints_pb2
import sys
import gc
import os
import socket
from collections import defaultdict
import pickle

gc.enable()
## Find out the variables according to the hostname
datalocation = '~'
if 'radiation-math' == socket.gethostname(): # LAB
    datalocation = "/mnt/fastdata/Data/spine360"
    dropbox = "/mnt/datadrive/Dropbox"
elif 'sharkpool' == socket.gethostname(): # MY HOUSE
    datalocation = "/home/wilmer/Dropbox/Data/spine360"
    dropbox = "/home/wilmer/Dropbox"
else:
    datalocation = "/home/wilmer/Dropbox/Data/spine360" # MY LAPTOP
    dropbox = "/home/wilmer/Dropbox"
datafiles = [datalocation + "/by-Structure/PsVM2m_2_90_2/fc0a4f7a-04ab-4e90-90ce-a39005760280",
             datalocation + "/by-Structure/PsVM1m_92_180_2/195af10c-705a-4867-a95a-dc3d2f60b0eb",
             datalocation + "/by-Structure/PsVM4m_182_270_2/f24672e5-c46c-44fa-9211-4df2591f1b4f",
             datalocation + "/by-Structure/PsVM3m_272_0_2/8cfc980e-9e04-4afb-b938-5f9702f7f4a6"]
resultslocation = "/mnt/datadrive/Dropbox/Data/spine360/by-Beam/"
# The first file will contain all the structure data, the rest will contain pointodoses.
alldata = DoseToPoints_pb2.DoseToPointsData()
numstructs = []
flag = True
accumulator = 0
accumulatorlist = []
for thisfile in datafiles:
    print('reading file ' + thisfile)
    # Start with reading structures, numvoxels and all that.
    try:
        dpdata = DoseToPoints_pb2.DoseToPointsData()
        f = open(thisfile, "rb")
        dpdata.ParseFromString(f.read())
        f.close()
    except IOError:
        print ("Could not open file.  Creating a new one.")

    if flag:
        # This is the case for the first file
        alldata = dpdata
        flag = False
    else:
        # All files need to follow a few procedures
        for b in dpdata.Beams:
            alldata.Beams.add()
            i = len(alldata.Beams) - 1
            alldata.Beams[i].StartBeamletIndex = b.StartBeamletIndex + accumulator
            alldata.Beams[i].EndBeamletIndex = b.EndBeamletIndex + accumulator
            alldata.Beams[i].Id = b.Id
            alldata.Beams[i].JawX1 = b.JawX1
            alldata.Beams[i].JawX2 = b.JawX2
            alldata.Beams[i].JawY1 = b.JawY1
            alldata.Beams[i].JawY2 = b.JawY2
        for b in dpdata.Beamlets:
            alldata.Beamlets.add()
            i = len(alldata.Beamlets) - 1
            alldata.Beamlets[i].Index = b.Index + accumulator
            alldata.Beamlets[i].BeamId = b.BeamId
            alldata.Beamlets[i].XStart = b.XStart
            alldata.Beamlets[i].YStart = b.YStart
            alldata.Beamlets[i].XSize = b.XSize
            alldata.Beamlets[i].YSize = b.YSize
    accumulatorlist.append(len(dpdata.Beamlets))
    accumulator += len(dpdata.Beamlets) #TotalBeamlets * numbeams in each file
# find out where beamlets live
beamletdict = defaultdict(list)
for b in alldata.Beams:
    thisindex = b.Id
    beamletdict[thisindex].append(b.StartBeamletIndex)
    beamletdict[thisindex].append(b.EndBeamletIndex)
[print(b) for b in beamletdict.items()]
# Perform tests on the data.
try:
    f = open(resultslocation + "identificationSpine2.protostream", "wb")
    f.write(alldata.SerializeToString())
    f.close()
except IOError:
    print("Problems while writing")
print('termino')
print('Reading test')
try:
    tester = DoseToPoints_pb2.DoseToPointsData()
    f = open(resultslocation + "identificationSpine2.protostream", "rb")
    tester.ParseFromString(f.read())
    #for p in tester.Points:
    #    print(p)
    f.close()
except IOError:
    print("Could not open file.  Creating a new one.")
print('lengths')
### Working on dose to points
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
accumulator = 0
datafolders = [datalocation + "/by-Structure/PsVM2m_2_90_2/",
             datalocation + "/by-Structure/PsVM1m_92_180_2/",
             datalocation + "/by-Structure/PsVM4m_182_270_2/",
             datalocation + "/by-Structure/PsVM3m_272_0_2/"]
for folder in datafolders:
    files = get_files_by_file_size(folder)
    files.pop(0)
    for file in files:
        d = defaultdict(list)
        print('reading file:', file)
        f = open(file, "rb")
        dpdata = DoseToPoints_pb2.DoseToPointsData()
        dpdata.ParseFromString(f.read())
        f.close()
        for pd in dpdata.PointDoses:
            # pd.Index here represents the index of the voxel
            for bd in pd.BeamletDoses:
                thisindex = bd.Index + accumulator #This is the index of the b
                for k, alist in beamletdict.items():
                    if thisindex <= alist[1] and thisindex >= alist[0]:
                        #print(pd.Index, bd.Dose)# eamlet
                        d[k].append([thisindex, bd]) # Wilmer! Revisa esto aca.
        print('resultslocations', resultslocation)
        for namefile, elements in d.items():
            output = open(resultslocation + namefile + '.pickle', 'ab')
            pickle.dump(elements, output)
            output.close()
        d = None
    accumulator += accumulatorlist.pop(0)
sys.exit()