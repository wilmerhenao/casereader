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
    datalocation = "/mnt/datadrive/Dropbox/Data/lung360"
    datalocation = "/mnt/datadrive/Dropbox/Data/brain360"
    dropbox = "/mnt/datadrive/Dropbox"
elif 'sharkpool' == socket.gethostname(): # MY HOUSE
    datalocation = "/home/wilmer/Dropbox/Data/spine360"
    dropbox = "/home/wilmer/Dropbox"
else:
    datalocation = "/home/wilmer/Dropbox/Data/spine360" # MY LAPTOP
    dropbox = "/home/wilmer/Dropbox"
#datafiles = [datalocation + "/by-Structure/PsVM2m_2_90_2/fc0a4f7a-04ab-4e90-90ce-a39005760280",
#             datalocation + "/by-Structure/PsVM1m_92_180_2/195af10c-705a-4867-a95a-dc3d2f60b0eb",
#             datalocation + "/by-Structure/PsVM4m_182_270_2/f24672e5-c46c-44fa-9211-4df2591f1b4f",
#             datalocation + "/by-Structure/PsVM3m_272_0_2/8cfc980e-9e04-4afb-b938-5f9702f7f4a6"]

datafiles = [datalocation + "/by-Structure/pVMAT2_Lung_2_90_2/36a98014-3901-4b79-86b2-67eb9724286a",
             datalocation + "/by-Structure/pVMAT1_Lung_92_180_2/83970c99-6b24-4846-8b6c-ca2636cfd8e6",
             datalocation + "/by-Structure/pVMAT4_Lung_182_270_2/16dd961c-6463-498b-a020-a34128695540",
             datalocation + "/by-Structure/pVMAT3_Lung_272_0_2/01602856-4fc6-4de4-ad26-a828168c0db8"]

datafiles = [datalocation + "/by-Structure/pVMATDTP1/4b88f023-379f-45ae-a82e-67b26fc5203f",
             datalocation + "/by-Structure/pVMATDTP2/095374a4-e859-4c6c-b2ae-19ac7487f9cf",
             datalocation + "/by-Structure/pVMATDTP3/1ad88e84-f4d8-4eba-90aa-5d306c2704ec",
             datalocation + "/by-Structure/pVMATDTP4/8ab1a759-8881-4c75-ae14-ef973fc313af"]

    
resultslocation = "/mnt/datadrive/Dropbox/Data/spine360/numbernames/"

resultslocation = "/mnt/datadrive/Dropbox/Data/lung360/numbernames/"

resultslocation = "/mnt/datadrive/Dropbox/Data/brain360/numbernames/"

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
    f = open(resultslocation + "identification.protostream", "wb")
    f.write(alldata.SerializeToString())
    f.close()
except IOError:
    print("Problems while writing")
print('finished')
print('Reading test')
try:
    tester = DoseToPoints_pb2.DoseToPointsData()
    f = open(resultslocation + "identification.protostream", "rb")
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

# This is for the spine case
datafolders = [datalocation + "/by-Structure/PsVM2m_2_90_2/",
             datalocation + "/by-Structure/PsVM1m_92_180_2/",
             datalocation + "/by-Structure/PsVM4m_182_270_2/",
             datalocation + "/by-Structure/PsVM3m_272_0_2/"]
# This is for the lung Case
datafolders = [datalocation + "/by-Structure/pVMAT2_Lung_2_90_2/",
               datalocation + "/by-Structure/pVMAT1_Lung_92_180_2/",
               datalocation + "/by-Structure/pVMAT4_Lung_182_270_2/",
               datalocation + "/by-Structure/pVMAT3_Lung_272_0_2/"]
# This is for the brain Case
datafolders = [datalocation + "/by-Structure/pVMATDTP1/",
               datalocation + "/by-Structure/pVMATDTP2/",
               datalocation + "/by-Structure/pVMATDTP3/",
               datalocation + "/by-Structure/pVMATDTP4/"]

for folder in datafolders:
    files = get_files_by_file_size(folder)
    files.pop(0)
    for file in files:
        d = defaultdict(list) # d is organized by beams [2, ..., 360]
        print('reading file:', file)
        f = open(file, "rb")
        dpdata = DoseToPoints_pb2.DoseToPointsData()
        dpdata.ParseFromString(f.read())
        f.close()
        for pd in dpdata.PointDoses:
            # pd.Index here represents the index of the voxel
            for bd in pd.BeamletDoses:
                bd.Index += accumulator #This is the index of the b
                for k, alist in beamletdict.items():
                    if bd.Index <= alist[1] and bd.Index >= alist[0]:
                        #print(file, pd.Index, bd.Dose)# eamlet
                        d[k].append([pd.Index, bd]) # Here I append a list of [voxel Index, dose] (a triplet for sparse matrices input)
        print('resultslocations', resultslocation)
        for namefile, elements in d.items():
            output = open(resultslocation + namefile + '.pickle', 'ab')
            pickle.dump(elements, output) # Here I dump [beamlet Index, Doses to all voxels]
            output.close()
        d = None
    accumulator += accumulatorlist.pop(0)

sys.exit()
