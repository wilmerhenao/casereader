#! /opt/intel/intelpython35/bin/python3.5

import dose_to_points_data_pb2
import sys
import gc
import socket

gc.enable()
## Find out the variables according to the hostname
datalocation = '~'
if 'radiation-math' == socket.gethostname(): # LAB
    datalocation = "/mnt/fastdata/Data/spine360"
    dropbox = "/mnt/datadrive/Dropbox"
elif 'sharkpool' == socket.gethostname(): # MY HOUSE
    datalocation = "/home/wilmer/Dropbox/Data/spine360"
    dropbox = "/home/wilmer/Dropbox"
#elif ('arc-ts.umich.edu' == socket.gethostname().split('.', 1)[1]): # FLUX
#    datalocation = "/scratch/engin_flux/wilmer/spine360"
#    dropbox = "/home/wilmer/Dropbox"
else:
    datalocation = "/home/wilmer/Dropbox/Data/spine360" # MY LAPTOP
    dropbox = "/home/wilmer/Dropbox"
datafiles = [datalocation + "/by-Structure/PsVM2_90_2_2/850747bb-6c8e-4c1c-897e-8788bf4c2858",
             datalocation + "/by-Structure/PsVM1_180_92_2/67d05ef0-1447-4ae3-9b46-926fe2f1a8cd",
             datalocation + "/by-Structure/PsVM4_270_182_2/472ae384-302f-41b5-b68f-d7c6990334a1",
             datalocation + "/by-Structure/PsVM3_0_272_2/d592474d-dc94-4f8a-b8e5-a6a35f895393"]
resultslocation = "/home/wilmer/Dropbox/Data/spine360/by-Beam/"
# The first file will contain all the structure data, the rest will contain pointodoses.
alldata = dose_to_points_data_pb2.DoseToPointsData()
numstructs = []
flag = True
accumulator = 0
for thisfile in datafiles:
    print('reading file ' + thisfile)
    # Start with reading structures, numvoxels and all that.
    try:
        dpdata = dose_to_points_data_pb2.DoseToPointsData()
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
    accumulator += len(dpdata.Beamlets) #TotalBeamlets * numbeams in each file



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
    tester = dose_to_points_data_pb2.DoseToPointsData()
    f = open(resultslocation + "identificationSpine2.protostream", "rb")
    tester.ParseFromString(f.read())
    f.close()
except IOError:
    print("Could not open file.  Creating a new one.")
print('lengths')
print(len(tester.Structures))
print(len(tester.Beamlets))
print(len(tester.Beams))
print(len(tester.Points))
print(tester.Beams[179])
print(tester.Beamlets[5040*4-1])

sys.exit()
