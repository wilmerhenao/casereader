#! /opt/intel/intelpython35/bin/python3.5

import dose_to_points_data_pb2
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
datafiles = [datalocation + "/by-Structure/PsVM2_90_2_2/850747bb-6c8e-4c1c-897e-8788bf4c2858",
             datalocation + "/by-Structure/PsVM1_180_92_2/67d05ef0-1447-4ae3-9b46-926fe2f1a8cd",
             datalocation + "/by-Structure/PsVM4_270_182_2/472ae384-302f-41b5-b68f-d7c6990334a1",
             datalocation + "/by-Structure/PsVM3_0_272_2/d592474d-dc94-4f8a-b8e5-a6a35f895393"]
resultslocation = "/mnt/datadrive/Dropbox/Data/spine360/by-Beam/"
# The first file will contain all the structure data, the rest will contain pointodoses.
alldata = dose_to_points_data_pb2.DoseToPointsData()
numstructs = []
flag = True
accumulator = 0
accumulatorlist = []
alldata = []
for thisfile in datafiles:
    print('reading file ' + thisfile)
    # Start with reading structures, numvoxels and all that.
    try:
        dpdata = dose_to_points_data_pb2.DoseToPointsData()
        f = open( thisfile, "rb" )
        dpdata.ParseFromString(f.read() )
        f.close()
        j=20000
        print(dpdata.Points[j].Index, dpdata.Points[j].X, dpdata.Points[j].Y, dpdata.Points[j].Z, dpdata.Points[j].StructureId, )
        alldata.append(dpdata.Points)
    except IOError:
        print ("Could not open file.  Creating a new one.")

sys.exit()


for i in range(len(alldata[0])):
    if alldata[0][i] != alldata[1][i]:
        print("Difference", alldata[0][i].X, alldata[1][i].X)
    if alldata[0][i] != alldata[2][i]:
        print("Difference", alldata[0][i].X, alldata[2][i].X)
    if alldata[0][i] != alldata[3][i]:
        print("Difference", alldata[0][i].X, alldata[3][i].X)
print(len(alldata))