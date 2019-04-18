#!/opt/intel/intelpython3/bin/python3.6

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

exec(open('VMATClasses.py').read())

caseis = "brain360"

datalocation = "D:/Dropbox/Data/" + caseis + "/by-Beam/"
cutter = 44
if "lung360" == caseis:
    cutter = 43

datafiles = get_files_by_file_size(datalocation)
dpdata = dose_to_points_data_pb2.DoseToPointsData()
print('datafiles:', datafiles)
f = open(datalocation + "identification.protostream", "rb")
dpdata.ParseFromString(f.read())
f.close()

numbeams = len(dpdata.Beams)
beamList = [None] * numbeams

for b in range(numbeams):
    a = dpdata.Beams[b]
    newpos = (int(int(a.Id)/2) + 90) % 180
    dpdata.Beams[b].Id = str(2 * newpos)

try:
    f = open(datalocation + "identificationTwister.protostream", "wb")
    f.write(dpdata.SerializeToString())
    f.close()
    print('saved the data')
except IOError:
    print("Problems while writing")

olddpdata = dose_to_points_data_pb2.DoseToPointsData()
newdpdata = dose_to_points_data_pb2.DoseToPointsData()

f = open(datalocation + "identification.protostream", "rb")
olddpdata.ParseFromString(f.read())
f.close()

f = open(datalocation + "identificationTwister.protostream", "rb")
newdpdata.ParseFromString(f.read())
f.close()

numbeamsold = len(olddpdata.Beams)

for i in range(numbeamsold):
    print(i, olddpdata.Beams[i].Id, newdpdata.Beams[i].Id)
