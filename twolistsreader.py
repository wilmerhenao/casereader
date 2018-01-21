import pickle
from collections import defaultdict

myfolder = '/home/wilmer/Dropbox/Data/spine360/by-Beam/'
PIK = myfolder + "twolists0.pickle"
with open(PIK, "rb") as f:
    a = pickle.load(f)
