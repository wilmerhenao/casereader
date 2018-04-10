import pickle
import sys
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from scipy import stats
from scipy.stats.stats import pearsonr

class beam(object):
    numBeams = 0
    M = 8
    N = 14
    # Initialize left and right leaf limits for all
    leftEdge = -1
    rightEdge = 14
    leftEdgeFract = None
    rightEdgeFract = None
    JawX1 = None
    JawX2 = None
    JawY1 = None
    JawY2 = None
    def __init__(self, sthing):
        # Update the static counter and other variables
        self.location = int(int(sthing.Id)/2)
        self.Id = self.location
        self.angle = 2 * self.Id   # Notice that this one changes per beam
        self.JawX1 = sthing.JawX1
        self.JawX2 = sthing.JawX2
        self.JawY1 = sthing.JawY1
        self.JawY2 = sthing.JawY2
        # Update local variables
        self.StartBeamletIndex = sthing.StartBeamletIndex
        self.EndBeamletIndex = sthing.EndBeamletIndex
        self.beamletsPerBeam = self.EndBeamletIndex - self.StartBeamletIndex
        self.beamletsInBeam = self.beamletsPerBeam
        self.llist = [self.leftEdge] * self.M # One position more or less after the edges given by XStart, YStart in beamlets
        self.rlist = [self.rightEdge] * self.M
        self.KellyMeasure = 0
        self.Perimeter = 0
        self.Area = 0
        beam.numBeams += 1

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
            # NEVER CHANGE THIS! YOU MIGHT HAVE TROUBLE LATER
            self.threshold = 0.0
            self.overdoseCoeff = 0.0
            self.underdoseCoeff = 0.0
        else:
            structure.numOARs += 1
            self.threshold = 0.0
            self.overdoseCoeff = 0.0
            self.underdoseCoeff = 0.0
        structure.numStructures += 1

def plotAperture(index, llist, rlist, befaft):
    magnifier = 50
    lmag = llist
    rmag = rlist
    nrows = 8
    ncols = 14
    ## Convert the limits to hundreds.
    for posn in range(0, len(lmag)):
        lmag[posn] = int(magnifier * lmag[posn])
        rmag[posn] = int(magnifier * rmag[posn])
    image = -1 * np.ones(magnifier * nrows * ncols)
        # Reshape things into a 9x9 grid
    image = image.reshape((nrows, magnifier * ncols))
    for i in range(0, beam.M):
        image[i, lmag[i]:(rmag[i]-1)] = 1 #intensity assignment
    image = np.repeat(image, 1*magnifier, axis = 0) # Repeat. Otherwise the figure will look flat like a pancake
    #image[0,0] = data.YU # In order to get the right list of colors
    # Set up a location where to save the figure
    fig = plt.figure(1)
    cmapper = plt.get_cmap("autumn_r")
    cmapper.set_under('black', 1.0)
    plt.imshow(image, cmap = cmapper, vmin = 0.0, vmax = 1)
    plt.axis('off')
    #plt.show()
    fig.savefig("/mnt/datadrive/Dropbox" + '/Research/VMAT/casereader/outputGraphics/apertureEvolution'+ befaft + str(index) + '.png')

def apertureEvolution(index):
    plotAperture(index, apertures[index][0], apertures[index][1], 'before')
    plotAperture(index, aps[index].llist, aps[index].rlist, 'after')

CValue = 0.0
PIK = "outputGraphics/pickle-C-" + str(CValue) + "-save.dat"
with open(PIK, "rb") as f:
    datasave = pickle.load(f)
f.close()
intensities = datasave[14]

PIK = "outputGraphics/allbeamshapes-save-" + str(CValue) + ".pickle"

with open(PIK, "rb") as f:
    apertures = pickle.load(f)
f.close()
#apertures = apertures[:180]
kellys = []
perimeter = []
area = []
for ap in apertures:
    kellys.append(ap[2])
    perimeter.append(ap[3])
    area.append(ap[4])
X = np.array([perimeter, area]).transpose()
Y = np.array(kellys)
reg = linear_model.LinearRegression(fit_intercept = False)
reg.fit(X, Y)
print(reg.coef_)
# Create our measure
ourmeasure = []
for i in range(len(perimeter)):
    ourmeasure.append(reg.coef_[0] * perimeter[i] + reg.coef_[1] * area[i])
ourmeasure = np.array(ourmeasure)
plt.scatter(Y, ourmeasure)
fit = np.polyfit(Y, ourmeasure, deg=1)
slope, intercept, r_value, p_value, std_err = stats.linregress(Y,ourmeasure)
print(slope, intercept, r_value)
plt.plot(Y, fit[0] * Y + fit[1], color='red')
plt.xlabel("Kelly's Measure")
plt.ylabel("Our Measure")
plt.title("Aperture Penalization Comparison " + str(len(kellys)) + " Apertures")
plt.savefig('outputGraphics/comparison' + str(CValue) + '.png')
plt.close()
# Calculate and print Kelly's edge metric.
averageNW = 0.0
averageW = 0.0
for i in range(180):
    averageNW += Y[i]
    averageW += Y[i] * intensities[i]
print('averageW before:', averageW/180)
print('averageNW before:', averageNW/180)

#sys.exit()

PIK = "outputGraphics/beamList-save-" + str(CValue) + ".pickle"
with open(PIK, "rb") as f:
    aps = pickle.load(f)
f.close()

kellys = []
perimeter = []
area = []
for ap in aps:
    kellys.append(ap.KellyMeasure)
    perimeter.append(ap.Perimeter)
    area.append(ap.Area)
X = np.array([perimeter, area]).transpose()
Y = np.array(kellys)
reg = linear_model.LinearRegression(fit_intercept = False)
reg.fit(X, Y)
print(reg.coef_)
ourmeasure = []
for i in range(len(perimeter)):
    ourmeasure.append(reg.coef_[0] * perimeter[i] + reg.coef_[1] * area[i])
ourmeasure = np.array(ourmeasure)
plt.scatter(Y, ourmeasure)
fit = np.polyfit(Y, ourmeasure, deg=1)
slope, intercept, r_value, p_value, std_err = stats.linregress(Y,ourmeasure)
print(slope, intercept, r_value)
plt.plot(Y, fit[0] * Y + fit[1], color='red')
plt.xlabel("Kelly's Measure")
plt.ylabel("Our Measure")
plt.title("Aperture Penalization Comparison")
plt.savefig('outputGraphics/comparisonFinalBeams' + str(CValue) + '.png')
plt.close()
print('correlation', pearsonr(Y, ourmeasure))
print('Ended lecture of beams')
apertureEvolution(158)

averageNW = 0.0
averageW = 0.0
for i in range(180):
    averageNW += Y[i]
    averageW += Y[i] * intensities[i]
print('averageW after:', averageW/180)
print('averageNW after:', averageNW/180)
print('Final Objective Value of the function is:', datasave[17])

sys.exit()