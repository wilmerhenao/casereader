import pickle
import sys
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from scipy import stats

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

CValue = 1e-09

PIK = "outputGraphics/allbeamshapes-save-" + str(CValue) + ".pickle"

with open(PIK, "rb") as f:
    apertures = pickle.load(f)
f.close()
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

sys.exit()

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

print('Ended lecture of beams')

sys.exit()