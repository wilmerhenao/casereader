import pickle
import sys
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from scipy import stats

PIK = "outputGraphics/allbeamshapes-save-1e-06.pickle"

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
plt.title("Aperture Penalization Comparison")
plt.savefig('outputGraphics/comparison.png')


sys.exit()