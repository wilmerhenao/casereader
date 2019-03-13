
## This class contains data about the running times of the program
# The class contains member functions that keep track of the reading times
# and the refinement loops
class timedata(object):
    def __init__(self):
        self.initialtime = time.time()
        self.lasttime = time.time()
        self.looptimes = list()
        self.readtime = np.inf

    def newloop(self):
        self.looptimes.append(time.time() - self.lasttime)
        self.lasttime = time.time()

    def readingtime(self):
        self.readtime  = time.time() - self.lasttime
        self.lasttime = time.time()

## This class contains a structure (region)
# It could be a PTV or an organ. It also keeps track of the associated goals such
# as the function coefficients as they pertain to the objective function
class structure(object):
    ## Static variable that keeps a tally of the total number of structures
    numStructures = 0
    numTargets = 0
    numOARs = 0
    listTargets = []
    def __init__(self, sthing, s):
        self.Id = sthing.Id
        self.pointsDistanceCM = sthing.pointsDistanceCM
        self.StartPointIndex = sthing.StartPointIndex
        self.EndPointIndex = sthing.EndPointIndex
        self.Size = self.EndPointIndex - self.StartPointIndex
        self.isTarget = False
        self.isKilled = False
        # Identify targets
        alb = "PTV" in self.Id;
        ala = "_GTV" in self.Id;
        alc = "CTV" in self.Id;
        if ( alb | ala  | alc):
            self.isTarget = True
        if self.isTarget:
            structure.numTargets = structure.numTargets + 1
            structure.listTargets.append(s)
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

## Contains the beam data
# It keeps the location, the beamlet members, different measures of the aperture
class beam(object):
    numBeams = 0
    M = None
    N = None
    JawX1 = None
    JawX2 = None
    JawY1 = None
    JawY2 = None
    def __init__(self, sthing):
        # Initialize left and right leaf limits for all
        self.leftEdge = self.N // 2 - 1
        self.rightEdge = self.leftEdge + 1/numsubdivisions
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
        self.xilist = [self.leftEdge] * self.M
        self.psilist = [self.rightEdge] * self.M
        # The two hard limits below exist only to test whether the hard limits changed or not
        self.initialxilist = [self.leftEdge] * self.M
        self.initialpsilist = [self.rightEdge] * self.M
        self.KellyMeasure = 0
        self.Perimeter = 0
        self.Area = 0
        self.leftEdgeFract = None
        self.rightEdgeFract = None
        beam.numBeams += 1

class voxel(object):
    numVoxels = 0
    def __init__(self, sthing):
        voxel.numVoxels += 1
        self.Index = sthing.Index
        self.StructureId = sthing.StructureId
        self.X = sthing.X
        self.Y = sthing.Y
        self.Z = sthing.Z

class beamlet(object):
    numBeamlets = 0
    XSize = None
    YSize = None
    def __init__(self, sthing):
        beamlet.numBeamlets += 1
        self.Index = sthing.Index
        self.XStart = sthing.XStart
        self.YStart = sthing.YStart
        beamlet.XSize = sthing.XSize
        beamlet.YSize = sthing.YSize
        self.belongsToBeam = None

## apertureList is a class definition of locs and angles that is always sorted.
# Its attributes are loc which is the numeric location; It has range 0 to 36 for
# the brain case; Angle is the numeric angle in degrees; It ranges from 0 to 360 degrees
# apertureList should be sorted in ascending order everytime you add a new element; User CAN make this safe assumption
class apertureList:
    ## constructor initializes empty lists
    def __init__(self):
        ## Location in index range(0,numbeams)
        self.loc = []
        ## Angles ranges from 0 to 360
        self.angle = []

    ## Insert a new angle in the list of angles to analyse.
    # Gets angle information and inserts location and angle
    # In the end it sorts the list in increasing order
    def insertAngle(self, i, aperangle):
        self.angle.append(aperangle)
        self.loc.append(i)
        # Sort the angle list in ascending order
        self.loc.sort()
        self.angle.sort()

    ## Removes the index and its corresponding angle from the list.
    # Notice that it only removes the first occurrence; but if you have done everything correctly this should never
    # be a problem
    def removeIndex(self, index):
        toremove = [i for i, x in enumerate(self.loc) if x == index]
        self.loc.pop(toremove[0])  # Notice that it removes the first entry
        self.angle.pop(toremove[0])

    ## Looks for the angle and removes the index and the angle corresponding to it from the list
    def removeAngle(self, tangl):
        toremove = [i for i, x in enumerate(self.angle) if x == tangl]
        self.loc.pop(toremove[0])
        self.angle.pop(toremove[0])

    ## Overloads parenthesis operator in order to fetch the ANGLE given an index.
    # Returns the angle at the ith location given by the index.
    # First Find the location of that index in the series of loc
    # Notice that this function overloads the parenthesis operator for elements of this class.
    def __call__(self, index):
        toreturn = [i for i, x in enumerate(self.loc) if x == index]
        return (self.angle[toreturn[0]])

    ## Returns the length of this instantiation without the need to pass parameters.
    def len(self):
        return (len(self.loc))

    ## Returns True if the list is empty; otherwise returns False.
    def isEmpty(self):
        if 0 == len(self.loc):
            return (True)
        else:
            return (False)


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
    filepaths.sort(key=lambda filename: filename[1], reverse=reverse)
    # Re-populate list with just filenames
    for i in range(len(filepaths)):
        filepaths[i] = filepaths[i][0]
    return (filepaths)


class problemData():
    def __init__(self):
        self.kappa = []
        self.notinC = apertureList()
        self.caligraphicC = apertureList()
        self.currentDose = np.zeros(voxel.numVoxels, dtype=float)
        self.quadHelperThresh = np.empty(voxel.numVoxels, dtype=float)
        self.quadHelperOver = np.empty(voxel.numVoxels, dtype=float)
        self.quadHelperUnder = np.empty(voxel.numVoxels, dtype=float)
        self.maskValue = np.empty(voxel.numVoxels, dtype=float)
        self.setQuadHelpers(structureList, voxelList)
        self.openApertureMaps = [[] for i in range(beam.numBeams)]
        self.diagmakers = [[] for i in range(beam.numBeams)]
        self.strengths = [[] for i in range(beam.numBeams)]
        self.DlistT = None
        self.currentIntensities = np.zeros(beam.numBeams, dtype=float)
        self.voxelsUsed = None
        self.structuresUsed = None
        self.structureIndexUsed = None
        self.YU = None
        self.RU = None
        self.speedlim = None
        self.rmpres = None
        self.listIndexofAperturesRemovedEachStep = []

    def setQuadHelpers(self, sList, vList):
        for i in range(voxel.numVoxels):
            sid = structureDict[vList[i].StructureId]  # Find structure of this particular voxel
            self.quadHelperThresh[i] = sList[sid].threshold
            self.quadHelperOver[i] = sList[sid].overdoseCoeff
            self.quadHelperUnder[i] = sList[sid].underdoseCoeff
            self.maskValue[i] = 2 ** sid

    # data class function
    def calcDose(self, newIntensities):
        self.currentIntensities = newIntensities
        self.currentDose = DmatBig.transpose() * newIntensities

def calcObjGrad(x):
    data.calcDose(x)
    oDoseObj = data.currentDose - data.quadHelperThresh
    oDoseObjCl = (oDoseObj > 0) * oDoseObj
    oDoseObjGl = oDoseObjCl * oDoseObjCl * data.quadHelperOver
    uDoseObj = data.quadHelperThresh - data.currentDose
    uDoseObjCl = (uDoseObj > 0) * uDoseObj
    uDoseObjGl = uDoseObjCl * uDoseObjCl * data.quadHelperUnder
    objectiveValue = sum(oDoseObjGl + uDoseObjGl)
    oDoseObjGl = oDoseObjCl * data.quadHelperOver
    uDoseObjGl = uDoseObjCl * data.quadHelperUnder
    mygradient = DmatBig * 2 * (oDoseObjGl - uDoseObjGl)
    return (objectiveValue, mygradient)


# The next function prints DVH values
def printresults(myfolder):
    data.maskValue = np.array([int(i) for i in data.maskValue])
    print('Starting to Print Result DVHs brain case')
    zvalues = data.currentDose
    maxDose = max([float(i) for i in zvalues])
    dose_resln = 0.1
    dose_ub = maxDose + 10
    bin_center = np.arange(0, dose_ub, dose_resln)
    # Generate holder matrix
    dvh_matrix = np.zeros((structure.numStructures, len(bin_center)))
    # iterate through each structure
    for s in data.structureIndexUsed:
        doseHolder = sorted(zvalues[[i for i, v in enumerate(data.maskValue & 2 ** s) if v > 0]])
        if 0 == len(doseHolder):
            continue
        histHolder, garbage = np.histogram(doseHolder, bin_center)
        histHolder = np.append(histHolder, 0)
        histHolder = np.cumsum(histHolder)
        dvhHolder = 1 - (np.matrix(histHolder) / max(histHolder))
        dvh_matrix[s,] = dvhHolder
    print('matrix shape:', dvh_matrix.shape)
    dvh_matrix = dvh_matrix[data.structureIndexUsed,]
    print(dvh_matrix.shape)

    myfig = pylab.plot(bin_center, dvh_matrix.T, linewidth=2.0)
    plt.grid(True)
    plt.xlabel('Dose Gray')
    plt.ylabel('Fractional Volume')
    plt.title('IMRT Solution')
    plt.legend(structureNames, prop={'size': 9})
    plt.savefig(myfolder + 'DVH-for-debugging-IMRT-' + caseis + '-Threshold' + str(strengthThreshold) + '.png')
    plt.close()
