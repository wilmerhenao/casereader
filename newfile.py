import numpy as np
import pandas as pd
np.random.seed(2019)
pd.DataFrame(np.random.randn(1000,100)).to_csv("data.csv")
pdata = pd.read_csv("data.csv", index_col=0)

## Function that will calculate the z-scores as defined on the document
def zscoreF(X, Y):
    r = X - Y
    cindex = np.cumprod(1+r)
    latest = cindex[999]
    meanval = np.mean(cindex)
    stdval = np.std(cindex)
    zscore = (latest - meanval) / stdval
    return(zscore)

# Preallocate empty lists to save time when I run the program
myscores = [0] * int(100*100)
mypairs = [None] * int(100*100)
counter = 0
for i in range(100):
    print('Working on ', i, 'pairs')
    X = pdata[str(i)]
    for j in range(i+1, 100):
        Y = pdata[str(j)]
        val = zscoreF(X, Y)
        myscores[counter] = val
        mypairs[counter] = (i, j)
        myscores[counter+1] = -val
        mypairs[counter+1] = (j, i)
        counter+=2

# Order both according to myscores
finalList = [x for _,x in sorted(zip(myscores, mypairs), reverse = True)][:3]
orderedZscores = sorted(myscores, reverse = True)
print('The list of pairs that should be used in this trade is:')
for counter, mypair in enumerate(finalList):
    print('Long:', mypair[0], 'and Short:', mypair[1], 'with z-score:', orderedZscores[counter])

print('Thank you!')

