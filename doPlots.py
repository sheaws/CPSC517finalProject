"""
Plots
"""
from typing import Any, Dict
import glob
import csv

#import dsdl
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import sys
from typing import Any, Callable, Dict, Optional
from numpy.typing import NDArray

sys.path.append('C:\\classes\\UBC\\CPSC517\\project\\projectCode\\preconditioner-search-main\\code\\src')


if __name__ == "__main__":
    fStars = {}
    fStarsFilename = "fStars.csv"
    with open(fStarsFilename, newline='') as fStarsCsvfile:
        fStarsReader = csv.DictReader(fStarsCsvfile)
        for row in fStarsReader:
            fStars[row['Dataset']] = float(row['fStar'])

    outputDir="experiment1\\linearRegression"
    for folderName in glob.iglob(outputDir+"\\*"):
        tokens = folderName.split("\\")
        dsName = tokens[-1]
        plt.close('all')
        plt.yscale('log')
        plt.title(dsName)
        for dataFilename in glob.iglob(folderName+"\\*"):
            tokens = dataFilename.split("\\")
            method = tokens[-1].split(".")[0]
            #print(f"Results using {method} found for {dsName}.")
            data = pd.read_csv(dataFilename)
            x = data['t'].tolist()
            y = (data['f']-fStars[dsName]).tolist()
            plt.plot(x,y,label=method)
        plt.xlabel("Iterations")
        plt.ylabel("f-f*")
        plt.legend()
        plt.savefig(dsName+".png")
