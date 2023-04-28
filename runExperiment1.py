"""
Example
"""
from typing import Any, Dict

import dsdl
import numpy as np
import scipy as sp
import scipy.sparse
import sys
import csv
import os

sys.path.append('C:\\classes\\UBC\\CPSC517\\project\\projectCode\\preconditioner-search-main\\code\\src')
import precsearch.optimizers
import precsearch.plotting_style
from precsearch.optimizers.base_optimizer import ProblemAtPoint
from precsearch import optimal_preconditioner

if __name__ == "__main__":
    fStars = {}
    fStarsFilename = "fStars.csv"
    with open(fStarsFilename, newline='') as fStarsCsvfile:
        fStarsReader = csv.DictReader(fStarsCsvfile)
        for row in fStarsReader:
            fStars[row['Dataset']] = float(row['fStar'])

    outputDir="experiment1\\linearRegression"

    datasetNames = dsdl.available_datasets()
    #datasetNames = ["cpusmall"]
    for dsName in datasetNames:
        if "movie" in dsName:
            continue
        ds = dsdl.load(dsName)

        if ds.task != "Regression":
            print(f"{dsName} is a {ds.task} task. Skipping.")
            continue

        X, y = ds.get_train()
        N, D = X.shape

        if N > 10000 or D > 30:
            continue

        print(f"Running {dsName}: {N} by {D}")

        directory = outputDir+"//"+dsName
        if not os.path.exists(directory):
            os.makedirs(directory)

        #outputFilename = directory+"//GDLS.csv"
        #outputFilename = directory+"//BoxPrecond.csv"
        outputFilename = directory+"//OptPrecond.csv"
        outputFH = open(outputFilename,'w')
        print("t,f",file=outputFH)

        if sp.sparse.issparse(X):
            X = X.toarray()

        def f(w):
            return 0.5 * np.mean((X @ w - y) ** 2)

        def g(w):
            return X.T @ (X @ w - y) / N

        t = 0
        T = 1000

        def callback(Sx: ProblemAtPoint, state: Dict[str, Any]):
            global t 
            #if t % int(T / 20) == 0:
                #gnorm = np.linalg.norm(Sx.g)
                #print(
                #    f"Iteration {t:>4}/{T}: Loss {Sx.f:.2e}  Grad norm {gnorm:.2e}  {state}"
                #)
            print(f"{t},{Sx.f:.20f}",file=outputFH)
            t += 1

        #opt = precsearch.optimizers.GDLS(starting_stepsize=1.0)
        #opt = precsearch.optimizers.BoxPreconditionerSearch() # requires changing the return value for solve
        M = X.T @ X
        optP = optimal_preconditioner(M,verbose=False)
        optP = np.diag(optP/N)
        opt = precsearch.optimizers.GD(P=optP,ss=1.0)

        np.random.seed(0)
        w0 = np.random.randn(X.shape[1]).reshape((-1,))

        pap = opt.solve(func=f, grad=g, x0=w0, maxiter=T, tol=10**-5, callback=callback)
        f = pap.f
        optGap = f - fStars[dsName]
        print(f"{dsName} ({N} by {D}): {f} - {fStars[dsName]} = {optGap}")
        outputFH.close()