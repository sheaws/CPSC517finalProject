"""
Looks at the preconditioner returned by optimal_preconditioner
"""

import dsdl
import numpy as np
import scipy as sp
import sys

sys.path.append('C:\\classes\\UBC\\CPSC517\\project\\projectCode\\preconditioner-search-main\\code\\src')
from precsearch import optimal_preconditioner

if __name__ == "__main__":
    #M = np.array([[1,0],[0,10]])
    ds = dsdl.load("cpusmall")
    X, y = ds.get_train()
    if sp.sparse.issparse(X):
        X = X.toarray() # need this for eigvalsh

    M = X.T @ X

    m,n = M.shape
    optP = optimal_preconditioner(M,verbose=False)
    optP = np.diag(optP)
    print(f"Optimal diagonal preconditioner for {m} x {n} matrix")
    #print(M)
    #print(f"is")
    #print(optP)
    print(M.shape)
    eigvalsOld = np.linalg.eigvalsh(M)
    eigvalsNew = np.linalg.eigvalsh(optP @ M @ optP)
    maxLambdaOld = np.max(eigvalsOld)
    minLambdaOld = np.min(eigvalsOld)
    kappaOld = maxLambdaOld/minLambdaOld
    maxLambdaNew = np.max(eigvalsNew)
    minLambdaNew = np.min(eigvalsNew)
    kappaNew = maxLambdaNew/minLambdaNew
    print(f"original M: lambda max={maxLambdaOld:.3f}, lambda min={minLambdaOld:.3f}, kappa={kappaOld:.3f}")
    print(f"precond  M: lambda max={maxLambdaNew:.3f}, lambda min={minLambdaNew:.3f}, kappa={kappaNew:.3f}")
