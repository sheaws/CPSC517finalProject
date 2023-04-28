"""
Example
"""
from typing import Any, Dict

import dsdl
import numpy as np
import scipy as sp
import scipy.sparse
import sys
from typing import Any, Callable, Dict, Optional
from numpy.typing import NDArray

sys.path.append('C:\\classes\\UBC\\CPSC517\\project\\projectCode\\preconditioner-search-main\\code\\src')
#import precsearch.optimizers
import precsearch.plotting_style
from precsearch.optimizers.base_optimizer import ProblemAtPoint
from precsearch.optimizers.base_optimizer import Optimizer
from precsearch.optimizers.base_optimizer import Problem
from precsearch.optimizers.lfgbs_scipy import fmin_l_bfgs_b

if __name__ == "__main__":
    fStarsFilename = "fStars.csv"
    fStarsFH = open(fStarsFilename,'w')
    print(f"Dataset,N,D,fStar",file=fStarsFH)
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

        if sp.sparse.issparse(X):
            X = X.toarray()

        def f(w):
            return 0.5 * np.mean((X @ w - y) ** 2)

        def g(w):
            return X.T @ (X @ w - y) / N
        
        class LBFGS(Optimizer):
            def __init__(self, L=10, iprint=0):
                self.L = L
                self.iprint = iprint
                super().__init__()

            def solve(
                self,
                func: Callable[[NDArray], float],
                grad: Callable[[NDArray], NDArray],
                x0: NDArray,
                tol: float = 10**-9,
                maxiter: int = 1000,
                callback: Optional[
                    Callable[[ProblemAtPoint, Dict[str, Any]], Optional[bool]]
                ] = None,
            ):
                problem = Problem(func, grad)
                x = np.copy(x0)

                def lbfgs_callback(x, state):
                    return callback(ProblemAtPoint(problem, x), state)

                res = fmin_l_bfgs_b(
                    func=func,
                    x0=x,
                    fprime=grad,
                    factr=tol / np.finfo(float).eps,
                    pgtol=tol,
                    epsilon=tol,
                    maxiter=maxiter,
                    maxfun=20 * maxiter,
                    m=self.L,
                    #callback=lbfgs_callback,
                    iprint=self.iprint,
                )

                return ProblemAtPoint(problem, res[0])

        t = 0
        T = 1000

        def callback(Sx: ProblemAtPoint, state: Dict[str, Any]):
            global t 
            t += 1

        opt = LBFGS()

        np.random.seed(0)
        w0 = np.random.randn(X.shape[1]).reshape((-1,))

        pap = opt.solve(func=f, grad=g, x0=w0, maxiter=10000, callback=callback)
        fStar = pap.f
        print(f"{dsName} ({N} by {D}): fStar={fStar}")
        print(f"{dsName},{N},{D},{fStar}",file=fStarsFH)
    fStarsFH.close()