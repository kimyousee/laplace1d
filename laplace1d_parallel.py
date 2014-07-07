import sys, slepc4py
slepc4py.init(sys.argv)

from petsc4py import PETSc
from slepc4py import SLEPc
import numpy as np
import matplotlib.pyplot as plt

Print = PETSc.Sys.Print
size = PETSc.COMM_WORLD.Get_size()
rank = PETSc.COMM_WORLD.Get_rank()

## Use a vector instead of a matrix to make 1d problem
## testing with parallel


def laplace1d(x, f, da):
    m,n = da.getSizes() 
    hx = 1.0/(m) # x grid spacing
    (xs,xe),(ys,ye) = da.getRanges()

    for i in range(xs,xe):
        u = x[i]
        u_e = u_w = 0
        if i > 0: u_w = x[i-1]
        if i < m-1: u_e = x[i+1]
        u_xx = (-u_e + 2*u - u_w)/hx
        f[i] = u_xx

class Laplacian1D(object):

    def __init__(self, da):
        #self.m = m
        scalar = PETSc.ScalarType
        self.da = da
        # self.U = np.zeros([m+2, 1], dtype=scalar)
        # self.U = PETSc.Vec().createMPI(m+2)
        #self.U = self.da.createGlobalVec()
        self.localX = self.da.createLocalVec()

    def mult(self, A, X, Y):
        #m, n = self.da.getSizes() # m,1
        self.da.globalToLocal(X,self.localX) # make x have local parts (?)
        x = self.da.getVecArray(self.localX)
        y = self.da.getVecArray(Y)
        #xx = x#[...].reshape(m,1)
        #yy = y#[...].reshape(m,1)
        laplace1d(x, y, self.da)
    def getDiagonal(self,A,diag):
        diag.set(4.0)

def construct_operator(m,da):
    """
    Standard symmetric eigenproblem corresponding to the
    Laplacian operator in 1 dimension. Uses *shell* matrix.
    """
    # Create shell matrix
    context = Laplacian1D(da)

    A = PETSc.Mat().createPython([m,m], context) # similar to petsc's MatCreateShell
    A.setUp()
    return A

def solve_eigensystem(A, problem_type=SLEPc.EPS.ProblemType.HEP):
    # Create the results vectors
    xr, tmp = A.getVecs()
    xi, tmp = A.getVecs()

    # Setup the eigensolver
    E = SLEPc.EPS().create(comm = SLEPc.COMM_WORLD)
    E.setOperators(A,None)
    E.setDimensions(3,PETSc.DECIDE) #Find 3 eigenvalues
    E.setFromOptions()
    E.setProblemType( problem_type )

    # Solve the eigensystem
    E.solve()
    Print("")
    its = E.getIterationNumber()
    Print("Number of iterations of the method: %i" % its)
    sol_type = E.getType()
    Print("Solution method: %s" % sol_type)
    nev, ncv, mpd = E.getDimensions()
    Print("Number of requested eigenvalues: %i" % nev)
    tol, maxit = E.getTolerances()
    Print("Stopping condition: tol=%.4g, maxit=%d" % (tol, maxit))
    nconv = E.getConverged()
    Print("Number of converged eigenpairs: %d" % nconv)
    if nconv > 0:
        Print("")
        Print("        k          ||Ax-kx||/||kx|| ")
        Print("----------------- ------------------")
        for i in range(nconv):
            k = E.getEigenpair(i, xr, xi)
            error = E.computeRelativeError(i)
            if k.imag != 0.0:
              Print(" %9f%+9f j  %12g" % (k.real, k.imag, error))
            else:
              Print(" %12f       %12g" % (k.real, error))
              plt.plot(i, k.real,'o')
        Print("")

if __name__ == '__main__':
    opts = PETSc.Options()
    N = opts.getInt('N', 32)
    m = opts.getInt('m', N)
    n = opts.getInt('n', 1)

    da = PETSc.DMDA().create([m,n],stencil_width=1,stencil_type=0,comm=PETSc.COMM_WORLD)
    #0=star(5pt), 1=box(9pt)
    #stencil_width is how many ghost values we need away from proc's range
    #pde = Laplacian1D(da)

    Print("Symmetric Eigenproblem (matrix-free), "
          "N=%d (%dx%d grid)" % (m*n, m, n))
    A = construct_operator(m,da)
    solve_eigensystem(A)
    #plt.show()
