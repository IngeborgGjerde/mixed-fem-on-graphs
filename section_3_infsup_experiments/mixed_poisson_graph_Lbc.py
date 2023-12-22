# Here the main idea is to eliminate boundary dofs by boundary conditions

from scipy.linalg import eigh
from dolfin import *
from xii import *


def get_system(mesh, f, p0):
    '''
    We are solving -div(grad(p)) = f with Dirichlet bcs
    '''
    V = FunctionSpace(mesh, 'DG', 1)
    Q = FunctionSpace(mesh, 'DG', 0)
    L = FunctionSpace(mesh, 'CG', 1)
    W = [V, Q, L]

    x, = SpatialCoordinate(mesh)
    
    u, p, lm = map(TrialFunction, W)
    v, q, dlm = map(TestFunction, W)

    a = block_form(W, 2)
    a[0][0] = inner(u, v)*dx
    a[0][1] = -inner(v.dx(0), p)*dx
    a[0][2] = inner(jump(v), avg(lm))*dS# + inner(lm, v)*ds
    
    a[1][0] = -inner(u.dx(0), q)*dx
    
    a[2][0] = inner(jump(u), avg(dlm))*dS# + inner(dlm, u)*ds

    hK = avg(CellDiameter(mesh))
    
    a_prec = block_form(W, 2)
    a_prec[0][0] = inner(u, v)*dx + inner(v.dx(0), u.dx(0))*dx + (1/hK)*inner(jump(u), jump(v))*dS
    a_prec[1][1] = inner(p, q)*dx
    a_prec[2][2] = hK*inner(avg(lm), avg(dlm))*dS

    n = FacetNormal(mesh)
    L_form = block_form(W, 1)
    # L_form[0] = -inner(p0, v*n[0])*ds
    L_form[1] = -inner(f, q)*dx

    bcs = [[], [], [DirichletBC(L, p0, 'on_boundary')]]

    A, B, b = map(ii_assemble, (a, a_prec, L_form))

    A, b = apply_bc(A, b=b, bcs=bcs)
    B, _ = apply_bc(B, b=None, bcs=bcs)

    return A, B, b, W, bcs
    
# --------------------------------------------------------------------

if __name__ == '__main__':
    from petsc4py import PETSc
    from scipy.linalg import eigvalsh
    import tabulate
    import numpy as np


    p0 = Expression('sin(k*pi*x[0])', degree=4, k=2)
    u0 = Expression('-(k*pi)*cos(k*pi*x[0])', degree=4, k=2)
    f = Expression('(k*pi)*(k*pi)*sin(k*pi*x[0])', degree=4, k=2)
    
    history, error_history = [], []
    for k in range(4, 10):
        n = 2**k

        mesh = UnitIntervalMesh(n)
        
        A, B, b, W, Lbcs = get_system(mesh, f, p0)

        A_, B_, b_ = map(ii_convert, (A, B, b))

        wh = ii_Function(W)
        solve(A_, wh.vector(), b_)

        uh, ph, _ = wh

        eu = sqrt(assemble(
            inner(u0-uh, u0-uh)*dx(metadata={'quadrature_degree': 5})
            + inner(uh.dx(0) - f, uh.dx(0) -f)*dx(metadata={'quadrature_degree': 5})
        ))
        ep = errornorm(p0, ph, 'L2')
        print(f'|eu|_1 = {eu:.4E} |ep|_0 = {ep:.4E}')

        
        Arr, Brr = A_.array(), B_.array()        
        eigw = eigvalsh(Arr, Brr)
        
        idx = np.argsort(np.abs(eigw))
        lmin, lmax = eigw[idx[0]], eigw[idx[-1]]

        lnegmin, lnegmax = np.min(eigw[eigw < 0]), np.max(eigw[eigw < 0])
        lposmin, lposmax = np.min(eigw[eigw > 0]), np.max(eigw[eigw > 0])        

        hmin = W[0].mesh().hmin()
        history.append((hmin, lmin, lmax, lmax/abs(lmin), lnegmin, lnegmax, lposmin, lposmax))
        error_history.append((hmin, eu, ep))
        
        print(history[-1])
        print('\t', error_history[-1])
        
    history = np.array(history)
    print()
    print(tabulate.tabulate(history, headers=('h', 'lmin', 'lmax', 'cond',
                                              'minl-', 'maxl-', 'minl+', 'maxl+'), tablefmt='latex'))
    print()

    error_history = np.array(error_history)
    print()
    print(tabulate.tabulate(error_history, headers=('h', '|eu|_0', '|ep|_0')))
    print()
    
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(history[:, 0], history[:, 1])
    plt.show()
