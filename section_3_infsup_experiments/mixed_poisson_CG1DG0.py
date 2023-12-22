# Compared to `mixed_poisson_graph.py` the main idea here is that we
# work on interval = (0, L) and we have no multipliers to get the continuity

from scipy.linalg import eigh
from dolfin import *
from xii import *

def get_system(mesh, f, p0):
    '''
    We are solving -div(grad(p)) = f with Dirichlet bcs
    '''
    V = FunctionSpace(mesh, 'CG', 1)
    Q = FunctionSpace(mesh, 'DG', 0)
    L = FunctionSpace(mesh, 'CG', 1)
    W = [V, Q]

    u, p = map(TrialFunction, W)
    v, q = map(TestFunction, W)

    a = block_form(W, 2)
    a[0][0] = inner(u, v)*dx
    a[0][1] = -inner(v.dx(0), p)*dx
    a[1][0] = -inner(u.dx(0), q)*dx

    hK = avg(CellDiameter(mesh))

    scale = Constant(assemble(Constant(1)*dx(domain=mesh)))
    
    a_prec = block_form(W, 2)
    a_prec[0][0] = inner(u, v)*dx + inner(v.dx(0), u.dx(0))*dx 
    a_prec[1][1] = inner(p, q)*dx

    n = FacetNormal(mesh)
    L_form = block_form(W, 1)
    L_form[0] = -inner(p0, v*n[0])*ds
    L_form[1] = -inner(f, q)*dx

    Lbcs = DirichletBC(L, Constant(0), 'on_boundary')

    A, B, b = map(ii_assemble, (a, a_prec, L_form))

    # A, _ = apply_bc(A, b=None, bcs=bcs)
    # B, _ = apply_bc(B, b=None, bcs=bcs)

    return A, B, b, W, Lbcs

    
# --------------------------------------------------------------------

if __name__ == '__main__':
    from petsc4py import PETSc
    from scipy.linalg import eigvalsh
    import tabulate, argparse
    import numpy as np

    from mixed_poisson_graphnics import unit_square_resize

    from graphnics.generators import honeycomb
    from graphnics.generate_arterial_tree import make_arterial_tree as arterial_tree

    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-length', type=float, default=1.0)
    args, _ = parser.parse_known_args()

    p0 = Expression('sin(k*pi*x[0])', degree=4, k=2)
    u0 = Expression('-(k*pi)*cos(k*pi*x[0])', degree=4, k=2)
    f = Expression('(k*pi)*(k*pi)*sin(k*pi*x[0])', degree=4, k=2)

    history, error_history = [], []
    for k in range(2, 10):
        n = 2**k

        mesh = UnitIntervalMesh(n)
        # mesh = honeycomb(k, k).mesh
        
        # mesh = unit_square_resize(mesh)        
        
        A, B, b, W, Lbcs = get_system(mesh, f, p0)

        A_, B_, b_ = map(ii_convert, (A, B, b))

        wh = ii_Function(W)
        solve(A_, wh.vector(), b_)

        uh, ph = wh

        eu = errornorm(u0, uh, 'L2')
        ep = errornorm(p0, ph, 'L2')
        print(f'|eu|_0 = {eu:.4E} |ep| = {ep:.4E}')

        Arr, Brr = A_.array(), B_.array()        
        eigw = eigvalsh(Arr, Brr)

        num_zeros = len(eigw[np.abs(eigw) < 1E-10])
        
        idx = np.argsort(np.abs(eigw))
        lmin, lmax = eigw[idx[0]], eigw[idx[-1]]
        
        lnegmin, lnegmax = np.min(eigw[eigw < 0]), np.max(eigw[eigw < 0])
        lposmin, lposmax = np.min(eigw[eigw > 0]), np.max(eigw[eigw > 0])        

        hmin = W[0].mesh().hmin()
        history.append((hmin, lmin, lmax, lmax/abs(lmin), lnegmin, lnegmax, lposmin, lposmax, num_zeros))
        error_history.append((hmin, eu, ep))
        
        print(history[-1])
        print('\t', error_history[-1])
        
    history = np.array(history)
    print()
    print(tabulate.tabulate(history, headers=('h', 'lmin', 'lmax', 'cond',
                                              'minl-', 'maxl-', 'minl+', 'maxl+',
                                              'nzeros'), tablefmt='latex'))
    print()

    error_history = np.array(error_history)
    print()
    print(tabulate.tabulate(error_history, headers=('h', '|eu|_0', '|ep|_0')))
    print()
    
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(history[:, 0], history[:, 1])
    plt.show()
        
