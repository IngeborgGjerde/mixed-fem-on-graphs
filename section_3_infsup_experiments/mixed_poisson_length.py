# Comparedd to `mixed_poisson_graph.py` the main idea here is that we
# work on interval = (0, L)

from scipy.linalg import eigh
from dolfin import *
from xii import *

from mixed_poisson_graph import get_system
    
# --------------------------------------------------------------------

if __name__ == '__main__':
    from petsc4py import PETSc
    from scipy.linalg import eigvalsh
    import tabulate, argparse
    import numpy as np

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
        mesh.coordinates()[:] *= args.length
        
        A, B, b, W, Lbcs = get_system(mesh, f, p0)

        A_, B_, b_ = map(ii_convert, (A, B, b))

        offsets = np.cumsum(np.r_[0, [Wi.dim() for Wi in W]])
        Vdofs, Qdofs, Ldofs = (np.arange(offsets[i], offsets[i+1]) for i in range(len(W)))
        
        Lbc_dofs = set(min(Ldofs) + dof for dof in Lbcs.get_boundary_values().keys())
        Ldofs = np.sort(np.fromiter(set(Ldofs) - Lbc_dofs, dtype=Vdofs.dtype))

        indices = np.r_[Vdofs, Qdofs, Ldofs]
        indices = PETSc.IS().createGeneral(np.asarray(indices, dtype='int32'))

        A_ = PETScMatrix(as_backend_type(A_).mat().createSubMatrix(indices, indices))
        B_ = PETScMatrix(as_backend_type(B_).mat().createSubMatrix(indices, indices))        
        b_ = PETScVector(as_backend_type(b_).vec().getSubVector(indices))
        
        x = Vector(mesh.mpi_comm(), A_.size(0))
        solve(A_, x, b_)

        x_ = x.get_local()
        uh = Function(W[0]); uh.vector().set_local(x_[Vdofs])
        ph = Function(W[1]); ph.vector().set_local(x_[Qdofs])

        eu = errornorm(u0, uh, 'L2')
        ep = errornorm(p0, ph, 'L2')
        print(f'|eu|_0 = {eu:.4E} |ep| = {ep:.4E}')

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
        
