# The main idea here is that we put the multiplier only on the interior
# facets. So there is some manipulation with the matrix in order to get
# the subsytem corresponding only to the interior facets.

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
    a[0][2] = inner(jump(v), avg(lm))*dS + inner(v, lm)*ds
    
    a[1][0] = -inner(u.dx(0), q)*dx
    
    a[2][0] = inner(jump(u), avg(dlm))*dS + inner(u, dlm)*ds

    hK = CellVolume(mesh)
    
    a_prec = block_form(W, 2)
    a_prec[0][0] = (inner(u, v)*dx + inner(v.dx(0), u.dx(0))*dx +
                    (1/avg(hK))*inner(jump(u), jump(v))*dS #+ (1/hK)*inner(u, v)*ds
                    )
    a_prec[1][1] = inner(p, q)*dx
    a_prec[2][2] = avg(hK)*inner(avg(lm), avg(dlm))*dS + hK*inner(lm, dlm)*ds

    n = FacetNormal(mesh)
    L_form = block_form(W, 1)
    L_form[0] = -inner(p0, v*n[0])*ds
    L_form[1] = -inner(f, q)*dx

    Lbcs = DirichletBC(L, Constant(0), 'on_boundary')
    bcs = [[], [], [Lbcs]]
    
    A, B, b = map(ii_assemble, (a, a_prec, L_form))

    A, _ = apply_bc(A, b=None, bcs=bcs)
    B, _ = apply_bc(B, b=None, bcs=bcs)

    return A, B, b, W, Lbcs
    
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
    for k in range(2, 10):
        n = 2**k

        mesh = UnitIntervalMesh(n)
        
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

        eu = assemble(
            inner(u0-uh, u0-uh)*dx(metadata={'quadrature_degree': 5})
            + inner(uh.dx(0) - f, uh.dx(0) -f)*dx(metadata={'quadrature_degree': 5})
        )
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
        
