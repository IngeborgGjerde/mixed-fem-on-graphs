# In contrast to mixed_poisson_graph here the multiplier space is really R^n

from collections import defaultdict
from scipy.linalg import eigh
import numpy as np
import itertools
from dolfin import *
from xii import *
import sys

sys.setrecursionlimit(5_000)


def vertex_degrees(mesh):
    '''Of a graph mesh'''
    assert mesh.topology().dim() == 1

    _, v2c = mesh.init(0, 1), mesh.topology()(0, 1)

    degrees = np.array([len(v2c(v)) for v in range(mesh.num_vertices())], dtype='uintp')
    vertex_f = MeshFunction('size_t', mesh, 0, 0)
    vertex_f.array()[:] = degrees

    return vertex_f


def cell_orientations(tau, lm_nodes):
    ''' plus/minus the cell based on tau in node'''
    mesh = tau.function_space().mesh()
    gdim = mesh.geometry().dim()
    
    orientations = defaultdict(dict)

    x = mesh.coordinates()
    _, v2c = mesh.init(0, 1), mesh.topology()(0, 1)
    for v_ in lm_nodes:
        v = x[v_]
        
        for cell_ in v2c(v_):
            cell = Cell(mesh, cell_)

            mp = cell.midpoint().array()[:gdim]
            tau_cell = tau(mp)

            orientations[v_][cell_] = 1 if np.dot(tau_cell, v - mp) > 0 else -1
    return orientations


def get_system(mesh, f, p0, kappa=Constant(1)):
    '''
    We are solving -div(grad(p)) = f with Dirichlet bcs
    '''
    vertex_f = vertex_degrees(mesh)
    interior_nodes, = np.where(vertex_f.array() > 1)
    exterior_nodes, = np.where(vertex_f.array() == 1)
    ni, ne = len(interior_nodes), len(exterior_nodes)
    assert ni

    tau = TangentCurve(mesh)
    Grad = lambda v, tau=tau: dot(grad(v), tau)
    
    orientations = cell_orientations(tau, np.r_[interior_nodes, exterior_nodes])

    V = FunctionSpace(mesh, 'DG', 1)
    Q = FunctionSpace(mesh, 'DG', 0)
    # NOTE: This could equally be a VectorFunctionSpace(...) but then
    # we'd trigger compiler based on the dimensionality of LM space. And
    # the same would happend in the coupling form
    nnodes = ni# + ne
    Ls = [FunctionSpace(mesh, 'R', 0)]*nnodes
    W = [V, Q] + Ls

    u, p, *lms = map(TrialFunction, W)
    v, q, *dlms = map(TestFunction, W)

    ikappa = 1/kappa
    
    a = block_form(W, 2)
    a[0][0] = ikappa*inner(u, v)*dx
    a[0][1] = -inner(Grad(v), p)*dx
    a[1][0] = -inner(Grad(u), q)*dx

    _, v2c = mesh.init(0, 1), mesh.topology()(0, 1)

    plus = lambda arg: arg('+')    
    scale_i = Constant(1./ni)
    dpi = lambda x, y: scale_i*inner(x, y)*dS
    
    # Deal with the coupling
    for i, vi in enumerate(interior_nodes):
        for cell in v2c(vi):
            orientation = Constant(orientations[vi][cell])
            trace_v, trace_u = plus(PointTrace(v, point=vi, cell=cell)), plus(PointTrace(u, point=vi, cell=cell)) 

            a[0][2+i] += orientation*dpi(trace_v, avg(lms[i]))
            a[2+i][0] += orientation*dpi(trace_u, avg(dlms[i]))

    # scale_e = Constant(1./ne)
    # dpe = lambda x, y: scale_e*inner(x, y)*ds
    
    # for i, vi in enumerate(exterior_nodes, len(interior_nodes)):
    #     cell, = v2c(vi)
        
    #     orientation = Constant(orientations[vi][cell])
    #     trace_v, trace_u = PointTrace(v, point=vi, cell=cell), PointTrace(u, point=vi, cell=cell)

    #     a[0][2+i] += orientation*dpe(trace_v, lms[i])
    #     a[2+i][0] += orientation*dpe(trace_u, dlms[i])            

                
    cell_volumes = [c.volume() for c in cells(mesh)]
    
    a_prec = block_form(W, 2)
    # V-inner product
    a_prec[0][0] = ikappa*inner(u, v)*dx + ikappa*inner(Grad(v), Grad(u))*dx 
    # add the penalty term
    for i, vi in enumerate(interior_nodes):
        cells_vi = v2c(vi)
        i_cell_volumes = [cell_volumes[cell] for cell in cells_vi]
        hK = Constant(np.mean(i_cell_volumes))

        for cu, cv in itertools.product(cells_vi, cells_vi):
            ou = Constant(orientations[vi][cu])
            ov = Constant(orientations[vi][cv])
            a_prec[0][0] += ikappa*(1/hK)*dpi(ou*plus(PointTrace(u, point=vi, cell=cu)),
                                              ov*plus(PointTrace(v, point=vi, cell=cv)))
            
    # Q-inner product
    a_prec[1][1] = kappa*inner(p, q)*dx

    # LM-inner product
    for i, vi in enumerate(interior_nodes):
        i_cell_volumes = [cell_volumes[cell] for cell in v2c(vi)]
        hK = Constant(np.mean(i_cell_volumes))

        a_prec[2+i][2+i] = kappa*hK*dpi(avg(lms[i]), avg(dlms[i]))

    # for i, vi in enumerate(exterior_nodes, ni):
    #     i_cell_volumes = [cell_volumes[cell] for cell in v2c(vi)]
    #     hK = Constant(np.mean(i_cell_volumes))

    #     a_prec[2+i][2+i] = hK*dpe(lms[i], dlms[i])

    n = FacetNormal(mesh)
    L_form = block_form(W, 1)
    L_form[0] = -inner(p0, v*n[0])*ds
    L_form[1] = -inner(f, q)*dx

    A, B, b = map(ii_assemble, (a, a_prec, L_form))

    # bcs = [[], []]
    # for _ in range(ni):
    #     bcs.append([])
    # for _ in range(ne):
    #     bcs.append([{0: 0}])
    
    # A, b = apply_bc(A, b=None, bcs=bcs)
    # B, _ = apply_bc(B, b=None, bcs=bcs)

    return A, B, b, W
    
# --------------------------------------------------------------------

if __name__ == '__main__':
    from petsc4py import PETSc
    from scipy.linalg import eigvalsh
    import tabulate
    import numpy as np

    set_log_level(40)

    p0 = Expression('sin(k*pi*x[0])', degree=4, k=2)
    u0 = Expression('-(k*pi)*cos(k*pi*x[0])', degree=4, k=2)
    f = Expression('(k*pi)*(k*pi)*sin(k*pi*x[0])', degree=4, k=2)
    
    history, error_history = [], []
    for k in range(2, 6):
        n = 2**k

        mesh = UnitIntervalMesh(n)
        
        A, B, b, W = get_system(mesh, f, p0)

        A_, B_, b_ = map(ii_convert, (A, B, b))

        wh = ii_Function(W)
        solve(A_, wh.vector(), b_)

        uh, ph, *_ = wh

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

        _, _, *Ls = W
        dimLM = sum(Li.dim() for Li in Ls)
        
        hmin = W[0].mesh().hmin()
        history.append((hmin, lmin, lmax, lmax/abs(lmin), lnegmin, lnegmax, lposmin, lposmax))
        error_history.append((hmin, dimLM, eu, ep))
        
        print(history[-1])
        print('\t', error_history[-1])
        
    history = np.array(history)
    print()
    print(tabulate.tabulate(history, headers=('h', 'lmin', 'lmax', 'cond',
                                             'minl-', 'maxl-', 'minl+', 'maxl+'), tablefmt='latex'))
    print()

    error_history = np.array(error_history)
    print()
    print(tabulate.tabulate(error_history, headers=('h', 'dimLM', '|eu|_0', '|ep|_0')))
    print()
