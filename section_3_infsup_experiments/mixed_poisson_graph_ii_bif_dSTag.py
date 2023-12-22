# In contrast to mixed_poisson_graph here the multiplier space is really R^n
# Use tagging

from collections import defaultdict
from scipy.linalg import eigh
import numpy as np
import itertools
from dolfin import *
from xii import *
import sys

sys.setrecursionlimit(5_000)


from mixed_poisson_graph_ii_bif import vertex_degrees, cell_orientations


def get_system(mesh, f, p0, kappa=Constant(1)):
    '''
    We are solving -div(grad(p)) = f with Dirichlet bcs
    '''
    vertex_f = vertex_degrees(mesh)
    bifurcation_nodes, = np.where(vertex_f.array() > 2)
    interior_nodes, = np.where(vertex_f.array() == 2)
    exterior_nodes, = np.where(vertex_f.array() == 1)
    nb, ni, ne = map(len, (bifurcation_nodes, interior_nodes, exterior_nodes))

    interior_vertex_f = MeshFunction('size_t', mesh, 0, 0)
    interior_vertex_f.array()[interior_nodes] = 2    

    dSDG = Measure('dS', domain=mesh, subdomain_data=interior_vertex_f)
    # Renumber
    active_vertex_f = MeshFunction('size_t', mesh, 0, 0)
    dS = Measure('dS', domain=mesh, subdomain_data=active_vertex_f)

    tau = TangentCurve(mesh)
    Grad = lambda v, tau=tau: dot(grad(v), tau)
    
    orientations = cell_orientations(tau, np.r_[bifurcation_nodes, interior_nodes, exterior_nodes])

    V = FunctionSpace(mesh, 'DG', 1)
    Q = FunctionSpace(mesh, 'DG', 0)
    # NOTE: This could equally be a VectorFunctionSpace(...) but then
    # we'd trigger compiler based on the dimensionality of LM space. And
    # the same would happend in the coupling form
    nnodes = nb # + ne
    Ls = [FunctionSpace(mesh, 'R', 0)]*nnodes
    W = [V, Q] + Ls

    u, p, *lms = map(TrialFunction, W)
    v, q, *dlms = map(TestFunction, W)

    ikappa = 1/kappa
    
    a = block_form(W, 2)
    a[0][0] = ikappa*inner(u, v)*dx
    a[0][1] = -inner(Grad(v), p)*dx
    a[1][0] = -inner(Grad(u), q)*dx
    
    # Add the DG penalty
    hKDG = avg(CellVolume(mesh))
    a[0][1] += -inner(avg(p), jump(v))*dSDG(2)
    a[1][0] += -inner(avg(q), jump(u))*dSDG(2)

    cell_volumes = [c.volume() for c in cells(mesh)]
    
    xmin, ymin = np.min(mesh.coordinates(), axis=0)
    xmax, ymax = np.max(mesh.coordinates(), axis=0)    
    #length = assemble(Constant(1)*dx(mesh))
    #diam = length 
    diam = Constant(max(xmax-xmin, ymax-ymin))
    
    a[0][0] += ikappa*diam**2*(1/hKDG)*inner(jump(u), jump(v))*dSDG(2)

    _, v2c = mesh.init(0, 1), mesh.topology()(0, 1)

    plus = lambda arg: arg('+')

    def dp(x, y, vi, foo=active_vertex_f, dS=dS):
        foo.set_all(0)
        foo[vi] = 1
        
        return inner(x, y)*dS(1)
    
    # Deal with the coupling
    for i, vi in enumerate(bifurcation_nodes):
        for cell in v2c(vi):
            orientation = Constant(orientations[vi][cell])
            trace_v, trace_u = plus(PointTrace(v, point=vi, cell=cell)), plus(PointTrace(u, point=vi, cell=cell)) 

            a[0][2+i] += orientation*dp(trace_v, avg(lms[i]), vi)
            a[2+i][0] += orientation*dp(trace_u, avg(dlms[i]), vi)

    a_prec = block_form(W, 2)
    # V-inner product
    a_prec[0][0] = ikappa*inner(u, v)*dx + ikappa*diam**2*inner(Grad(v), Grad(u))*dx
    # DG bit
    gamma = Constant(1)
    a_prec[0][0] += ikappa*diam**2*(gamma/hKDG)*inner(jump(u), jump(v))*dSDG(2)    
     
    # add the penalty term
    for i, vi in enumerate(bifurcation_nodes):
        cells_vi = v2c(vi)
        i_cell_volumes = [cell_volumes[cell] for cell in cells_vi]
        hK = Constant(np.mean(i_cell_volumes))

        for cu, cv in itertools.product(cells_vi, cells_vi):
            ou = Constant(orientations[vi][cu])
            ov = Constant(orientations[vi][cv])
            a_prec[0][0] += ikappa*diam**2*(1/hK)*dp(ou*plus(PointTrace(u, point=vi, cell=cu)),
                                                     ov*plus(PointTrace(v, point=vi, cell=cv)),
                                                     vi)
            
    # Q-inner product
    a_prec[1][1] = kappa*(1/diam**2)*inner(p, q)*dx

    # LM-inner product
    for i, vi in enumerate(bifurcation_nodes):
        i_cell_volumes = [cell_volumes[cell] for cell in v2c(vi)]
        hK = Constant(np.mean(i_cell_volumes))

        a_prec[2+i][2+i] = kappa*(1/diam**2)*hK*dp(avg(lms[i]), avg(dlms[i]), vi)

   
    n = FacetNormal(mesh)
    L_form = block_form(W, 1)
    L_form[0] = -inner(p0, v*n[0])*ds
    L_form[1] = -inner(f, q)*dx

    A, B, b = map(ii_assemble, (a, a_prec, L_form))

    print('Assembly')
    
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
    for k in range(2, 9):
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
