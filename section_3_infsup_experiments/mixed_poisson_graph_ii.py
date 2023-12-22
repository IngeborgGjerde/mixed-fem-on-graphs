# In contrast to mixed_poisson_graph here the multiplier space is really R^n

from scipy.linalg import eigh
from dolfin import *
from xii import *


def vertex_degrees(mesh):
    '''Of a graph mesh'''
    assert mesh.topology().dim() == 1

    _, v2c = mesh.init(0, 1), mesh.topology()(0, 1)

    degrees = np.array([len(v2c(v)) for v in range(mesh.num_vertices())], dtype='uintp')
    vertex_f = MeshFunction('size_t', mesh, 0, 0)
    vertex_f.array()[:] = degrees

    return vertex_f


def get_system(mesh, f, p0):
    '''
    We are solving -div(grad(p)) = f with Dirichlet bcs
    '''
    vertex_f = vertex_degrees(mesh)
    lm_nodes, = np.where(vertex_f.array() == 2)
    ni = len(lm_nodes)
    assert ni

    dS = Measure('dS', domain=mesh, subdomain_data=vertex_f)
    
    V = FunctionSpace(mesh, 'DG', 1)
    Q = FunctionSpace(mesh, 'DG', 0)
    # NOTE: This could equally be a VectorFunctionSpace(...) but then
    # we'd trigger compiler based on the dimensionality of LM space. And
    # the same would happend in the coupling form
    Ls = [FunctionSpace(mesh, 'R', 0)]*ni
    W = [V, Q] + Ls

    u, p, *lms = map(TrialFunction, W)
    v, q, *dlms = map(TestFunction, W)

    a = block_form(W, 2)
    a[0][0] = inner(u, v)*dx
    a[0][1] = -inner(v.dx(0), p)*dx
    a[1][0] = -inner(u.dx(0), q)*dx

    _, v2c = mesh.init(0, 1), mesh.topology()(0, 1)

    plus = lambda arg: arg('+')    
    scale = Constant(1./ni)
    # Deal with the coupling
    for i, vi in enumerate(lm_nodes):
        cell0, cell1 = v2c(vi)
        # NOTE: In xii interpreter we get in trouble with term(trace_v0 - trace_v1, ...)
        # because it things it has 2 TestFunctions in the integrand therefore we do the linear
        # combination on the level for form. Also, the restriction is to satisfy FFC;
        # since point trace will end up in R space any restriction would do
        trace_v0, trace_v1 = plus(PointTrace(v, point=vi, cell=cell0)), plus(PointTrace(v, point=vi, cell=cell1))
        trace_u0, trace_u1 = plus(PointTrace(u, point=vi, cell=cell0)), plus(PointTrace(u, point=vi, cell=cell1))       

        # Fake orientation -> jump
        if Cell(mesh, cell0).midpoint().array()[0] > Cell(mesh, cell1).midpoint().array()[0]:
            # Finally, after restriction we are integrating 2 reals and ideally we do this
            # only at the relavant facet. However, that would mean wiring up dS with some 
            # enumeration of the relevant facets -> triggers the compiler. So to avoid that
            # we do something suboptimal from the implementation point of view - intergrate
            # on every facet and then we componsante for this by scaling
            a[0][2+i] = scale*inner(trace_v0, avg(lms[i]))*dS(2) - scale*inner(trace_v1, avg(lms[i]))*dS(2)
            a[2+i][0] = scale*inner(trace_u0, avg(dlms[i]))*dS(2) - scale*inner(trace_u1, avg(dlms[i]))*dS(2)
        else:
            a[0][2+i] += scale*inner(trace_v1, avg(lms[i]))*dS(2) - scale*inner(trace_v0, avg(lms[i]))*dS(2)
            a[2+i][0] += scale*inner(trace_u1, avg(dlms[i]))*dS(2) - scale*inner(trace_u0, avg(dlms[i]))*dS(2)

    hK = avg(CellVolume(mesh))

    a_prec = block_form(W, 2)
    a_prec[0][0] = inner(u, v)*dx + inner(v.dx(0), u.dx(0))*dx + (1/hK)*inner(jump(u), jump(v))*dS(2)
    a_prec[1][1] = inner(p, q)*dx

    cell_volumes = [c.volume() for c in cells(mesh)]
    for i, vi in enumerate(lm_nodes):
        i_cell_volumes = [cell_volumes[cell] for cell in v2c(vi)]
        hK = Constant(np.mean(i_cell_volumes))

        a_prec[2+i][2+i] = hK*scale*inner(avg(lms[i]), avg(dlms[i]))*dS(2)

    n = FacetNormal(mesh)
    L_form = block_form(W, 1)
    L_form[0] = -inner(p0, v*n[0])*ds
    L_form[1] = -inner(f, q)*dx

    A, B, b = map(ii_assemble, (a, a_prec, L_form))

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
    for k in range(2, 8):
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
