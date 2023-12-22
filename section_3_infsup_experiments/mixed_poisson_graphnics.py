# In contrast to mixed_poisson_graph here the multiplier space is really R^n
import petsc4py, sys
petsc4py.init(sys.argv)

from collections import defaultdict
from scipy.linalg import eigh
import itertools
from dolfin import *
import numpy as np
from xii import *
import sys

sys.setrecursionlimit(1000_000)


def unit_square_resize(mesh):
    x = mesh.coordinates()
    xmin, ymin = np.min(x, axis=0)[:2]
    xmax, ymax = np.max(x, axis=0)[:2]
    print(f'>> [{xmin}, {xmax}] x [{ymin}, {ymax}]')
    
    if mesh.geometry().dim() == 3:
        shift = np.array([[xmin, ymin, 0]])
    else:
        shift = np.array([[xmin, ymin]])
    x -= shift
        
    box_length = max(xmax - xmin, ymax - ymin)
    x /= box_length

    xmin, ymin = np.min(x, axis=0)[:2]
    xmax, ymax = np.max(x, axis=0)[:2]
    print(f'<< [{xmin}, {xmax}] x [{ymin}, {ymax}]')    

    return mesh


def ReLU_mesh(n):
    mesh = UnitSquareMesh(n, n, 'crossed')
    facet_f = MeshFunction('size_t', mesh, 1, 0)


    # CompiledSubDomain('near(x[1], 0)').mark(facet_f, 1)
    
    subdomains = ('(near(x[1], 0) && x[0] < 0.5 + tol)',
                  '(near(x[0]-0.5, x[1]))')

    subdomains = ('(near(x[1], 0.5) && x[0] < 0.5 + tol)',
                  '(near(x[0], x[1]) && x[0] > 0.5 - tol)',
                  '(near(1-x[0], x[1]) && x[0] > 0.5 - tol)')

    subdomains = ('(near(x[1], 0.5) && x[0] < 0.5 + tol)',
                  '(near(x[0], x[1]) && x[0] > 0.5 - tol)',
                  '(near(1-x[0], x[1]) && x[0] > 0.5 - tol)')

    subdomains = ('(near(x[1], 0.5) && x[0] < 0.5 + tol)',
                  '(near(x[0], x[1]) && x[0] > 0.5 - tol)',
                  '(near(1-x[0], x[1]) && x[0] > 0.5 - tol)',
                  '(near(x[1], 0.75) && x[0] > 0.75 + tol)',
                  '(near(x[1], 0.25) && x[0] > 0.75 + tol)')
    
    CompiledSubDomain(' || '.join(subdomains), tol=1E-10).mark(facet_f, 1)    

    return EmbeddedMesh(facet_f, 1)
    
# --------------------------------------------------------------------

if __name__ == '__main__':
    from petsc4py import PETSc
    from scipy.linalg import eigvalsh, eigh
    from collections import Counter
    import tabulate, os, argparse
    import numpy as np

    from mixed_poisson_graph_ii_bif_dSTag import get_system
    from graphnics.generators import honeycomb
    from graphnics.generate_arterial_tree import make_arterial_tree as arterial_tree
    import binary_tree

    OptDB = PETSc.Options()
    geometry = OptDB.getString('geometry', 'line')
    kmax = OptDB.getInt('kmax', 5)
    rescale = bool(OptDB.getInt('rescale', 1))

    nrefs = OptDB.getInt('nrefs', 0)
    
    root = os.path.join('./results', geometry)
    not os.path.exists(root) and os.makedirs(root)

    def get_path(filename, root=root, rescale=rescale, nrefs=nrefs):
        return os.path.join(root, '_'.join([f'rescale{rescale}', f'nrefs{nrefs}', filename]))
    
    set_log_level(40)
    
    p0 = Expression('sin(k*pi*x[0])', degree=4, k=2)
    u0 = Expression('-(k*pi)*cos(k*pi*x[0])', degree=4, k=2)
    f = Expression('(k*pi)*(k*pi)*sin(k*pi*x[0])', degree=4, k=2)

    tree_generator = binary_tree.binary_tree()
    
    history, error_history = [], []
    for k in range(2, kmax):
        n = 2**k

        if geometry == 'honeycomb':
            mesh = honeycomb(k, k).mesh
            
        elif geometry == 'artery':
            mesh = arterial_tree(k, seed=46).mesh
            
        elif geometry == 'tree':
            tree = next(tree_generator)
            mesh = binary_tree.tree_mesh(tree)

        elif geometry == 'line':
            mesh = UnitIntervalMesh(n)
            
        else:
            raise ValueError

        for _ in range(nrefs):
            mesh = refine(mesh)
        
        _, v2c = mesh.init(0, 1), mesh.topology()(0, 1)
        node_degrees = Counter([len(v2c(v)) for v in range(mesh.num_vertices())])
        print('\t', node_degrees)
        
        # Resize for poincare
        if mesh.geometry().dim() > 1:
            if rescale:
                mesh = unit_square_resize(mesh)
            
            xmin, ymin = mesh.coordinates().min(axis=0)
            xmax, ymax = mesh.coordinates().max(axis=0)
            diameter = max(xmax - xmin, ymax - ymin)
        else:
            xmin, = mesh.coordinates().min(axis=0)
            xmax,  = mesh.coordinates().max(axis=0)
            diameter = xmax - xmin
        length = sum(c.volume() for c in cells(mesh))
        
        A, B, b, W = get_system(mesh, f, p0)

        foo = A[0][1]
        A_, B_, b_ = map(ii_convert, (A, B, b))

        print('Conversion')
        
        offsets = np.r_[np.cumsum([0, W[0].dim(), W[1].dim()]), A_.size(0)]
        offsets = [PETSc.IS().createGeneral(np.arange(offsets[i], offsets[i+1], dtype='int32'))
                   for i in range(3)]

        wh = ii_Function(W)
        solve(A_, wh.vector(), b_)

        uh, ph, *_ = wh
        V, Q, *Ls = W

        # FIXME: what's with the mode?
        #        tangent curve?
        
        eu = sqrt(assemble(
            inner(u0-uh, u0-uh)*dx(metadata={'quadrature_degree': 5})
            + inner(uh.dx(0) - f, uh.dx(0) -f)*dx(metadata={'quadrature_degree': 5})
        ))
        ep = errornorm(p0, ph, 'L2')
        print(f'|eu|_1 = {eu:.4E} |ep|_0 = {ep:.4E}')

        Arr, Brr = A_.array(), B_.array()        
        eigw, eigv = eigh(Arr, Brr)
        eigv = eigv.T

        zeros = np.abs(eigw) < 1E-10
        nzeros = sum(zeros)
        
        idx = np.argsort(np.abs(eigw))
        lmin, lmax = eigw[idx[0]], eigw[idx[-1]]
        
        lnegmin, lnegmax = np.min(eigw[eigw < 0]), np.max(eigw[eigw < 0])
        lposmin, lposmax = np.min(eigw[eigw > 0]), np.max(eigw[eigw > 0])        

        dimLM = sum(Li.dim() for Li in Ls)
        
        hmin = W[0].mesh().hmin()
        dimLM = sum(count for degree, count in node_degrees.items() if degree > 1)
        nbifs = sum(count for degree, count in node_degrees.items() if degree > 2)
        ndofs = A_.size(0)
        
        history.append((k, hmin,
                        lmin, lmax, lmax/abs(lmin),
                        lnegmin, lnegmax, lposmin, lposmax, nzeros,
                        dimLM, nbifs, ndofs,
                        diameter, length
                        ))
        error_history.append((k, hmin, ndofs, dimLM, eu, ep))

        # Visualize given inf-sup mode
        mode_coefs = eigv[idx[0]]
        inf_sup_mode = ii_Function(W)
        inf_sup_mode[0].vector().set_local(mode_coefs[np.array(offsets[0])])
        inf_sup_mode[1].vector().set_local(mode_coefs[np.array(offsets[1])])        

        File(get_path(f'{geometry}_infsup_{k}_uh.pvd')) << inf_sup_mode[0]
        File(get_path(f'{geometry}_infsup_{k}_ph.pvd')) << inf_sup_mode[1]
        
        print(history[-1])
        print('\t', error_history[-1])

    File(os.path.join('results', 'uh.pvd')) << uh
    File(os.path.join('results', 'ph.pvd')) << ph
        
    history = np.array(history)
    print()
    print(tabulate.tabulate(history, headers=('N', 'h', 'lmin', 'lmax', 'cond',
                                              'minl-', 'maxl-', 'minl+', 'maxl+', 'nzeros',
                                              'dimLM', 'nbifs', 'ndofs',
                                              'diam', 'length'), tablefmt='latex'))
    print()

    with open(get_path('spectra.txt'), 'w') as txt:
        print(get_path('spectra.txt'))        
        txt.write(tabulate.tabulate(history, headers=('N', 'h', 'lmin', 'lmax', 'cond',
                                                      'minl-', 'maxl-', 'minl+', 'maxl+', 'nzeros',
                                                      'dimLM', 'nbifs', 'ndofs',
                                                      'diam', 'length')))
        
    error_history = np.array(error_history)
    print()
    print(tabulate.tabulate(error_history, headers=('h', 'ndofs', 'dimLM', '|eu|_1', '|ep|_0')))
    print()

    with open(get_path('error.txt'), 'w') as txt:
        print(get_path('error.txt'))
        txt.write(tabulate.tabulate(error_history, headers=('h', 'ndofs', 'dimLM', '|eu|_1', '|ep|_0')))
