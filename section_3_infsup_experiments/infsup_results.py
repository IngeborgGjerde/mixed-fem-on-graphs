
from graphnics import *
import sys
sys.path.append('../other/')
from infsup_eigvals import *
sys.path.append('../graphnics/data')
from generate_arterial_tree import *
import imp
import infsup_eigvals



imp.reload(infsup_eigvals)


print('n_bifurcations \\ N_refs \\\\')


history = []
for N in [2, 3, 4, 5, 6, 7, 8]:
    G = make_arterial_tree(N, uniform_lengths=False)
        
    strr = ''
    
    for n in [3, ]:
    
        G.make_mesh(n)
        G.make_submeshes()
        G.compute_vertex_degrees()
        
        length = interpolate(Constant(1), FunctionSpace(G.mesh, 'CG', 1))
        length = norm(length)**2
        
        we = Constant(1) # 10/length
        wv = Constant(1) # length**0.5*we
        model = infsup_eigvals.MixedHydraulicNetwork_EP(G)

        # model = infsup_eigvals.HydraulicNetwork_EP(G)

        a = model.a_form()
        A = ii_assemble(a)

        a_r = model.b_form_eigval(w_e=we, w_v=wv)
        # a_r = model.b_form_eigval()
        
        B = ii_assemble(a_r)

        # from IPython import embed
        # embed()
        
        W_bcs = model.get_bc()
        A, _ = apply_bc(A, b=None, bcs=W_bcs)
        B, _ = apply_bc(B, b=None, bcs=W_bcs)                

        A, B = map(ii_convert, (A, B))
        
        lam = solve_infsup_eigproblem(A, B, model)
        
        strr += f' {lam[0]:3.2e}  &'
    
    print(f'{G.num_bifurcations:n} &' + strr[0:-2] + '\\\\')
    history.append(f'{G.num_bifurcations:n} &' + strr[0:-2] + '\\\\')

    for row in history: print(row)



