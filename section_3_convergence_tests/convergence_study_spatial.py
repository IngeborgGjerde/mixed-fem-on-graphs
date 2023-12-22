from graphnics import *
import ufl
from fenics import *
set_log_level(50)

def analytic_solution_spatial(G):
    '''
    Make analytical solution on single vessel or Y bifurcation
    The solution is independent of time
    '''
    
    pi = 3.141592653589793
    
    s = interpolate(DistFromSource(G, 0, degree=1), FunctionSpace(G.mesh, 'CG', 1))
    
    tau = G.tangent

    if len(G.edges)==1:
        # Manufacture q so that jump(q)=0 at bifurcation 
        # we split q into two parts, q1 and q2
        # q1 is edgewise constant 1 on the parent and 0.5 on the daughters
        # q2 is a sine wave so that q2 = 0 at bifurcation
        
        q1 = 0
        q2 = cos(pi*s) + sin(2*pi*s)
        q=q1+q2
        
        p = sin(pi*s) + cos(2*pi*s)
        
        f = dot(tau, ufl.grad(q))
        g = q + dot(tau, ufl.grad(p)) # g = q_t + q + p_x
        
    
    elif len(G.edges)==3:
        # Manufacture q so that jump(q)=0 at bifurcation 
        # we split q into two parts, q1 and q2
        # q1 is edgewise constant 1 on the parent and 0.5 on the daughters
        # q2 is a sine wave so that q2 = 0 at bifurcation
        
        parent = conditional(tau[0]**2<1e-3, 1, 0)
        daughters = 1-parent
        q1 = parent + 0.5*daughters
        q2 = cos(pi*s) + sin(2*pi*s)
        q=q1+q2
        
        p = sin(pi*s) + cos(2*pi*s)
        
        f = dot(tau, ufl.grad(q))
        g = q + dot(tau, ufl.grad(p)) # g = q_t + q + p_x
        

    return p, q, f, g

    

def rate(e_prev, e, h, h_prev):
    # convergence rate for error e
    if e_prev is None: return 0
    return -(np.log(e_prev)-np.log(e))/(np.log(h)-np.log(h_prev))
    
    
def test_primal_spatial(degree, G):
    
    
    print(f'$h$     & $\\norm{{q}}{{L^2(\\Lambda)}}$ & '
                    + f'$\\norm{{p}}{{H^1(\\Lambda)}}$ \\ \\hline')
    
    err_q_L2_prev, err_p_L2_prev, err_p_H1_prev, h_prev = None, None, None, None
    
    
    for N in [4, 5, 6, 7]:
        G.make_mesh(N)
        pa, qa, f, g = analytic_solution_spatial(G)

        model = HydraulicNetwork(G, f=f, p_bc=pa, g=g, degree=degree)
        qp = model.solve()        
        
        qa = project(qa, FunctionSpace(G.mesh, 'DG', 5))
        pa = project(pa, FunctionSpace(G.mesh, 'CG', 6))
        
        e_q_L2 = errornorm(qa, qp[0], 'L2')
        e_p_L2 = errornorm(pa, qp[1], 'L2')
        e_p_H1 = errornorm(pa, qp[1], 'H1')
        
        h = G.mesh.hmin()
        rate_q_L2 = rate(err_q_L2_prev, e_q_L2, h, h_prev)
        rate_p_H1 = rate(err_p_H1_prev, e_p_H1, h, h_prev)
        rate_p_L2 = rate(err_p_L2_prev, e_p_L2, h, h_prev)
        
        err_q_L2_prev, err_p_L2_prev, err_p_H1_prev, h_prev = e_q_L2, e_p_L2, e_p_H1, h
        
        print(f'{h:<3.1e} &    {e_p_L2:<9.1e} ({rate_p_L2:1.1f}) &    {e_q_L2:<9.1e} ({rate_q_L2:1.1f}) &               {e_p_H1:<9.1e} ({rate_p_H1:1.1f}) \\\\')


def test_dual_spatial(degree, G):
    
    
    err_p_L2_prev, err_q_Hdiv_prev, h_prev =  None, None, None
    
    print(f'$h$    & $\\norm{{q}}{{H(\\mathrmdiv;\\Lambda)}}$ &' +
                    f'$\\norm{{p}}{{L^2(\\Lambda)}}$ \\\\ \\hline')
    
    for N in [4, 5, 6, 7]:
        G.make_mesh(N)
        G.make_submeshes()
        pa, qa, f, g = analytic_solution_spatial(G)
        bp = G.nodes()[1]['pos']
        pa_i = project(pa, FunctionSpace(G.mesh, 'CG', 1))
        
        model = MixedHydraulicNetwork(G, f=f, p_bc=pa, degree=degree, g=g)
        sol = model.solve()

        
        qa_i = project(qa, FunctionSpace(G.mesh, 'DG', degree+1))
        pa = project(pa, FunctionSpace(G.mesh, 'DG', degree+1))

        if len(G.edges)==1:
            q, p = sol
            qs = [q]
        elif len(G.edges)==3:
            qp, qd1, qd2, p, p_b = sol 
            [func.set_allow_extrapolation(True) for func in [qp, qd1, qd2]]
            q = GlobalCrossectionFlux(G, [qp, qd1, qd2])
            q = interpolate(q, FunctionSpace(G.mesh, 'CG', degree+1))
            qs = [qp, qd1, qd2]
        
        e_p_L2 = errornorm(pa, p, 'L2') 
        if len(G.edges)==3:
            pa_b = pa_i(bp[0], bp[1], bp[2])
            e_p_L2 += np.abs(pa_b-p_b(bp[0], bp[1], bp[2]))

        
        h = G.mesh.hmin()
        
        edges = list(G.edges())
        e_q_H1 = 0
        for i, qc in enumerate(qs):
            submesh = G.edges()[edges[i]]['submesh']
            qa_c_i = project(qa, FunctionSpace(submesh, 'CG', degree+1))
            e_q_H1 += errornorm(qa_c_i, qc, 'H1')

        e_q_Hdiv = e_q_H1
        if len(G.edges)==3:
            jump = qp(bp[0], bp[1], bp[2])-qd1(bp[0], bp[1], bp[2])-qd2(bp[0], bp[1], bp[2])
            e_q_Hdiv += np.abs(jump)
        
        rate_p_L2 = rate(err_p_L2_prev, e_p_L2, h, h_prev)
        rate_q_H1 = rate(err_q_Hdiv_prev, e_q_Hdiv, h, h_prev)
        
        err_p_L2_prev, err_q_Hdiv_prev, h_prev = e_p_L2, e_q_Hdiv, h
        
        print(f'{h:<3.1e} &     {e_q_Hdiv:<9.1e} ({rate_q_H1:1.1f}) &       {e_p_L2:<9.1e} ({rate_p_L2:1.1f}) \\\\')

    for e in G.edges():
        G.edges()[e]['radius'] = 1
    
    #q.rename('q', 'q')
    #p.rename('p', 'p')
    #File('plots/q.pvd') << q
    #File('plots/p.pvd') << p
    
    #pa.rename('pa', 'p')
    #qa.rename('qa', 'q')
    #TubeFile(G, 'plots/qa.pvd') << qa
    #TubeFile(G, 'plots/pa.pvd') << pa  
    
    
    
      
if __name__ == '__main__':

    G = line_graph(n=2, dx=0.5, dim=3)
    #G = Y_bifurcation(dim=3)

    print('\n\n** Primal convergence rates for spatial discretization **')

    for degree in [1, 2, 3, 4]:    
        print('\n\nDegree = ', degree)
        test_primal_spatial(degree, G)
       
    print('\n\n** Dual convergence rates for spatial discretization **')
    
    for degree in [1, 2, 3, 4]:    
        print('\n\nDegree =s ', degree)
        test_dual_spatial(degree, G)
    
    