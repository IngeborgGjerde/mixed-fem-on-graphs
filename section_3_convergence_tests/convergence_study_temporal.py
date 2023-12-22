from graphnics import *
import ufl
from fenics import *
set_log_level(50)

def analytic_solution_spatial(G):
    '''
    Make analytical solution on single vessel or Y bifurcation
    The solution is independent of time
    '''
    
    
    t = Constant(0)
    pi = 3.141592653589793
    
    s = DistFromSource(G, 0, degree=1)
    s = interpolate(s, FunctionSpace(G.mesh, 'CG', 1),)
    
    tau = G.tangent

    if len(G.edges==1):
        # Manufacture q so that jump(q)=0 at bifurcation 
        # we split q into two parts, q1 and q2
        # q1 is edgewise constant 1 on the parent and 0.5 on the daughters
        # q2 is a sine wave so that q2 = 0 at bifurcation
        
        q1 = 1
        q2 = sin(4*pi*s) 
        q=q1+q2
        
        # then p=p1+p2, where d/ds p1 = q1 and d/ds p2 = q2
        # -d/ds p1=q1 -> p1 = -q1*s + p1(0), we set p1(0)=0 for parent and xx for daughters
        # -d/ds p2=q2 -> p2 = (1/4pi)*cos(4*pi*s) + p2(0), we set p2(0)=0
        p1 = - q1*s
        p2 = (1/(4*pi))*cos(4*pi*s)
        p = p1+p2
        
        f = dot(tau, ufl.grad(q))
        g = ufl.diff(q, t) + q + dot(tau, ufl.grad(p)) # g = q_t + q + p_x
        
    
    elif len(G.edges==3):
        # Manufacture q so that jump(q)=0 at bifurcation 
        # we split q into two parts, q1 and q2
        # q1 is edgewise constant 1 on the parent and 0.5 on the daughters
        # q2 is a sine wave so that q2 = 0 at bifurcation
        
        parent = conditional(tau[0]**2<1e-3, 1, 0)
        daughters = 1-parent
        q1 = parent + 0.5*daughters
        q2 = sin(4*pi*s) 
        q=q1+q2
        
        # then p=p1+p2, where d/ds p1 = q1 and d/ds p2 = q2
        # -d/ds p1=q1 -> p1 = -q1*s + p1(0), we set p1(0)=0 for parent and xx for daughters
        # -d/ds p2=q2 -> p2 = (1/4pi)*cos(4*pi*s) + p2(0), we set p2(0)=0
        p1 = - q1*s - 0.25*daughters
        p2 = (1/(4*pi))*cos(4*pi*s)
        p = p1+p2
        
        f = dot(tau, ufl.grad(q))
        g = ufl.diff(q, t) + q + dot(tau, ufl.grad(p)) # g = q_t + q + p_x
        

    return p, q, f, g, t


def analytic_solution_temporal(G):
    '''
    Make analytical solution on a Y bifurcation
    '''
    
    t = Constant(0)
    pi = 3.141592653589793
    
    s = DistFromSource(G, 0, degree=1)
    s = interpolate(s, FunctionSpace(G.mesh, 'CG', 1),)
    
    tau = G.global_tangent
    
    # Manufacture q so that jump(q)=0 at bifurcation 
    # we split q into two parts, q1 and q2
    # q1 is edgewise constant 1 on the parent and 0.5 on the daughters
    # q2 is a sine wave so that q2 = 0 at bifurcation
    
    parent = conditional(tau[0]**2<1e-3, 1, 0)
    daughters = 1-parent
    q1 = parent + 0.5*daughters
    q=q1*sin(2*pi*t)
       
    # then p=p1+p2, where d/ds p1 = q1 and d/ds p2 = q2
    # -d/ds p1=q1 -> p1 = -q1*s + p1(0), we set p1(0)=0 for parent and xx for daughters
    # -d/ds p2=q2 -> p2 = (1/4pi)*cos(4*pi*s) + p2(0), we set p2(0)=0
    p1 = - q1*s - 0.25*daughters
    p = p1*sin(2*pi*t)
    
    f = Constant(0)#dot(tau, ufl.grad(q))
    g = ufl.diff(q, t) + q + dot(tau, ufl.grad(p))
    
    return p, q, f, g, t
    

def rate(e_prev, e, h, h_prev):
    # convergence rate for error e
    if e_prev is None: return 0
    return -(np.log(e_prev)-np.log(e))/(np.log(h)-np.log(h_prev))
    
    
def test_primal_spatial(degree, G):
    
    print('\n\n** Primal convergence rates for spatial discretization **')
    

    print(f'$h$     & $\\norm{{q}}{{L^2(\\Lambda)}}$ & '
                    f'$\\norm{{p}}{{L^2(\\Lambda)}}$ &'
                    f'$\\norm{{p}}{{H^1(\\Lambda)}}$ \\ \\hline')
    
    err_q_L2_prev, err_p_L2_prev, err_p_H1_prev, h_prev = None, None, None, None
    
    
    for N in [4, 5, 6, 7]:
        G.make_mesh(N)
        pa, qa, f, g, t = analytic_solution_spatial(G)

        model = HydraulicNetwork(G, f=f, p_bc=pa, g=g, degree=degree)
        qp = model.solve()        
        
        qa = project(qa, FunctionSpace(G.mesh, 'DG', 2))
        pa = project(pa, FunctionSpace(G.mesh, 'CG', 3))
        
        e_q_L2 = errornorm(qa, qp[0], 'L2')
        e_p_L2 = errornorm(pa, qp[1], 'L2')
        e_p_H1 = errornorm(pa, qp[1], 'H1')
        
        h = G.mesh.hmin()
        rate_q_L2 = rate(err_q_L2_prev, e_q_L2, h, h_prev)
        rate_p_H1 = rate(err_p_H1_prev, e_p_H1, h, h_prev)
        rate_p_L2 = rate(err_p_L2_prev, e_p_L2, h, h_prev)
        
        err_q_L2_prev, err_p_L2_prev, err_p_H1_prev, h_prev = e_q_L2, e_p_L2, e_p_H1, h
        
        print(f'{h:<3.1e} &        {e_q_L2:<9.1e} ({rate_q_L2:1.1f}) &        {e_p_L2:<9.1e} ({rate_p_L2:1.1f}) &       {e_p_H1:<9.1e} ({rate_p_H1:1.1f}) \\\\')



def test_primal_temporal(lump_errors=True):
    
    print('\n\n** Primal convergence rates  for temporal discretization **')
    G = Y_bifurcation()
    
    
    err_q_L2_prev, err_p_L2_prev, err_p_H1_prev, dt_prev, tot_err_prev = None, None, None, None, None
    
    G.make_mesh(9)
    T = 2*np.pi
    
    for disc_scheme in ['IE', 'CN']:
        print(f'\n - Scheme: {disc_scheme}')
        
        if lump_errors:
            print(f'$\\Delta t$    & $\\norm{{(q,p)}}{{W_h}}$ \\ \hline')
        else:
            print(f'$\\Delta t$    & $\\norm{{q}}{{L^2(\\Lambda)}}$ &' 
                                +  f'$\\norm{{p}}{{L^2(\\Lambda)}}$ &'
                                   f'$\\norm{{p}}{{H^1(\\Lambda)}}$ \\\\ \hline')
        
        for t_steps in [32, 64, 128, 256]:
            dt = T/t_steps
                            
            pa, qa, f, g, t = analytic_solution_temporal(G)
            
            t.assign(0)
            qpa = as_vector([qa, pa])
            
            model = TimeDepHydraulicNetwork(G, f=f, p_bc=pa, g=g)
            qps = time_stepping_stokes(model, t=t, t_steps=t_steps, T=T, qp0=qpa, t_step_scheme=disc_scheme)
            qp = qps[-1]
            
            
            t.assign(T)
            qa = project(qa, FunctionSpace(G.mesh, 'DG', 1))
            pa = project(pa, FunctionSpace(G.mesh, 'CG', 2))
            
            e_q_L2 = errornorm(qa, qp[0], 'L2')
            e_p_L2 = errornorm(pa, qp[1], 'L2')
            e_p_H1 = errornorm(pa, qp[1], 'H1')
            tot_err = np.sqrt(e_q_L2**2 + e_p_L2**2 + e_p_H1**2)
            
            rate_q_L2 = rate(err_q_L2_prev, e_q_L2, dt, dt_prev)
            rate_p_H1 = rate(err_p_H1_prev, e_p_H1, dt, dt_prev)
            rate_p_L2 = rate(err_p_L2_prev, e_p_L2, dt, dt_prev)
            rate_tot = rate(tot_err_prev, tot_err, dt, dt_prev)
            
            err_q_L2_prev, err_p_L2_prev, err_p_H1_prev, dt_prev, tot_err_prev = e_q_L2, e_p_L2, e_p_H1, dt, tot_err
            

            if lump_errors:
                print(f'{dt:<3.1e} &        {tot_err:<9.1e} ({rate_tot:1.1f}) \\\\')
            else:
                print(f'{dt:<3.1e} &        {e_q_L2:<9.1e} ({rate_q_L2:1.1f}) &        {e_p_L2:<9.1e} ({rate_p_L2:1.1f}) &       {e_p_H1:<9.1e} ({rate_p_H1:1.1f}) \\\\')



def test_dual_spatial(degree, G):
    
    print('\n** Dual mixed convergence rates  for spatial discretization **')
    
    err_q_L2_prev, err_p_L2_prev, err_q_Hdiv_prev, h_prev = None, None, None, None
    
    print(f'$h$    & $\\norm{{q}}{{L^2(\\Lambda)}}$ &' + 
                    f'$\\norm{{q}}{{H(\\mathrmdiv;\\Lambda)}}$ &' +
                    f'$\\norm{{p}}{{L^2(\\Lambda)}}$ \\\\ \\hline')
    
    for N in [4, 5, 6, 7]:
        G.make_mesh(N)
        G.make_submeshes()
        pa, qa, f, g, t = analytic_solution_spatial(G)
        bp = G.nodes()[1]['pos']
        pa_i = project(pa, FunctionSpace(G.mesh, 'CG', 1))
        pa_b = pa_i(bp[0], bp[1], bp[2])

        model = MixedHydraulicNetwork(G, f=f, p_bc=pa, degree=degree)
        qp, qd1, qd2, p, p_b = model.solve()    
        
        qa = project(qa, FunctionSpace(G.mesh, 'DG', 3))
        pa = project(pa, FunctionSpace(G.mesh, 'CG', 4))
        
        [func.set_allow_extrapolation(True) for func in [qp, qd1, qd2]]
        q = GlobalCrossectionFlux(G, [qp, qd1, qd2])
        q = interpolate(q, FunctionSpace(G.mesh, 'DG', 3))
        
        e_p_L2 = errornorm(pa, p, 'L2') + np.abs(pa_b-p_b(bp[0], bp[1], bp[2]))
        e_q_L2 = errornorm(qa, q, 'L2')
        
        h = G.mesh.hmin()
        
        edges = list(G.edges())
        e_q_H1 = 0
        for i, qc in enumerate([qp, qd1, qd2]):
            submesh = G.edges()[edges[i]]['submesh']
            qa_i = Restriction(qa, submesh)
            qa_i.set_allow_extrapolation(True)
            qa_i = interpolate(qa_i, FunctionSpace(submesh, 'CG', degree))
            e_q_H1 += np.sum(h*(qa_i.vector().get_local() - qc.vector().get_local())**2)

        jump = qp(bp[0], bp[1], bp[2])-qd1(bp[0], bp[1], bp[2])-qd2(bp[0], bp[1], bp[2])
        e_q_Hdiv = e_q_H1 + np.abs(jump)
            
        
        rate_p_L2 = rate(err_p_L2_prev, e_p_L2, h, h_prev)
        rate_q_L2 = rate(err_q_L2_prev, e_q_L2, h, h_prev)
        rate_q_H1 = rate(err_q_Hdiv_prev, e_q_Hdiv, h, h_prev)
        
        err_p_L2_prev, err_q_L2_prev, err_q_Hdiv_prev, h_prev = e_p_L2, e_q_L2, e_q_Hdiv, h
        
        print(f'{h:<3.1e} &        {e_q_L2:<9.1e} ({rate_q_L2:1.1f}) &        {e_q_Hdiv:<9.1e} ({rate_q_H1:1.1f}) &       {e_p_L2:<9.1e} ({rate_p_L2:1.1f}) \\\\')

    for e in G.edges():
        G.edges()[e]['radius'] = 1
    
    q.rename('q', 'q')
    p.rename('p', 'p')
    File('plots/q.pvd') << q
    File('plots/p.pvd') << p
    
    pa.rename('pa', 'p')
    qa.rename('qa', 'q')
    TubeFile(G, 'plots/qa.pvd') << qa
    TubeFile(G, 'plots/pa.pvd') << pa  
    
    
    
    

def test_dual_temporal(lump_errors = True):
    
    print('\n** Dual mixed convergence rates  for temporal discretization **')
    G = Y_bifurcation(dim=3)
    
    err_q_L2_prev, err_p_L2_prev, err_q_H1_prev, dt_prev, tot_err_prev = None, None, None, None, None
    
    T = 2*np.pi
    
    for disc_scheme in ['IE', 'CN']:
        
        print(f'\n - Scheme: {disc_scheme}')
        
        if lump_errors:
            print(f'$\Delta t$    & $\\norm{{(p,q)}}{{W_h}}$ \\\\ \\hline')
        else:
            print(f'$\Delta t$    & $\\norm{{q}}{{L^2(\\Lambda)}}$ &' +
                                f'$\\norm{{q}}{{H(\\mathrmdiv;\\Lambda)}}$ &'
                                f'$\\norm{{p}}{{L^2(\\Lambda)}}$ \\\\ \\hline')
        
        for t_steps in [32, 64, 128, 256]:
            G.make_mesh(9)
            dt = T/t_steps
            
            pa, qa, f, g, t = analytic_solution_temporal(G)
        
            t.assign(0)
            qpa = as_vector([qa, qa, qa, pa, Constant(0)]) #last one maybe not constant?
        
            model = TimeDepMixedHydraulicNetwork(G, f=f, p_bc=pa, g=g)
            qps = time_stepping_stokes(model, t=t, t_steps=t_steps, T=T, qp0=qpa, t_step_scheme=disc_scheme)
            qp, qd1, qd2, p, p_b = qps[-1]
            
            t.assign(T)
            qa = project(qa, FunctionSpace(G.mesh, 'DG', 1))
            pa = project(pa, FunctionSpace(G.mesh, 'CG', 2))
            
            
            q = GlobalCrossectionFlux(G, [qp, qd1, qd2])
            q = interpolate(q, FunctionSpace(G.mesh, 'DG', 2))
            
            e_p_L2 = errornorm(pa, p, 'L2')
            e_q_L2 = errornorm(qa, q, 'L2')
            
            h = G.mesh.hmin()
            
            edges = list(G.edges())
            e_q_H1 = 0
            for i, qc in enumerate([qp, qd1, qd2]):
                submesh = G.edges()[edges[i]]['submesh']
                qa_i = Restriction(qa, submesh)
                qa_i = interpolate(qa_i, FunctionSpace(submesh, 'CG', 1))
                e_q_H1 += np.sum(h*(qa_i.vector().get_local() - qc.vector().get_local())**2)
                
            tot_err = np.sqrt(e_q_L2**2 + e_q_H1**2 + e_p_L2**2)
            
            rate_p_L2 = rate(err_p_L2_prev, e_p_L2, dt, dt_prev)
            rate_q_L2 = rate(err_q_L2_prev, e_q_L2, dt, dt_prev)
            rate_q_H1 = rate(err_q_H1_prev, e_q_H1, dt, dt_prev)
            rate_tot_err = rate(tot_err_prev, tot_err, dt, dt_prev)
            
            err_p_L2_prev, err_q_L2_prev, err_q_H1_prev, tot_err_prev, dt_prev = e_p_L2, e_q_L2, e_q_H1,  tot_err, dt
            
            
            
            if lump_errors:
                print(f'{dt:<3.1e} &   {tot_err:<9.1} ({rate_tot_err:1.1f}) \\\\')
            else:
                print(f'{dt:<3.1e} &        {e_q_L2:<9.1e} ({rate_q_L2:1.1f}) &        {e_q_H1:<9.1e} ({rate_q_H1:1.1f}) &       {e_p_L2:<9.1e} ({rate_p_L2:1.1f}) \\\\')

    for e in G.edges():
        G.edges()[e]['radius'] = 1
    
    q.rename('q', 'q')
    p.rename('p', 'p')
    File('plots/q.pvd') << q
    File('plots/p.pvd') << p
    
    pa.rename('pa', 'p')
    qa.rename('qa', 'q')
    TubeFile(G, 'plots/qa.pvd') << qa
    TubeFile(G, 'plots/pa.pvd') << pa  

      
if __name__ == '__main__':

    for degree in [1, 2, 3, 4]:    
        print('\n\nDegree = ', degree)
        test_primal_spatial(degree)
       

    for degree in [1, 2, 3, 4]:    
        print('\n\nDegree = ', degree)
        test_dual_spatial(degree)
    
    #test_primal_temporal()
    #test_dual_temporal()