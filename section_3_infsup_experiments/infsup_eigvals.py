import networkx as nx
from fenics import *
from xii import *
import sys

sys.path.append("../")
from graphnics import *

from slepc4py import SLEPc
from petsc4py import PETSc

class NetworkPoisson_EP(NetworkPoisson):
    """
    Bilinear forms a and L for the graph Laplacian
            d/ds Res* d/ds p = f
    on graph G
    
    Args:
        G (FenicsGraph): Network domain
        f (df.expr): source term
        p_bc (df.expr): pressure boundary condition
    """

    def b_form_eigval(self, w):
        """
        The right-hand side linear form
        """

        u = TrialFunction(self.V)
        v = TestFunction(self.V)
        
        self.W = [self.V]
        
        G = self.G

        b = Constant(1/w**2)*inner(u, v)*dx + inner(G.dds(u), G.dds(v))*dx

        return b


class HydraulicNetwork_EP(HydraulicNetwork):
    """
    Bilinear forms a and L for the primal mixed hydraulic equations
            Res*q + d/ds p = g
            d/ds q = f
    on graph G, with bifurcation conditions q_in = q_out and continuous
    normal stress

    Args:
        G (FenicsGraph): Network domain
        Res (dict): dictionary with edge->resistance
        f (df.expr): source term
        p_bc (df.expr): pressure boundary condition
    """

    def b_form_eigval(self, w=None):
        """
        The right-hand side linear form
        """

        a = [[0 for i in range(0, len(self.qp))] for j in range(0, len(self.qp))]
        q, p = self.qp
        v, phi = self.vphi
        

        a[0][0] = q*v*dx
        # a[1][1] = Constant(w**2)*p*phi*dx + self.G.dds(p)*self.G.dds(phi)*dx
        a[1][1] = self.G.dds(p)*self.G.dds(phi)*dx
        
        return a


class MixedHydraulicNetwork_EP(MixedHydraulicNetwork):
    """
    Bilinear forms a and L for the dual mixed hydraulic equations
            Res*q + d/ds p = g
            d/ds q = f
    on graph G, with bifurcation conditions q_in = q_out and continuous
    normal stress

    Args:
        G (FenicsGraph): Network domain
        Res (dict): dictionary with edge->resistance
        f (df.expr): source term
        p_bc (df.expr): pressure boundary condition
    """

    def b_form_eigval(self, w_e=1, w_v=None):
        """
        The right-hand side linear form
        """
        
        a = self.init_a_form()
        qp, vphi = self.qp, self.vphi
        G = self.G

        length = interpolate(Constant(1), FunctionSpace(G.mesh, 'CG', 1))
        length = norm(length)**2
        
        # split out the components
        n_edges = G.num_edges
        
        qs, vs = qp[:n_edges], vphi[:n_edges]

        M = len(G.nodes())
                
        # flux norm
        for i, e in enumerate(G.edges):
            dx_edge = Measure("dx", domain=G.edges[e]["submesh"])
            ds_edge = Measure("ds", domain=G.edges[e]["submesh"], subdomain_data=G.edges[e]["vf"])
            
            length_e = G.edges[e]["length"]
            alpha_inv = Constant(length_e/M)

            a[i][i] = (
                + qs[i] * vs[i] * dx_edge
                + Constant(length**2)*G.dds_i(qs[i], i) * G.dds_i(vs[i], i) * dx_edge
                + Constant(length**2)*alpha_inv*(qs[i]*vs[i]*ds_edge(BIF_IN))
                + Constant(length**2)*alpha_inv*(qs[i]*vs[i]*ds_edge(BIF_OUT)))

            
        # pressure norm
        a[n_edges][n_edges] = Constant(1/length**2)*qp[n_edges] * vphi[n_edges] * dx
    
        for i in range(0, G.num_bifurcations):
            
            mesh_ = qp[n_edges + 1 + i].ufl_domain().ufl_cargo()
            vol = sum(c.volume() for c in cells(mesh_))

            # iterate over edges connected to vertex
            length_i = 0
            for e in G.edges_in(i):
                length_i += G.edges[e]["length"]
            for e in G.edges_out(i):
                length_i += G.edges[e]["length"]

            a[n_edges + 1 + i][n_edges + 1 + i] += ( Constant(1/vol)*
                Constant(M/length_i)*alpha*qp[n_edges + 1 + i] * vphi[n_edges + 1 + i] * dx
            )    
        
        return a


def solve_infsup_eigproblem(A, B, model):
    """
    Solve the eigenproblem
        Ax = lamda B x
    associated with the infsup condition for the bilinear form of a mixed problem
    """
    
    AA = as_backend_type(ii_convert(A)).mat()
    #AAA = PETSc.Mat().createDense(AA.array().shape)
    #AAA.setUp()
    #AAA[:, :] = AA.array()
    #AAA.assemble()

    BB = as_backend_type(ii_convert(B)).mat()
    #BBB = PETSc.Mat().createDense(BB.array().shape)
    #BBB.setUp()
    # BBB[:, :] = BB.array()
    # BBB.assemble()

    E = SLEPc.EPS()
    E.create()

    E.setProblemType(SLEPc.EPS.ProblemType.GHEP)
    E.setOperators(AA, BB)

    opts = PETSc.Options()    

    opts.setValue('eps_monitor', None)
    opts.setValue('eps_view', None)

    if AA.size[0] < 10_000:
        opts.setValue('eps_type', 'lapack')
        # opts.setValue('eps_nev', 'all')        
    else:
        E.setWhichEigenpairs(E.Which.SMALLEST_MAGNITUDE)        

        opts.setValue('eps_type', 'krylovschur')
        
        opts.setValue('eps_tol', 1E-4)
        opts.setValue('eps_nev', 2)
        opts.setValue('eps_max_it', 10_000)
    
    E.setFromOptions()
    E.solve()
    nconv = E.getConverged()

    u_r, u_im = AA.createVecs()    
    u_lams, eigvals = [], []
    for i in range(0, nconv):

        eigvals.append(np.abs(np.real(E.getEigenvalue(i))))

        E.getEigenpair(i, u_r, u_im)
        u_lams.append(u_r)

        if u_r.norm(2) < 1E-8:
            print('>>>>>>', E.getEigenvalue(i), u_r.norm(2), u_im.norm(2))

    # Remove eigenvalue corresponding to zero eigenmode
    u_lam_norms = [u.norm(2) for u in u_lams]
    lams = np.sort(eigvals)
    non_zero = np.where(np.asarray(u_lam_norms) > 1e-8)[0]

    print('Converged modes', nconv, len(non_zero))
    
    lams = lams[non_zero]
    print(min(np.abs(lams)), max(np.abs(lams)))
    lam = lams[0:2]
 
    return lam


