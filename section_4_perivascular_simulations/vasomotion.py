from graphnics import *
import networkx as nx
from xii import *
import ufl
import numpy as np

    
    

def vasomotion_as_ufl(G, ven_moves):
    """
    Create a vasomotion model for a travelling wave
        R_0 = R_0(1+epsilon*sin(k*s-w*t))
    
    The function returns the ufl expressions needed to implement this
    in a network model
    
    Args:
        G (nx.Graph): graph representing the network
    The edges of the graph must have the following attributes:
        'radius1' (float): radius of the inner boundary at rest
        'radius2' (float): radius of the outer boundary
        'Res' (float): resistance
                
    Returns:
        f (ufl.Expression): source term
        Ainv (ufl.Expression): inverse of cross sectional area
        Res (ufl.Expression): resistance
        g (ufl.Expression): force term
        t_ (ufl.Constant): time
        k_  (ufl.Constant): wave number
        w_ (ufl.Constant): wave frequency
        s_ (ufl.Expression): distance from source
        eps_ (ufl.Constant): amplitude of vasomotion
        R1 (ufl.Expression): radius of the vessel
        
        
    """
    
    
    # Fenics constants that parametrize the vasomotion model
    t_ = Constant(0)
    w_ = Constant(1)
    eps_ = Constant(0.1)

    # Radius at rest
    R1_0 = nxgraph_attribute_to_dolfin(G, 'radius1')

    # Vasomotion R_0 = R_0(1+epsilon*sin(k*s-w*t))
    psi = sin(w_*t_)

    R1 = R1_0*(1+eps_*psi)

    # Source term due to arterial wall motion
    def to_color_type(vtype):
        if vtype==3: #artery
            return 1
        else: #capillary or vein
            return 0

    mesh0, foo = G.get_mesh(0)
    DG = FunctionSpace(mesh0, 'DG', 0)
    moves = Function(DG)
    moves.vector()[:] = [to_color_type(G.get_edge_data(u,v)['type']) for u,v in G.edges]
    moves.set_allow_extrapolation(True)
    moves = interpolate(moves, FunctionSpace(G.mesh, 'DG', 0))
    
    if ven_moves:
        print('Venous portion moves')
        f = 2.0*np.pi*R1*ufl.diff(R1, t_) # source term   
    else:
        print('Venous segments fixed')
        f = 2.0*np.pi*R1*ufl.diff(R1, t_)*moves # source term   
        
    R2 = nxgraph_attribute_to_dolfin(G, 'radius2')
    beta = nxgraph_attribute_to_dolfin(G, 'beta')
    
    nu = water_properties['nu']
    Res = nu*annular_resistance(R1/R1_0, beta)/R1_0**4
    
    # Inverse of area
    Ainv = 1/(np.pi*(R2**2-R1**2))   
    
    # Force term in hydraulic model
    g = Constant(0)#(nu*Ainv)*ufl.diff(f, s_)
    
    return f, Ainv, Res, g, t_, w_, eps_, R1



def vasomotion_simulation(G, freq, n_cycles, tsteps_per_cycle, epsilon=0.1, ven_moves=True):
    '''
    Simulate pulsatile flow due to vasomotion in an arterial tree
    
    Args:
        G (nx.DiGraph): network
        freq (float): frequency
        beta (float): aspect ratio of R1 vs R2 (so R2 = beta*R1)
        n_cycles (int): number of cycles to simulate
        tsteps_per_cycle (int): number of time steps per cycle
        epsilon (float): amplitude of vasomotion
        ven_moves (bool): whether the venous/capillaries should pulsate

    Returns:
        experiments (list): list of dicts containing parameters and results
    '''
    
    print('Starting simulation')
    
    
    # time parameters
    time_steps = tsteps_per_cycle*n_cycles
    T = n_cycles/freq

    
    # We make a list "experiments" containing dicts that stores parameters and results
    G.make_mesh(2)
    G.compute_edge_lengths()
    
    # Vasomotion model is expressed via ufl functions depending
    # on constant t_, k_, w_ and s_
    f, area_inv, res, g, t_, w_, eps_, R1 = vasomotion_as_ufl(G, ven_moves=ven_moves)

    # find inlet pos
    pos = nx.get_node_attributes(G, 'pos')
    inlet_pos = [pos[node] for node in G.nodes if G.degree(node)==1][0]
    inlet_pos = np.asarray(inlet_pos)

    for e in G.edges():
        G.edges()[e]['Res'] = res
        G.edges()[e]['Ainv'] = area_inv
    
    print('Solving...')
    t_.assign(0)
    w_.assign(get_w(freq))
    eps_.assign(epsilon)
    
    model = TimeDepHydraulicNetwork(G, p_bc=Constant(0), f=f, Ainv=area_inv, Res=res, g=g)

    qps = time_stepping_stokes(model, t=t_, qp0=None, t_steps = time_steps, T=T, reassemble_lhs=True)
        
    return qps


def get_lamda(k):
    # get wave length from wave number: lamda = 2*pi/k
    lamda = 2*np.pi/k
    return lamda

def get_k(lamda):
    # get wave number from wave length: k = 2*pi/lamda
    k = 2*np.pi/lamda
    return k

def get_freq(w):
    # get frequency from angular frequency: freq = w/(2*pi)
    freq = w/(2*np.pi)
    return freq

def get_w(freq):
    # get angular frequency from frequency: w = 2*pi*freq
    w = 2*np.pi*freq
    return w   

import matplotlib.pyplot as plt
def plot_tree(G, fname=None):
    pos = nx.get_node_attributes(G, 'pos')
    pos2d = [coord[0:2] for coord in list(pos.values())]

    radius = np.asarray(list(nx.get_edge_attributes(G, 'radius').values()))

    fig, ax = plt.subplots(1,1, figsize=(5,5))
    nx.draw_networkx(G, pos2d, width=8*radius, edge_color='gray', 
                    with_labels=False, node_size=0.1, node_color='r', arrowsize=0.1, ax=ax)

    plt.grid('on')
    ax.set_xlabel('x [mm]', fontsize=16)
    ax.set_ylabel('y [mm]', fontsize=16)
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True, labelsize=12)

    if fname:
        plt.savefig(fname, dpi=300, bbox_inches='tight')
    
    return fig, ax


def annular_resistance(ra, b, c=0):
    '''
    Compute annular resistance using (10)
    
    Args: 
        ra (df.function): non-dimensionalized inner radius, ra = R1/R1_0 
        b (df.function): non-dimensionalized outer radius, b = R2/R1_0
        c (df.function): displacement of inner circle from center of annulus
        
    Returns:
        resistance (df.function): resistance of annulus
    '''
    
    # compute "permeability" kappa
    kappa = (np.pi/8)*(1+1.5*(c/(b-ra))**2 ) * (b**4-ra**4- (b**2-ra**2)**2/(ln(b/ra)) )
    
    resistance = kappa**(-1) # inverse of permeability is resistance
    return resistance

def Delta(b, c=0):
    '''
    Compute Delta according to (19), i.e. the resistance change due to arterial wall motion
    
    Args:
        b (float): non-dimensionalized outer radius, b = R2/R1_0
        c (float): displacement of inner circle from center of annulus
    '''
    
    import math
    term1 = 1#1 + 1.5*(c / (b - 1))**2
    term2 = 4 + (b**2 - 1)/math.log(b)*((b**2 - 1)/math.log(b) - 4)
    term3 = 0#3*c**2/(b-1)**3
    term4 = 1#1 + 1.5*c**2/(b - 1)**2
    term5 = b**4 - 1 - (b**2-1)**2/math.log(b)
    
    delta = (term1 * term2 - term3) /( term4*term5)
    
    return delta