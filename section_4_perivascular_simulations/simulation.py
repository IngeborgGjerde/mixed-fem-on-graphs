

from graphnics import *
from fenics import *
from xii import *
import sys
sys.path.append('../graphnics/data')
from generate_arterial_tree import *
import vasomotion as vm
import imp
import seaborn as sns
sns.set(style="whitegrid")
import matplotlib.pyplot as plt

def simulation(G, time_steps, desc, N, ven_moves):

    for e in G.edges():
        G.edges()[e]['radius1'] = G.edges()[e]['radius']
        G.edges()[e]['radius2'] = 3*G.edges()[e]['radius']
        G.edges()[e]['beta'] = 3
        G.edges()[e]['fixed'] = False

    # find minimal radius1
    radii1 = nx.get_edge_attributes(G, 'radius1')
    min_radius1 = np.min(list(radii1.values()))
    print('min_radius1', min_radius1)


    eps = 0.1
    freq = 1
    tsteps_per_cycle = time_steps
    n_cycles = time_steps

    qps = vm.vasomotion_simulation(G, freq=freq, n_cycles=n_cycles, tsteps_per_cycle=tsteps_per_cycle, epsilon=eps, ven_moves=ven_moves)    

    print(f' $N$     &   $\langle Q \\rangle$    \\\\ \n')
    
    # Compute net flow from simulation
    yss = []
    tss = []
        
    nodes = [list(G.nodes())[ix] for ix in [0]]
    pos = [G.nodes()[n]['pos'] for n in nodes] 
    time_steps = len(qps)
    T = n_cycles/freq # total simulation time
    dt = T/time_steps
    
    for i, n in enumerate(nodes):
        outflows = [sol[0](pos[i]) for sol in qps]
        
        Qhs = np.cumsum(outflows[tsteps_per_cycle:])*dt
        ts = np.linspace(0, T, time_steps)[tsteps_per_cycle:]
        ts -= ts[0]
        
        Qhs_last_cycle = Qhs[-tsteps_per_cycle:]
        
        T_cycle = ts[-1]-ts[-tsteps_per_cycle]
        Qh_avg_tilde = (Qhs[-1]-Qhs[-tsteps_per_cycle-1])/T_cycle

        amplitude = np.max(Qhs_last_cycle) - np.min(Qhs_last_cycle)  - Qh_avg_tilde
        
        directionality = Qh_avg_tilde/amplitude*100

        tss.append(ts)
        yss.append(Qhs)
        
        print(f'{N}    &  {Qh_avg_tilde:<1.3f}({directionality:1.1f})\\\\' )
    
    # Save ts and ys to file
    text_file = open("data/flow_"+desc+".txt", "w")
    for i in range(0,1):
        text_file.write(str(tss[i])+', ' + str(yss[i])+'\n')
        

    fig, ax = plt.subplots(1, 1, figsize=(5,5)) 
    sns.set(font_scale=2.2, style='whitegrid')

    markers = ['r']
    for i in range(0,1):
        
        ax.plot(tss[i], yss[i],  markers[i], linewidth=3)

        ax.set_ylabel('$Q$ [$\mu$L]', fontsize=24)
        ax.set_xlabel('$t$ [s]', fontsize=24)
    plt.savefig('sim.png', bbox_inches='tight', dpi=300)


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--Ns', type=int, default=[2], nargs = '+')
    parser.add_argument('--time_steps', type=int, default=25)
    parser.add_argument('--uniform_radius', type=bool, default=False)
    parser.add_argument('--connectome', type=int, default=0)
    parser.add_argument('--gamma', type=float, default=0.8)
    parser.add_argument('--ven_moves', type=int, default=0)
    parser.add_argument('--scaling', nargs = '+', type=float, default=[1.0])
    
    

    args = parser.parse_args()

    desc0 = 'uniform' if args.uniform_radius else 'nonuniform'
    desc0 += '_connectome' if args.connectome else '_tree'
    desc0 += f'_{args.time_steps}_time_steps'

    from generate_arterial_tree import *
        
    directions = [1,1]*100
    radius0 = 1
    for scal in args.scaling:
        print('Scaling ', scal)
        for N in args.Ns:

            if not args.connectome:
                G = make_arterial_tree(N=N, radius0=radius0, L0=10, directions=[1,1]*100, gam=args.gamma, remove_overlapping_edges=False)
            else:
                G = make_arterial_venous_connectome(N=N, radius0=1, L0=10, directions=[1,1]*100, gam=args.gamma, remove_overlapping_edges=False, scaling=scal)

            if args.uniform_radius:
                for e in G.edges():
                    G.edges()[e]['radius'] = radius0

            desc = desc0+f'_gens{N}'

            simulation(G, args.time_steps, desc, N, args.ven_moves)