from graphnics import *
import networkx as nx
from xii import *
import seaborn as sns
import numpy as np
sns.set()
import sys
sys.path.append('../graphnics/data/')
import generate_arterial_tree as gat



# The arterial tree can be built randomly, with the direction (sign) of the branch being random.
# Keeping reproducability in mind we here override this behaviour
signs = np.tile([-1,1], 10).tolist()

L = 1 
radius0 = 0.1*L

G = gat.make_arterial_tree(6, directions = signs, gam=0.9, L0=L, radius0=radius0)
G.make_mesh(2)

class InitRadius(UserExpression):
    def __init__(self, G, **kwargs):
        self.G = G
        super().__init__(**kwargs)
    def eval_cell(self, values, x, cell):
        edge = list(self.G.edges())[self.G.mf[cell.index]]        
        values[0] = self.G.edges()[edge]['radius']
        

radius = InitRadius(G, degree=1)
radius_i = interpolate(radius, FunctionSpace(G.global_mesh, 'CG', 1))
s = DistFromSource(G, 0)
s = interpolate(s, FunctionSpace(G.global_mesh, 'CG', 1))

p_drop = -1
p = 1+0.25*s*p_drop

nu = water_properties['nu']
Qs =[15.5, 0.191] 
As = [14.5, 5.7]

qfile = File('plots/q.pvd')
pfile = TubeFile(G, 'plots/p.pvd')

for ix, QA in enumerate(zip(Qs, As)):

    Qtilde, Atilde = QA
    print(f'Qtilde = {Qtilde}, Atilde = {Atilde}')
    
    Res = Constant(nu/Qtilde)*(1/radius_i**4)
        
    p_i = project(p, FunctionSpace(G.global_mesh, 'CG', 1))
    model = HydraulicNetwork(G, p_bc = p_i, Res=Res)
    q, p = model.solve()

    q = GlobalFlux(G, [q])
    
    q = project(q, VectorFunctionSpace(G.global_mesh, 'DG', 0, 3))   
    q.rename('q', 'q')
    p.rename('p', 'p')

    qfile << (q, float(ix))
    pfile << (p, float(ix))
    
    q_max = np.max(q.vector().get_local())
    print(f'q_max = {q_max} at {np.where(q.vector().get_local() == q_max)[0][0]}')