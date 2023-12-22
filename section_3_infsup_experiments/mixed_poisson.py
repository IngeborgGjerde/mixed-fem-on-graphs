from dolfin import *

mesh = UnitSquareMesh(32, 32)

Velm = VectorElement('Lagrange', triangle, 2)
Qelm = FiniteElement('Lagrange', triangle, 1)
Welm = MixedElement([Velm, Qelm])

W = FunctionSpace(mesh, Welm)
u, p = TrialFunctions(W)
v, q = TestFunctions(W)

# div(u) = f
# where u = -grad(p)
a = inner(u, v)*dx - inner(p, div(v))*dx - inner(q, div(u))*dx
L = -inner(Constant(1), q)*dx

# Set some flux bcs
bdry = MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
CompiledSubDomain('near(x[0], 0)').mark(bdry, 1)

bcs = DirichletBC(W.sub(0), Constant((0, 0)), bdry, 1)

wh = Function(W)
solve(a == L, wh, bcs)

uh, ph = split(wh)

# Is the flux bcs correct?
ds1 = Measure('ds', domain=mesh, subdomain_data=bdry, subdomain_id=1)
print(assemble(inner(uh, uh)*ds1))

# In terms of definition ?
print(assemble(inner(grad(ph), grad(ph))*ds1))

# When projected? Natural space
S = VectorFunctionSpace(mesh, 'DG', 1)
grad_ph = project(grad(ph), S)

print(assemble(inner(grad_ph, grad_ph)*ds1))

# Something else
S = VectorFunctionSpace(mesh, 'DG', 0)
grad_ph = project(grad(ph), S)

print(assemble(inner(grad_ph, grad_ph)*ds1))

# Something else
S = VectorFunctionSpace(mesh, 'CG', 1)
grad_ph = project(grad(ph), S)

print(assemble(inner(grad_ph, grad_ph)*ds1))
