import pyscf
from pyscf.geomopt.geometric_solver import optimize as geomopt
import quadrupole

mol = pyscf.M(atom="N 0 0 0; H 1 1 0; H 1 -1 0; H 0 0 1", basis="sto3g")
mol = geomopt(mol.RHF())
hf = mol.RHF().run()
D = hf.make_rdm1()

q = quadrupole.get_quadrupole(mol, D)

print(q)
