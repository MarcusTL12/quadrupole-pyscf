import numpy as np


def get_nuc_quadrupole(mol):
    r = mol.atom_coords()
    q = mol.atom_charges()

    q1 = np.einsum("ki,kj,k -> ij", r, r, q)

    q_diag = sum(np.dot(r, r) * q for r, q in zip(r, q))

    return 0.5 * (3 * q1 - np.eye(3) * q_diag)


def get_electronic_quadrupole(mol, D):
    nao = mol.nao

    D = D.reshape(nao**2)

    q_int = mol.intor("int1e_rr").reshape(3, 3, 8, 8)

    q1 = np.zeros((3, 3))
    for i, qv in enumerate(q_int):
        for j, qv in enumerate(qv):
            q1[i, j] = np.dot(qv.reshape(nao**2), D)

    q_diag = np.dot(mol.intor("int1e_r2").reshape(nao**2), D)

    return 0.5 * (3 * q1 - np.eye(3) * q_diag)


def get_quadrupole(mol, D):
    return get_nuc_quadrupole(mol) - get_electronic_quadrupole(mol, D)
