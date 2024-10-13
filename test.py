from elemaghf import EleMagHF
from pyscf import gto, scf
import numpy as np
import copy
import pytest

def test_elemaghf():
    E0 = np.array([0.0, 0, 0])
    B0 = np.array([0.0, 0, 0])
    E1 = np.array([0, 0, 0])
    B1 = np.array([10, 0, 10])
    mol_H2_triplet = gto.M(
        atom = 'H 0 0 0; H 0 0 1.1',  # in Angstrom
        basis = 'sto-3g',
        charge = 1,
        spin = 1,
        verbose=4,
    )

    mol_H2 = gto.M(
        atom = 'H 0 0 0; H 0 0 1.1',  # in Angstrom
        basis = 'sto-3g',
        verbose=4,
    )

    myhf_H2_tri = scf.GHF(mol_H2_triplet)

    myhf_H2 = scf.GHF(mol_H2)

    emhf = EleMagHF(myhf_H2_tri)
    scf_conv, e_tot_E0B0_tri, mo_energy, mo_coeff, mo_occ = emhf.kernel(E0, B0)

    emhf = EleMagHF(myhf_H2_tri)
    scf_conv, e_tot_E1B1_tri, mo_energy, mo_coeff, mo_occ = emhf.kernel(E1, B1)

    print(e_tot_E0B0_tri, e_tot_E1B1_tri, e_tot_E1B1_tri - e_tot_E0B0_tri)

    emhf = EleMagHF(myhf_H2)
    scf_conv, e_tot_E0B0, mo_energy, mo_coeff, mo_occ = emhf.kernel(E0, B0)

    emhf = EleMagHF(myhf_H2)
    scf_conv, e_tot_E1B1, mo_energy, mo_coeff, mo_occ = emhf.kernel(E1, B1)

    print(e_tot_E0B0, e_tot_E1B1, e_tot_E1B1 - e_tot_E0B0)

    assert(abs(e_tot_E1B1_tri - e_tot_E0B0_tri) > abs(e_tot_E1B1 - e_tot_E0B0))
    return