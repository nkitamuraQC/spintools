from elemaghf import EleMagHF
from pyscf import gto, scf
import numpy as np
import copy
import pytest

def test_elemaghf():
    E0 = np.array([0.0, 0, 0])
    B0 = np.array([0.0, 0, 0])
    E1 = np.array([10, 10, 10])
    B1 = np.array([10, 0, 10])
    mol_H2_triplet = gto.M(
        atom = 'H 0 0 0; F 0 0 1.1',  # in Angstrom
        basis = 'sto-3g',
        charge = 1,
        spin = 1,
        verbose=4,
    )

    mol_H2 = gto.M(
        atom = 'H 0 0 0; F 0 0 1.1',  # in Angstrom
        basis = 'sto-3g',
        verbose=4,
    )

    myhf_H2_tri = scf.GHF(mol_H2_triplet)

    myhf_H2 = scf.GHF(mol_H2)

    emhf = EleMagHF(myhf_H2_tri)
    mf = emhf.kernel(E0, B0)
    B0_energy_H2_tri = mf.e_tot

    emhf = EleMagHF(myhf_H2_tri)
    mf = emhf.kernel(E1, B1)
    B1_energy_H2_tri = mf.e_tot

    emhf = EleMagHF(myhf_H2)
    mf = emhf.kernel(E0, B0)
    B0_energy_H2 = mf.e_tot

    emhf = EleMagHF(myhf_H2)
    mf = emhf.kernel(E1, B1)
    B1_energy_H2 = mf.e_tot

    assert(abs(B1_energy_H2_tri - B0_energy_H2_tri) > abs(B1_energy_H2 - B0_energy_H2))
    return