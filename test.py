from elemaghf import EleMagHF
from pyscf import gto, scf
import numpy as np
import copy

def test_elemaghf():
    E0 = np.array([0.0, 0, 0])
    B0 = np.array([0.0, 0, 0])
    E1 = np.array([0.01, 0, 0])
    B1 = np.array([0.01, 0, 0])
    mol_O2 = gto.M(
        atom = 'O 0 0 0; O 0 0 1.1',  # in Angstrom
        basis = '6-31g',
        symmetry = True,
    )
    mol_O2.spin = 2

    mol_CO = gto.M(
        atom = 'C 0 0 0; O 0 0 1.1',  # in Angstrom
        basis = '6-31g',
        symmetry = True,
    )

    myhf_O2 = scf.GHF(mol_O2)
    myhf_O2.kernel()

    myhf_CO = scf.GHF(mol_CO)
    myhf_CO.kernel()

    emhf = EleMagHF(myhf_O2)
    mf = emhf.kernel(E0, B0)
    B0_energy_O2 = mf.e_tot

    emhf = EleMagHF(myhf_O2)
    mf = emhf.kernel(E0, B1)
    B1_energy_O2 = mf.e_tot

    emhf = EleMagHF(myhf_CO)
    mf = emhf.kernel(E0, B0)
    B0_energy_CO = mf.e_tot

    emhf = EleMagHF(myhf_CO)
    mf = emhf.kernel(E0, B1)
    B1_energy_CO = mf.e_tot

    assert(abs(B1_energy_O2 - B0_energy_O2) > abs(B1_energy_CO - B0_energy_CO))

    emhf = EleMagHF(myhf_O2)
    mf = emhf.kernel(E0, B0)
    E0_energy_O2 = mf.e_tot

    emhf = EleMagHF(myhf_O2)
    mf = emhf.kernel(E1, B0)
    E1_energy_O2 = mf.e_tot

    emhf = EleMagHF(myhf_CO)
    mf = emhf.kernel(E0, B0)
    E0_energy_CO = mf.e_tot

    emhf = EleMagHF(myhf_CO)
    mf = emhf.kernel(E1, B0)
    E1_energy_CO = mf.e_tot

    assert(abs(E1_energy_O2 - E0_energy_O2) < abs(E1_energy_CO - E0_energy_CO))
