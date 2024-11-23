from pyscf.scf import ghf
import numpy as np
import copy
from spinqc.mod_kernel import kernel


class EleMagHF:
    def __init__(self, ghfmf):
        self.ghfmf = ghfmf
        self.mo_coeff = ghfmf.mo_coeff
        self.r = self.ghfmf.mol.intor("int1e_r")

        self.nso = len(ghfmf.mol.ao_labels()) * 2
        self.nmo = self.nso // 2

        self.original_get_fock = self.ghfmf.get_fock

    def modify_fock(self, E, B):
        I = np.identity(self.nmo)
        zero = np.zeros((self.nmo, self.nmo))

        sigmax1 = np.concatenate([zero, I], axis=1)
        sigmax2 = np.concatenate([-I, zero], axis=1)
        sigmax = np.concatenate([sigmax1, sigmax2], axis=0)

        sigmay1 = np.concatenate([zero, -1.0j * I], axis=1)
        sigmay2 = np.concatenate([1.0j * I, zero], axis=1)
        sigmay = np.concatenate([sigmay1, sigmay2], axis=0)

        sigmaz1 = np.concatenate([I, zero], axis=1)
        sigmaz2 = np.concatenate([zero, -I], axis=1)
        sigmaz = np.concatenate([sigmaz1, sigmaz2], axis=0)

        # Bs = B[0] * sigmax + B[1] * sigmay + B[2] * sigmaz
        # Complex matrix is ignored
        Bs = B[0] * sigmax + B[2] * sigmaz
        Er = E[0] * self.r[0] + E[1] * self.r[1] + E[2] * self.r[2]
        Er1 = np.concatenate([Er, zero], axis=1)
        Er2 = np.concatenate([zero, Er], axis=1)
        Er = np.concatenate([Er1, Er2], axis=0)

        # print("Bs =", Bs)
        # print("Er =", Er)

        def custom_get_fock(h1e, s1e, vhf, dm):
            fock_matrix = self.original_get_fock(h1e, s1e, vhf, dm)
            # print("Before: ", fock_matrix)
            fock_matrix += Bs + Er
            # print("After: ", fock_matrix)
            return fock_matrix

        self.ghfmf.get_fock = custom_get_fock

        return

    def kernel(self, E, B):
        self.modify_fock(E, B)
        scf_conv, e_tot, mo_energy, mo_coeff, mo_occ = kernel(self.ghfmf)
        return scf_conv, e_tot, mo_energy, mo_coeff, mo_occ
