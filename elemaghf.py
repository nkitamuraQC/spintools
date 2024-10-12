from pyscf.scf import ghf
import numpy as np
import copy

class EleMagHF:
    def __init__(self, ghfmf):
        self.ghfmf = ghfmf
        self.mo_coeff = ghfmf.mo_coeff
        self.r = self.ghfmf.mol.intor('int1e_r')

        self.nso = self.mo_coeff.shape[0]
        self.nmo = self.nso // 2

    def modify_fock(self, E, B):
        I = np.identity(self.nmo)
        zero = np.zeros((self.nmo, self.nmo))

        sigmax1 = np.concatenate([zero, I], axis=1)
        sigmax2 = np.concatenate([-I, zero], axis=1)
        sigmax = np.concatenate([sigmax1, sigmax2], axis=0)

        sigmay1 = np.concatenate([zero, -1.0j*I], axis=1)
        sigmay2 = np.concatenate([1.0j*I, zero], axis=1)
        sigmay = np.concatenate([sigmay1, sigmay2], axis=0)

        sigmaz1 = np.concatenate([I, zero], axis=1)
        sigmaz2 = np.concatenate([zero, -I], axis=1)
        sigmaz = np.concatenate([sigmaz1, sigmaz2], axis=0)

        Bs = B[0] * sigmax + B[1] * sigmay + B[2] * sigmaz
        Er = E[0] * self.r[0] + E[1] * self.r[1] + E[2] * self.r[2]
        Er1 = np.concatnate([Er, zero], axis=1)
        Er2 = np.concatnate([zero, Er], axis=1)
        Er = np.concatenate([Er1, Er2], axis=0)

        def get_fock(mf):
            fock = self.ghfmf.get_fock() + Bs + Er
            return fock
        
        self.ghfmf.get_fock = get_fock

        return
    
    def kernel(self, E, B):
        self.modify_fock(E, B)
        self.ghfmf.kernel()
        ghfmf = copy.deepcopy(self.ghfmf)
        return ghfmf

    
