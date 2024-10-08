from pyscf.scf import ghf
import numpy as np

class MagHF:
    def __init__(self, ghfmf):
        self.ghfmf = ghfmf
        self.mo_coeff = ghfmf.mo_coeff

        self.nso = self.mo_coeff.shape[0]
        self.nmo = self.nso // 2

    def modify_fock(self, B):
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

        def get_fock(mf):
            fock = self.ghfmf.get_fock() + Bs
            return fock
        
        self.ghfmf.get_fock = get_fock

        return
    
    def kernel(self, B):
        self.modify_fock(B)
        self.ghfmf.kernel()
        return

    
