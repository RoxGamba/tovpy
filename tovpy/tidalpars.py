"""
Copyright (C) 2024 Sebastiano Bernuzzi and others

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
"""

import numpy as np
from scipy.special import factorial2

def q_of_nu(q):
    """
    mass-ratio to symmetric mass-ratio
    assume q>=1
    """
    return q/((1.+q)**2)

def q_of_eta(q):
    """
    mass-ratio to symmetric-mass ratio
    assume q>=1
    dummy (double notation)
    """
    return q_of_nu(q)

def nu_of_q(nu):
    """
    symmetric mass-ratio to mass-ratio 
    assume q>=1
    """
    if nu==0:
        q = []
    else:
        q = (1. + np.sqrt(1 - 4.*nu) - 2.*nu)/(2.*nu)
    return q

def eta_of_q(nu):
    """
    symmetric mass-ratio to mass-ratio 
    assume q>=1
    dummy (double notation)
    """
    return nu_of_q(nu)

def Lamtilde_of_eta_Lam1_Lam2(eta, Lam1, Lam2):
    r"""
    $\tilde\Lambda(\eta, \Lambda_1, \Lambda_2)$.
    Lambda_1 is assumed to correspond to the more massive (primary) star m_1.
    Lambda_2 is for the secondary star m_2.
    """
    return (8.0/13.0)*((1.0+7.0*eta-31.0*eta**2)*(Lam1+Lam2) + np.sqrt(1.0-4.0*eta)*(1.0+9.0*eta-11.0*eta**2)*(Lam1-Lam2))
    
def deltaLamtilde_of_eta_Lam1_Lam2(eta, Lam1, Lam2):
    """
    This is the definition found in Les Wade's paper.
    Les has factored out the quantity \sqrt(1-4\eta). It is different from Marc Favata's paper.
    $\delta\tilde\Lambda(\eta, \Lambda_1, \Lambda_2)$.
    Lambda_1 is assumed to correspond to the more massive (primary) star m_1.
    Lambda_2 is for the secondary star m_2.
    """
    return (1.0/2.0)*(
        np.sqrt(1.0-4.0*eta)*(1.0 - 13272.0*eta/1319.0 + 8944.0*eta**2/1319.0)*(Lam1+Lam2)
        + (1.0 - 15910.0*eta/1319.0 + 32850.0*eta**2/1319.0 + 3380.0*eta**3/1319.0)*(Lam1-Lam2)
    )
    
def Lam1_Lam2_of_pe_params(eta, Lamt, dLamt):
    """
    Lam1 is for the the primary mass m_1.
    Lam2 is for the the secondary mass m_2.
    m_1 >= m2.
    """
    a = (8.0/13.0)*(1.0+7.0*eta-31.0*eta**2)
    b = (8.0/13.0)*np.sqrt(1.0-4.0*eta)*(1.0+9.0*eta-11.0*eta**2)
    c = (1.0/2.0)*np.sqrt(1.0-4.0*eta)*(1.0 - 13272.0*eta/1319.0 + 8944.0*eta**2/1319.0)
    d = (1.0/2.0)*(1.0 - 15910.0*eta/1319.0 + 32850.0*eta**2/1319.0 + 3380.0*eta**3/1319.0)
    den = (a+b)*(c-d) - (a-b)*(c+d)
    Lam1 = ( (c-d)*Lamt - (a-b)*dLamt )/den
    Lam2 = (-(c+d)*Lamt + (a+b)*dLamt )/den
    # Adjust Lam1 and Lam2 if Lam1 becomes negative
    # Lam2 should be adjusted such that Lamt is held fixed
    if Lam1<0:
        Lam1 = 0
        Lam2 = Lamt / (a-b)
    return Lam1, Lam2

def Yagi2013_fitcoefs(ell):
    """
    Coefficients of Yagi 2013 fits for multipolar
    $\bar{\lambda}_\ell = 2 k_\ell/(C^{2\ell+1} (2\ell-1)!!)$
    Tab.I (NS) http://arxiv.org/abs/1311.0872
    Note: Yagi's $\bar{\lambda}_\ell$ is $\Lambda_\ell$
    """
    if ell==3:
        c = [-1.15,1.18,2.51e-2,-1.31e-3,2.52e-5]
    elif ell==4:
        c = [-2.45,1.43,3.95e-2,-1.81e-3,2.8e-5]
    else:
        c = []
    return c

def Yagi2013_fit_Laml(Lam2, ell):
    """
    Yagi 2013 fits for multipolar
    $\bar{\lambda}_\ell$ = 2 k_\ell/(C^{2\ell+1} (2\ell-1)!!)$
    Eq.(10),(61); Tab.I; Fig.8 http://arxiv.org/abs/1311.0872
    Note: Yagi's $\bar{\lambda}_\ell$ is $\Lambda_\ell$
    """
    lnx = np.log(Lam2)
    coefs = Yagi2013_fitcoefs(ell)
    lny = np.polyval(coefs[::-1], lnx)
    return np.exp(lny)

def Laml_to_kappal(q, LamAl, LamBl, ell):
    """
    $\kappa^{A,B}_\ell(\bar{\lambda}_\ell)$
    Assume $q=M_A/M_B>=1$
    Note: Yagi's $\bar{\lambda}_\ell$ is $\Lambda_\ell$
    """
    XA = q/(1.+q)
    XB = 1. - XA
    f2l1 = factorial2(2*ell-1)
    p = 2*ell + 1
    kappaAl = f2l1 * LamAl * XA**p / q
    kappaBl = f2l1 * LamBl * XB**p * q
    return  kappaAl, kappaBl

def klC_to_Laml(C, kl, ell):
    """
    $\bar{\lambda}_\ell(k_\ell,C)$
    Compactness and Love numbers to Yagi tidal parameters  
    Note: Yagi's $\bar{\lambda}_\ell$ is $\Lambda_\ell$
    """
    div = 1.0/(factorial2(2*ell-1)*C**(2*ell+1))
    return 2*kl*div

def LamlC_to_kl(Laml,C,ell=2):
    """
    Inverse of LoveC_to_Laml() 
    """
    fact = factorial2(2*ell-1)*C**(2*ell+1)
    return 0.5*Laml*fact

def Chang2014_fitcoefs(ell):
    """
    Coefficients of Chang+ 2014 fits for f-mode frequency
    Tab.I of https://arxiv.org/abs/1408.3789
    """
    if ell == 2:
        a = np.array([+1.820*1e-1,
                      -6.836*1e-3,
                      -4.196*1e-3,
                      +5.215*1e-4,
                      -1.857*1e-5])
    elif ell == 3:
        a = np.array([+2.245*1e-1,
                      -1.500*1e-2,
                      -1.412*1e-3,
                      +1.832*1e-4,
                      -5.561*1e-6])
    elif ell == 4:
        a = np.array([+2.501*1e-1,
                      -1.646*1e-2,
                      -5.897*1e-4,
                      +8.695*1e-5,
                      -2.368*1e-6])
    elif ell == 5:
        a = np.array([+2.681*1e-1,
                      -1.638*1e-2,
                      -2.497*1e-4,
                      +4.712*1e-5,
                      -1.166*1e-6])
    else:
        print('no fit for ell>5')
        a = np.array([])
    return a

def Chang2014_fit_omgf(Lam,ell=2):
    """
    Fit f-mode frequency vs Lambda
    Eq.(3.5) and Tab.I of https://arxiv.org/abs/1408.3789
    """
    y = np.log(Lam)
    a = Chang2014_fitcoefs(ell)
    return  np.polyval(a[::-1], y)




