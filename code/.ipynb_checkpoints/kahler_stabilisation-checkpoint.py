# ===================================================================================
# Copyright 2024 Liam McAllister, Jakob Moritz, Richard Nally, and Andreas Schachner
#
#   This script validates the results from Kähler moduli stabilization in the various
#   de Sitter and anti-de Sitter vacua obtained in ArXiv:2406.13751.
#
#   In the event of bugs or other issues, please reach out via as3475@cornell.edu 
#   or a.schachner@lmu.de.
#
# ===================================================================================
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------------


# Srandard imports
import warnings, os, sys, time

# Import numpy
import numpy as np

# Import scipy libraries
import scipy as sc
from scipy.special import zeta

# CYTools imports
from cytools import Polytope

# custom imports
from polylogs import *


def compute_corrected_cy_volume(cy,kahler,gs,bfield=None,GV=None):
    """
    **Summary:**
    Computes corrected CY volume including the tree level BBHL correction and worldsheet instantons as defined in Eq. (2.71).

    Args:
        cy (cytools.CalabiYau): CY threefold as class object of cytools.CalabiYau.
        kahler (np.ndarray): Kähler parameters specifying a point in Kähler moduli space.
        gs (np.float64): Value of the string coupling.
        bfield (None, optional): Values of the b-fields. Input should be an numpy.ndarray. Defaults to None.
        GV (None, optional): Array of GV invariants for the CY threefold. Input should be a numpy.ndarray. Defaults to None.
    
    Returns:
        np.float64: Corrected CY volume.
    
    Raises:
        ValueError: If GVs are provided, then also values for the b-field have to be given as input.
    """
    
    # Check if GVs are being provided
    if GV is not None:
        # Report Error if no bfield is provided
        if bfield is None:
            raise ValueError("If we compute with WSI correction, we need to provide the B-field!")
            
        # Report Error if no GVs have wrong format
        if type(GV)!=np.ndarray:
            raise ValueError("GV need to be provided through a numpy.ndarray!")

    # Define polytope and set Hodge numbers
    h11 = cy.h11()
    h21 = cy.h21()
    
    # Compute string frame kahler parameters
    kahler_str_frame = np.sqrt(gs)*kahler
    
    # Compute classical CY volume
    cy_volume = cy.compute_cy_volume(kahler_str_frame)
    
    ## alpha' corrections
    # Add BBHL corrections to the volume
    xi_BBHL = 1/4*zeta(3)/(2*np.pi)**3*2*(h21-h11)
    cy_volume = cy_volume+xi_BBHL
    
    # Add GV corrections if required
    if not GV is None:
    
        # Extract charges and GVs
        GV_charges = GV[:,:-1].astype(int)
        GVinvariants = GV[:,-1]

        # Compute polylogs
        complex_curve_volume = GV_charges@(1j*kahler_str_frame+bfield)
        plylog3_vals = np.real(polylog3(complex_curve_volume))
        gv_polylog2 = GVinvariants*np.real(polylog2(complex_curve_volume))

        # WSI corrections to the volume:
        cy_volume_wsi = np.real(1/(16*np.pi**3)*(GVinvariants@plylog3_vals+2*np.pi*(gv_polylog2)@(GV_charges@kahler_str_frame)))
        
        cy_volume = cy_volume+cy_volume_wsi
    
    return cy_volume
    
def compute_corrected_divisor_volumes(cy,kahler,gs,bfield=None,GV=None):
    """
    **Summary:**
    Computes corrected divisor volumes including alpha'^2 corrections as well as worlsheet instantons as defined in Eq. (2.73).
    
    Args:
        cy (cytools.CalabiYau): CY threefold as class object of cytools.CalabiYau.
        kahler (np.ndarray): Kähler parameters specifying a point in Kähler moduli space.
        gs (np.float64): Value of the string coupling.
        bfield (None, optional): Values of the b-fields. Input should be an numpy.ndarray. Defaults to None.
        GV (None, optional): Array of GV invariants for the CY threefold. Input should be a numpy.ndarray. Defaults to None.
    
    Returns:
        np.ndarray: Values of the corrected divisor volumes.
    
    Raises:
        ValueError: If GVs are provided, then also values for the b-field have to be given as input.
    """

    # Check if GVs are being provided
    if GV is not None:
        # Report Error if no bfield is provided
        if bfield is None:
            raise ValueError("If we compute with WSI correction, we need to provide the B-field!")

        # Report Error if no GVs have wrong format
        if type(GV)!=np.ndarray:
            raise ValueError("GV need to be provided through a numpy.ndarray!")
            

    # Define polytope and set Hodge numbers
    p = cy.polytope()
    h11 = cy.h11()
    h21 = cy.h21()
    
    # classical geometric data
    glsm = p.glsm_charge_matrix(include_origin=False)

    # Compute classical divisor volumes
    div_volumes = cy.compute_divisor_volumes(kahler)

    # Compute string frame kahler parameters
    kahler_str_frame = np.sqrt(gs)*kahler

    # Compute intersection numbers
    triple_int_numbers = cy.intersection_numbers()
    
    # Add correction to divisor volumes
    diagonal_intnums = np.array([triple_int_numbers.get((i,i,i),0) for i in range(1,h11+5)])
    chi_D = cy.second_chern_class()[1:]+diagonal_intnums
    div_volumes = div_volumes-chi_D/24/gs
    
    # Add GV corrections if required
    if not GV is None:
    
        # Extract charges and GVs
        GV_charges = GV[:,:-1].astype(int)
        GVinvariants = GV[:,-1]

        # Compute polylogs
        complex_curve_volume = GV_charges@(1j*kahler_str_frame+bfield)
        gv_polylog2 = GVinvariants*np.real(polylog2(complex_curve_volume))
        
        # WSI correction to the coordinates:
        div_volumes = div_volumes+1/gs*(1/(2*np.pi)**2*(gv_polylog2@GV_charges)@glsm)
    
    return div_volumes

def terms_non_pert_superpotential(cy,kahler,phi,rigid_divisors,dual_cox,gs,Pfaffians,bfield=None,GV=None):
    """
    **Summary:**
    Computes the values of each term in the non-perturbative superpotential for the Käher moduli.
    
    Args:
        cy (cytools.CalabiYau): CY threefold as class object of cytools.CalabiYau.
        kahler (np.ndarray): Kähler parameters specifying a point in Kähler moduli space.
        phi (np.ndarray): Values of the C4-axions.
        rigid_divisors (np.ndarray): Indices of the rigid prime toric divisors for the CY threefold hypersurface.
        dual_cox (np.ndarray): Dual coxeter numbers as defined by the corresponding orientifold.
        gs (np.float64): Value of the string coupling.
        Pfaffians (np.ndarray): Values of the Pfaffians.
        bfield (None, optional): Values of the b-fields. Input should be an numpy.ndarray. Defaults to None.
        GV (None, optional): Array of GV invariants for the CY threefold. Input should be a numpy.ndarray. Defaults to None.
        
    Returns:
        np.float64: Values of each term in the non-perturbative superpotential for the Käher moduli.
    
    Raises:
        ValueError: If GVs are provided, then also values for the b-field have to be given as input.
    """
    
    # Check if GVs are being provided
    if GV is not None:
        # Report Error if no bfield is provided
        if bfield is None:
            raise ValueError("If we compute with WSI correction, we need to provide the B-field!")
            
        # Report Error if no GVs have wrong format
        if type(GV)!=np.ndarray:
            raise ValueError("GV need to be provided through a numpy.ndarray!")

    # Define polytope and set Hodge numbers
    p = cy.polytope()
    h11 = cy.h11()
    h21 = cy.h21()
    
    # classical geometric data
    glsm = p.glsm_charge_matrix(include_origin=False)

    # Compute corrected divisor volumes
    div_volumes = compute_corrected_divisor_volumes(cy,kahler,gs,bfield=bfield,GV=GV)

    # Define chiral coordinates
    Tprime = div_volumes+1j*phi@glsm

    return Pfaffians*np.exp(-2*np.pi*(Tprime/dual_cox))

def potential_km(cy,kahler,phi,rigid_divisors,dual_cox,W0,gs,Pfaffians,bfield=None,GV=None,c_uplift=1,power=4/3,uplift=False,return_potential=True,vtilde=None,CSFterm=0):
    """
    **Summary:**
    Computes the value of the F-term scalar potential for the Kähler moduli and the C4-axions as defined in Eq. (D.9).
    
    Args:
        cy (cytools.CalabiYau): CY threefold as class object of cytools.CalabiYau.
        kahler (np.ndarray): Kähler parameters specifying a point in Kähler moduli space.
        phi (np.ndarray): Values of the C4-axions.
        rigid_divisors (np.ndarray): Indices of the rigid prime toric divisors for the CY threefold hypersurface.
        dual_cox (np.ndarray): Dual coxeter numbers as defined by the corresponding orientifold.
        W0 (np.float64): Value of the flux superpotential.
        gs (np.float64): Value of the string coupling.
        Pfaffians (np.ndarray): Values of the Pfaffians.
        bfield (None, optional): Values of the b-fields. Input should be an numpy.ndarray. Defaults to None.
        GV (None, optional): Array of GV invariants for the CY threefold. Input should be a numpy.ndarray. Defaults to None.
        c_uplift (np.float64, optional): Value of the coefficient in the upliftung potential from anti-D3 branes.
        power (np.float64, optional): Power of the inverse volume to be used in the uplifting potential. Defaults to 4/3 which is the right value for anti-D3 branes.
        uplift (bool, optional): Whether or not to include the uplifting potential. Defaults to False.
        return_potential (bool, optional): Whether or not to return also the value of the F-term scalar potential. Defaults to False.
        vtilde (None, optional): Value of the mirror dual CY volume at specific point in complex structure moduli space. Defaults to None.
        CSFterm (np.float64, optional): Value of the F-terms for the complex structure moudli to be used in the uplift.
    
    Returns:
        np.float64: Value of the F-term scalar potential for the Kähler moduli and the C4-axions.
    
    Raises:
        ValueError: If GVs are provided, then also values for the b-field have to be given as input.
    """
    
    # Check if GVs are being provided
    if GV is not None:
        # Report Error if no bfield is provided
        if bfield is None:
            raise ValueError("If we compute with WSI correction, we need to provide the B-field!")
            
        # Report Error if no GVs have wrong format
        if type(GV)!=np.ndarray:
            raise ValueError("GV need to be provided through a numpy.ndarray!")
            
    # Define polytope and set Hodge numbers
    p = cy.polytope()
    h11 = cy.h11()
    h21 = cy.h21()
    
    # classical geometric data
    glsm = p.glsm_charge_matrix(include_origin=False)

    # get charges of rigid divisors
    rigid_div_charges = np.transpose(glsm)[rigid_divisors-1]

    # Compute string frame kahler parameters
    kahler_str_frame = np.sqrt(gs)*kahler

    # Compute classical divisor volumes
    div_volumes = cy.compute_divisor_volumes(kahler)

    # Compute classical CY volume
    cy_volume = cy.compute_cy_volume(kahler_str_frame)

    # Compute kappa_ijk*t^k
    kappa_matrix = cy.compute_AA(kahler_str_frame)

    # Compute intersection numbers
    triple_int_numbers = cy.intersection_numbers()
    
    ## alpha' corrections
    # Add BBHL corrections to the volume
    xi_BBHL = 1/4*zeta(3)/(2*np.pi)**3*2*(h21-h11)
    cy_volume = cy_volume+xi_BBHL
    
    # Add correction to divisor volumes
    diagonal_intnums = np.array([triple_int_numbers.get((i,i,i),0) for i in range(1,h11+5)])
    chi_D = cy.second_chern_class()[1:]+diagonal_intnums
    div_volumes = div_volumes-chi_D/24/gs
    
    # Add GV corrections if required
    if GV is not None:

        # Extract charges and GVs
        GV_charges = GV[:,:-1].astype(int)
        GVinvariants = GV[:,-1]

        # Compute polylogs
        complex_curve_volume = GV_charges@(1j*kahler_str_frame+bfield)
        plylog3_vals = np.real(polylog3(complex_curve_volume))
        gv_polylog1 = GVinvariants*np.real(polylog1(complex_curve_volume))
        gv_polylog2 = GVinvariants*np.real(polylog2(complex_curve_volume))

        # WSI corrections to the volume:
        cy_volume_wsi = np.real(1/(16*np.pi**3)*(GVinvariants@plylog3_vals+2*np.pi*(gv_polylog2)@(GV_charges@kahler_str_frame)))
        cy_volume = cy_volume+cy_volume_wsi
        
        # WSI correction to the coordinates:
        div_volumes = div_volumes+1/gs*(1/(2*np.pi)**2*(gv_polylog2@GV_charges)@glsm)

        # Needed for later
        kappa_matrix = kappa_matrix-1/(2*np.pi)*np.einsum('k,ki,kj->ij',gv_polylog1,GV_charges,GV_charges)
        


    # Compute kappa_ijk*t^j*t^k + ... corrections
    y = kappa_matrix@kahler_str_frame

    # Define chiral coordinates
    Tprime = div_volumes+1j*phi@glsm

    # To absorb W0 in Wnp
    T0 = np.log(1/W0)/(2*np.pi) 

    # Define non-perturbative superpotential
    Wnp = Pfaffians*np.exp(-2*np.pi*(Tprime/dual_cox-T0))

    # Define 1st derivative of non-perturbative superpotential
    dWnp = -(2*np.pi)/(dual_cox)*Wnp

    # Define rescaled superpotential
    w = 1+np.sum(Wnp)

    # Define 1st derivative of rescaled superpotential
    d1w = glsm@dWnp
    
    # Define some useful terms
    yt = kahler_str_frame@y
    eps = (yt-2*cy_volume)**(-1)
    
    X = 1/2*y@d1w
    Y = d1w@kappa_matrix@np.conj(d1w)
    XbX = X*np.conj(X)
    XbW = X*np.conj(w)
    ww = gs**2*abs(w)**2

    # Compute corrections
    eta = 3*xi_BBHL/(2*cy_volume)
    
    if GV is not None:

        gv_term1 = 1/(2*np.pi)**2*GVinvariants@plylog3_vals
        gv_term2 = 1/(2*np.pi)*gv_polylog2@GV_charges
        gv_term3 = np.einsum('k,ki,kj->ij',gv_polylog1,GV_charges,GV_charges)

        eta += 1/(2*np.pi)*(3*gv_term1+3*kahler_str_frame@gv_term2+kahler_str_frame@gv_term3@kahler_str_frame)/(4*cy_volume)

    # Set prefactor coming from e^K factor
    prefactor = 4*gs**2/cy_volume
    
    # F-term scalar potential
    potential_fterms = np.real(prefactor*(-Y+eps*(4*XbX-4*gs*np.real(XbW)+eta*ww)))
    

    if uplift:
        potential_fterms += c_uplift/cy_volume**power

        c_uplift_CS = gs**3*CSFterm
        potential_fterms += c_uplift_CS/cy_volume**2

    if vtilde is None:
        print("Scalar potential may not be correctly normalised because of unknown factor e^(K_cs)\
                from the complex structure sector! Please provide the mirror Calabi-Yau volume as input!")
        vtilde = 1

    return potential_fterms/((128*vtilde)/((W0)**2))
    

def gradient_potential_km(cy,kahler,phi,rigid_divisors,dual_cox,W0,gs,Pfaffians,bfield=None,GV=None,c_uplift=0.,power=4/3,uplift=False,return_potential=False,vtilde=None,CSFterm=0.):
    """
    **Summary:**
    Computes the gradient of the F-term potential for the Kähler moduli and C4 axions as defined in Eq. (D.30) and (D.31).
    
    Args:
        cy (cytools.CalabiYau): CY threefold as class object of cytools.CalabiYau.
        kahler (np.ndarray): Kähler parameters specifying a point in Kähler moduli space.
        phi (np.ndarray): Values of the C4-axions.
        rigid_divisors (np.ndarray): Indices of the rigid prime toric divisors for the CY threefold hypersurface.
        dual_cox (np.ndarray): Dual coxeter numbers as defined by the corresponding orientifold.
        W0 (np.float64): Value of the flux superpotential.
        gs (np.float64): Value of the string coupling.
        Pfaffians (np.ndarray): Values of the Pfaffians.
        bfield (None, optional): Values of the b-fields. Input should be an numpy.ndarray. Defaults to None.
        GV (None, optional): Array of GV invariants for the CY threefold. Input should be a numpy.ndarray. Defaults to None.
        c_uplift (np.float64, optional): Value of the coefficient in the upliftung potential from anti-D3 branes.
        power (np.float64, optional): Power of the inverse volume to be used in the uplifting potential. Defaults to 4/3 which is the right value for anti-D3 branes.
        uplift (bool, optional): Whether or not to include the uplifting potential. Defaults to False.
        return_potential (bool, optional): Whether or not to return also the value of the F-term scalar potential. Defaults to False.
        vtilde (None, optional): Value of the mirror dual CY volume at specific point in complex structure moduli space. Defaults to None.
        CSFterm (np.float64, optional): Value of the F-terms for the complex structure moudli to be used in the uplift.
    
    Returns:
        np.ndarray: Gradient of the F-term potential for the Kähler moduli and C4 axions.
    
    Raises:
        ValueError: If GVs are provided, then also values for the b-field have to be given as input.
    """
    
    # Check if GVs are being provided
    if GV is not None:
        # Report Error if no bfield is provided
        if bfield is None:
            raise ValueError("If we compute with WSI correction, we need to provide the B-field!")
            
        # Report Error if no GVs have wrong format
        if type(GV)!=np.ndarray:
            raise ValueError("GV need to be provided through a numpy.ndarray!")


    if vtilde is None:
        print("Scalar potential may not be correctly normalised because of unknown factor e^(K_cs)\
            from the complex structure sector! Please provide the mirror Calabi-Yau volume as input!")
        vtilde = 1

    if uplift:
        c_uplift_CS = gs**3*CSFterm

    # Define polytope and set Hodge numbers
    p = cy.polytope()
    h11 = cy.h11()
    h21 = cy.h21()
    
    # classical geometric data
    glsm = p.glsm_charge_matrix(include_origin=False)

    # get charges of rigid divisors
    rigid_div_charges = np.transpose(glsm)[rigid_divisors-1]

    # Compute string frame kahler parameters
    kahler_str_frame = np.sqrt(gs)*kahler

    # Compute classical divisor volumes
    div_volumes = cy.compute_divisor_volumes(kahler)

    # Compute classical CY volume
    cy_volume = cy.compute_cy_volume(kahler_str_frame)

    # Compute kappa_ijk*t^k
    kappa_matrix = cy.compute_AA(kahler_str_frame)
    
    # Keep copy for later purposes
    kappa_matrix_cop = kappa_matrix.copy()

    # Compute intersection numbers
    triple_int_numbers = cy.intersection_numbers()
    
    ## alpha' corrections
    # Add BBHL corrections to the volume
    xi_BBHL = 1/4*zeta(3)/(2*np.pi)**3*2*(h21-h11)
    cy_volume = cy_volume+xi_BBHL
    
    # Add correction to divisor volumes
    diagonal_intnums = np.array([triple_int_numbers.get((i,i,i),0) for i in range(1,h11+5)])
    chi_D = cy.second_chern_class()[1:]+diagonal_intnums
    div_volumes = div_volumes-chi_D/24/gs
    
    # Add GV corrections if required
    if GV is not None:

        # Extract charges and GVs
        GV_charges = GV[:,:-1].astype(int)
        GVinvariants = GV[:,-1]

        # Compute polylogs
        complex_curve_volume = GV_charges@(1j*kahler_str_frame+bfield)
        plylog3_vals = np.real(polylog3(complex_curve_volume))
        gv_tensor0 = GVinvariants*np.real(polylog0(complex_curve_volume))
        gv_polylog1 = GVinvariants*np.real(polylog1(complex_curve_volume))
        gv_polylog2 = GVinvariants*np.real(polylog2(complex_curve_volume))
        gv_tensor3 = GVinvariants*plylog3_vals

        # WSI corrections to the volume:
        cy_volume_wsi = np.real(1/(16*np.pi**3)*(GVinvariants@plylog3_vals+2*np.pi*(gv_polylog2)@(GV_charges@kahler_str_frame)))
        cy_volume = cy_volume+cy_volume_wsi
        
        # WSI correction to the coordinates:
        div_volumes = div_volumes+1/gs*(1/(2*np.pi)**2*(gv_polylog2@GV_charges)@glsm)

        # Needed for later
        kappa_matrix = kappa_matrix-1/(2*np.pi)*np.einsum('k,ki,kj->ij',gv_polylog1,GV_charges,GV_charges)
        tmp = gv_tensor0*(GV_charges@kahler_str_frame)
        kappa_matrix_cop = kappa_matrix_cop+np.einsum('k,ki,kj->ij',tmp,GV_charges,GV_charges)


    # Compute kappa_ijk*t^j*t^k + ...
    y = kappa_matrix@kahler_str_frame

    # Define chiral coordinates
    Tprime = div_volumes+1j*phi@glsm

    # To absorb W0 in Wnp
    T0 = np.log(1/W0)/(2*np.pi) 

    # Define non-perturbative superpotential
    Wnp = Pfaffians*np.exp(-2*np.pi*(Tprime/dual_cox-T0))

    # Define 1st derivative of non-perturbative superpotential
    dWnp = -(2*np.pi)/(dual_cox)*Wnp

    # Define 2nd derivative of non-perturbative superpotential
    ddWnp =  (2*np.pi)**2/(dual_cox)**2*Wnp

    # Define rescaled superpotential
    w = 1+np.sum(Wnp)

    # Define 1st derivative of rescaled superpotential
    d1w = glsm@dWnp

    # Define 2nd derivative of rescaled superpotential
    d2w = np.einsum('kj,lj,j->kl',glsm,glsm,ddWnp)
    
    # Compute kappa_ijk * (\p_k W)
    T2W = cy.compute_AA(np.real(d1w))+1j*cy.compute_AA(np.imag(d1w))
    
    # Add GV corrections
    if GV is not None:
        tmp = gv_tensor0*(GV_charges@d1w)
        T2W = T2W+np.einsum('k,ki,kj->ij',tmp,GV_charges,GV_charges)

    # Define some useful terms
    Gamma = d2w@kappa_matrix@np.conj(d1w)
    z = kappa_matrix_cop@kahler_str_frame
    yt = kahler_str_frame@y
    xi = d2w@y
    Omega = kappa_matrix@d1w 
    eps = (yt-2*cy_volume)**(-1)
    eps_1 = -eps**2*(y+z)
    
    
    
    
    if GV is not None:

        gv_term1 = 1/(2*np.pi)**3*GVinvariants@plylog3_vals
        gv_term2 = 1/(2*np.pi)**2*gv_polylog2@GV_charges
        gv_term3 = 1/(2*np.pi)*np.einsum('k,ki,kj->ij',gv_polylog1,GV_charges,GV_charges)

        eta = (6*xi_BBHL+3*gv_term1+3*kahler_str_frame@gv_term2+kahler_str_frame@gv_term3@kahler_str_frame)/(4*cy_volume)

    else:

        eta = 3*xi_BBHL/(2*cy_volume)
        

    if GV is not None:

        tmp = gv_tensor0*(GV_charges@kahler_str_frame)
        gv_term4 = np.einsum('k,ki,kj->ij',tmp,GV_charges,GV_charges)

        eta_1 = (-y*2*eta-gv_term3@kahler_str_frame-gv_term4@kahler_str_frame)/(4*cy_volume)

    else:

        eta_1 = -y/(2*cy_volume)*eta

        
    

    X = 1/2*y@d1w
    X1down = 1/2*(kappa_matrix_cop+kappa_matrix)@d1w+1/2*gs**(-1)*kappa_matrix@xi
    X1up = 1j/2*xi
    Y = d1w@kappa_matrix@np.conj(d1w)
    Y1down = T2W@np.conj(d1w)+2/gs*kappa_matrix@np.real(Gamma)
    Y1up = -2*np.imag(d2w@np.conj(Omega))
    XbX = X*np.conj(X)
    XbW = X*np.conj(w)
    ww = gs**2*abs(w)**2

    # Define prefactor for scalar potential
    prefactor = 4*gs**2/cy_volume
    
    # F-term scalar potential
    potential_fterms = np.real(prefactor*(-Y+eps*(4*XbX-4*gs*np.real(XbW)+eta*ww)))
    
    # Derivative of V with respect to phi
    dVphi = -prefactor*np.real(Y1up+eps*(2*1j*gs*d1w*np.conj(2*X-gs*eta*w)+4*X1up*np.conj(gs*w-2*X)))
    
    # Derivative of V with respect to t
    ## Trivial piece from differentiating e^K
    dVkahler_kp = -y/(2*cy_volume)*potential_fterms

    ## Additional piece from F-terms
    dVkahler_fterms = - prefactor*(Y1down-eps_1*(eta*ww+4*XbX-4*gs*XbW)+eps*(4*X1down*np.conj(gs*w-2*X)-eta_1*ww+2*Omega*np.conj(2*X-gs*eta*w)))
    
    ## Combining the two terms
    dVkahler = np.real(dVkahler_kp+dVkahler_fterms)

    # Appropriately rescale gradients
    dVkahler = dVkahler/((128*vtilde)/((W0)**2))
    
    # Add uplifting piece if required:
    if uplift:

        # Add gradient from anti-D3 brane potential
        dVkahler = dVkahler -power*c_uplift*y/(2*cy_volume**(power+1))

        # Add gradient from complex structure potential
        dVkahler = dVkahler -2*c_uplift_CS*y/(2*cy_volume**3)

    # Combine derivatives with appropriate rescalings
    dV = np.concatenate((dVkahler*np.sqrt(gs) , dVphi/((128*vtilde)/((W0)**2))))

    if return_potential:

        # Appropriately rescale scalar potential and 
        potential_fterms = potential_fterms/((128*vtilde)/((W0)**2))

        if uplift:

            # Add anti-D3 brane potential
            potential_fterms += c_uplift/cy_volume**power

            # Add complex structure potential
            potential_fterms += c_uplift_CS/cy_volume**2

        return dV, potential_fterms
    else:
        return dV


def filter_curve_charges(gvs1,gvs2):
    """
    **Summary:**
    Remove charges from array of GVs given a second list of GVs.
    
    Args:
        gvs1 (np.ndarray): Array of GVs from which to remove elements.
        gvs2 (np.ndarray): Array of GVs to be removed from gvs1.

    
    Returns:
        np.ndarray: Curve charges of gvs1 not contained in gvs2.

    """

    # Get charges
    set1 = gvs1[:,:-1]
    set2 = gvs2[:,:-1]
    
    # Loop through charges in set1
    new_charges = []
    for curve in set1:

        # Test if curve is in set2
        flag = np.any(np.all(set2==curve,axis=1))

        # If curve exists in set2, continue
        if flag:
            continue

        # Otherwise append
        new_charges.append(curve)
        
    return np.array(new_charges)

def compute_curve_volumes(gvs,kahler_st):
    """
    **Summary:**
    Compute string frame volumes for curves.
    
    Args:
        gvs (np.ndarray): Array of GV invariants.
        kahler_st (np.ndarray): String frame Kähler parameters.

    
    Returns:
        np.ndarray: String frame volumes of the curves.

    """

    return gvs[:,:-1]@kahler_st


























