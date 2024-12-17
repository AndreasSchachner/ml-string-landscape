# ===================================================================================
# Copyright 2024 Stefano Lanza, and Andreas Schachner
#
#   This script provides code to compute flux vacua for the rigid CY manifold.
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

# IMPORTS
# =======
# standard
# --------
import warnings, os, sys, time
import numpy as np

# jax
# --------
from jax import jit, vmap
import jax 
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

# typing
# --------
from typing import Any, Callable, Sequence, Tuple
from jax import Array
from jax.typing import ArrayLike


@jit
def W(flux: ArrayLike, tau: complex) -> complex:
    r"""
    **Description:**
    Computes value of the superpotential.
    
    Args:
        flux (ArrayLike): Flux choice.
        tau (complex): Value of axio-dilaton.
    
    Returns:
        complex: Value of the superpotential.
    """
    f1,f2,h1,h2 = flux[0],flux[1],flux[2],flux[3]
    
    return f1+1j*f2-tau*(h1+1j*h2)


#Get Axio-Dilaton
@jit
def tau_val(flux: ArrayLike) -> complex:
    r"""
    **Description:**
    Computes value of the axio-dilaton VEV at the SUSY minimum.
    
    Args:
        flux (ArrayLike): Flux choice.
    
    Returns:
        complex: Value of the axio-dilaton VEV at the SUSY minimum.
    """
    f1,f2,h1,h2 = flux[0],flux[1],flux[2],flux[3]
    
    real_tau = f1*h1+f2*h2
    
    imag_tau = -f2*h1+f1*h2
    
    tau_denom = h1**2+h2**2
    
    tau_value = real_tau/tau_denom+1j*imag_tau/tau_denom
        
    return tau_value



def map_to_FD_tau(tau: complex, flux: ArrayLike) -> Tuple[Array,Array]:
    r""" 
    **Description:**
    Maps choice of axio-dilaton and fluxes to the fundamental domain under SL(2,Z).
    
    Args:
        tau (complex): Value of the axio-dilaton.
        flux (ArrayLike): Flux choice.
    
    Returns:
        float: Value of axio-dilaton in the fundamental domain under SL(2,Z).
        Array: Value of fluxes in the fundamental domain under SL(2,Z).
    """
    f1,f2,h1,h2 = flux[0],flux[1],flux[2],flux[3]
    
    tau0 = tau.real
    tau1 = tau.imag
    
    count=0
    
    end_loop=0
    
    tau00=tau0
    tau11=tau1
    
    if np.abs(tau00)<=0.5 and np.sqrt(tau00**2+tau11**2)>=1.:
        
        #tau_value=np.array([tau0,tau1,f1,f2,h1,h2]).astype(float)
    
        return tau00+1j*tau11,np.array([f1,f2,h1,h2]).astype(float)
        
    while end_loop<1:  
        
        
        if count>1000:
            #print("Needed to stop map to FD domain!")
            return np.array([0,0]).astype(float)
        
        temp1=int(np.floor(tau00))
        p_list=[tau00%1,tau11]
        f1=f1-temp1*h1
        f2=f2-temp1*h2
        
        
        if np.abs(p_list[0])>1/2:
            
            temp2=np.sign(p_list[0])
            p_list[0]=p_list[0]-temp2
            f1=f1-temp2*h1
            f2=f2-temp2*h2
            
            
            count=count+1
            
        if np.sqrt(p_list[0]**2+p_list[1]**2)<1:
            f1_old=f1
            f2_old=f2
            
            f1=h1
            f2=h2
            
            h1=-f1_old
            h2=-f2_old
            
            norm_p=p_list[0]**2+p_list[1]**2
            p_list[0]=-p_list[0]/norm_p
            p_list[1]=p_list[1]/norm_p
            count=count+1
            
        tau00=p_list[0]
        tau11=p_list[1]
        if np.abs(tau00)<=0.5 and np.sqrt(tau00**2+tau11**2)>=1.:
            end_loop=1
            
    
    return tau00+1j*tau11,np.array([f1,f2,h1,h2]).astype(float)

#Get NFlux
@jit
def Nflux(flux: ArrayLike) -> float:
    r"""
    **Description:**
    Computes D3-charge induced by the fluxes.
    
    Args:
        flux (ArrayLike): Flux choice.
    
    Returns:
        float: D3-charge induced by the fluxes.
    """
    f1,f2,h1,h2 = flux[0],flux[1],flux[2],flux[3]
    
    return f1*h2-f2*h1



#Get value scalar potential
@jit
def V(flux: ArrayLike, tau: complex) -> float:
    r"""
    **Description:**
    Compute the scalar potential for input fluxes.
    
    Args:
        tau (complex): Value of the axio-dilaton.
        flux (ArrayLike): Flux choice.
    
    Returns:
        float: Value of scalar potential.
    """
    f1,f2,h1,h2 = flux[0],flux[1],flux[2],flux[3]
    
    tau1 = tau.imag
    tau0 = tau.real
    
    VPotDenom=tau1/2.
    
    WV=W(flux, tau)
    Vpot=(-h1-1j*h2+1j/(2.*tau1)*(WV[0]+1j*WV[1]))*(-h1+1j*h2-1j/(2.*tau1)*(WV[0]-1j*WV[1]))*4.*tau1**(2.)
    Vpot=Vpot/VPotDenom
    
    
    return Vpot.real

vtau_val = vmap(tau_val)
vNflux = vmap(Nflux)
vW = vmap(W)


def sample_vacua(num_vacua: int, max_flux: int, Qmax: int = 100) -> Tuple[Array,Array]:
    r"""
    **Description:**
    Samples vacua by uniformly picking flux vectors.
    
    Args:
        num_vacua (int): Number of vacua to be sampled.
        max_flux (int): Maximum absolute value for the fluxes.
        Qmax (int, optional): Maximally allowed tadpole.
        
    Returns:
        ArrayLike: Array of sampled fluxes.
        ArrayLike: Array of sampled values of the axio-dilaton.
    
    """

    fluxes = []
    while len(fluxes)<num_vacua:
        tmp = np.random.randint(-max_flux,max_flux+1,(num_vacua,4))
        tmp = np.unique(tmp,axis=0)
        nflux = vNflux(tmp)
        tau_values = vtau_val(tmp)
        flag = (nflux<=Qmax)&(tau_values.imag>0)
        if len(fluxes)==0:
            fluxes = tmp[flag]
        else:
            fluxes = np.append(fluxes,tmp[flag],axis=0)

    fluxes = np.array(fluxes)
    tau_values = np.array(vmap(tau_val)(fluxes))

    tau_values_fd = []
    fluxes_fd = []
    for i in range(len(tau_values)):
        
        tau,flux=map_to_FD_tau(tau_values[i],fluxes[i])
        tau_values_fd.append(tau)
        fluxes_fd.append(flux)

    fluxes = np.array(fluxes_fd)
    tau_values = np.array(tau_values_fd)
    
    return fluxes, tau_values






