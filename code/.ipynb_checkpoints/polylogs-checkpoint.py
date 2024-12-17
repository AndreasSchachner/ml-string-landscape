# ===================================================================================
# Copyright 2024 Liam McAllister, Jakob Moritz, Richard Nally, and Andreas Schachner
#
#   This script contains functions to compute poly-logarithms relevant for the various
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

# mpmath imports
from mpmath import polylog as polylog_mpmath



# polylogarithms
zetaprime = np.array([sc.special.zeta(3),np.pi**2/6, 3/2, -1/2, -1/12, 0, 1/120, 0, -1/252])
otherfactors = np.array([(2*np.pi*1j)**i/np.math.factorial(i) for i in range(5)])
expansion_coefficients = np.array([zetaprime[0+i:5+i]*otherfactors for i in range(5)])

def polylog3_pert(z):
    """
    **Summary:**
    Computes perturbative polylog 3.
    
    Args:
        z (np.ndarray): Values of the complex structure moduli.
    
    Returns:
        np.complex128: Value of perturbative polylog 3.
    """
    test = np.where(np.array([abs(z)>0,abs(z)==0]))[0][0]
    zpowers = z**np.arange(5)
    term1 = np.array([-(2*np.pi*1j)**3*(z**2/(4*np.pi*1j)*np.log(-2*np.pi*1j*z)),0])[test]
    term2 = expansion_coefficients[0]@zpowers
    return term1+term2

def polylog2_pert(z):
    """
    **Summary:**
    Computes perturbative polylog 2.

    Args:
        z (np.ndarray): Values of the complex structure moduli.
    
    Returns:
        np.complex128: Value of perturbative polylog 2.
    """
    test = np.where(np.array([abs(z)>0,abs(z)==0]))[0][0]
    zpowers = z**np.arange(5)
    term1 = np.array([-(2*np.pi*1j)**2*(z/(2*np.pi*1j)*(np.log(-2*np.pi*1j*z)+1/2)),0])[test]
    term2 = expansion_coefficients[1]@zpowers
    return term1+term2

def polylog1_pert(z):
    """
    **Summary:**
    Computes perturbative polylog 1.

    Args:
        z (np.ndarray): Values of the complex structure moduli.
    
    Returns:
        np.complex128: Value of perturbative polylog 1.
    """
    zpowers = z**np.arange(5)
    term1 = -(2*np.pi*1j)**1*(1/(2*np.pi*1j)*(np.log(-2*np.pi*1j*z)+3/2))
    term2 = expansion_coefficients[2]@zpowers
    return term1+term2

def polylog0_pert(z):
    """
    **Summary:**
    Computes perturbative polylog 0.

    Args:
        z (np.ndarray): Values of the complex structure moduli.
    
    Returns:
        np.complex128: Value of perturbative polylog 0.
    """
    zpowers = z**np.arange(5)
    term1 = -(2*np.pi*1j)**0*(1/(2*np.pi*1j*z))
    term2 = expansion_coefficients[3]@zpowers
    return term1+term2

def polylogm1_pert(z):
    """
    **Summary:**
    Computes perturbative polylogm 1.

    Args:
        z (np.ndarray): Values of the complex structure moduli.
    
    Returns:
        np.complex128: Value of perturbative polylogm 1.
    """
    zpowers = z**np.arange(5)
    term1 = -(2*np.pi*1j)**(-1)*(-1/(2*np.pi*1j*z**2))
    term2 = expansion_coefficients[4]@zpowers
    return term1+term2

def polylogm1(z):
    """
    **Summary:**
    Computes polylogm 1.
    
    Args:
        z (np.ndarray): Values of the complex structure moduli.
    
    Returns:
        np.complex128: Value of polylogm 1.
    """
    q = np.exp(2*np.pi*1j*z)
    test = np.array([abs(z)<1e-5,abs(z)>=1e-5])
    results = np.array([polylogm1_pert(z),q/(1-q)**2])
    result = results[np.where(test)[0][0]]
    return result

def polylog0(z):
    """
    **Summary:**
    Computes polylog 0.
    
    Args:
        z (np.ndarray): Values of the complex structure moduli.
    
    Returns:
        np.complex128: Value of polylog 0.
    """
    q = np.exp(2*np.pi*1j*z)
    test = np.array([abs(z)<1e-5,abs(z)>=1e-5])
    results = np.array([polylog0_pert(z),q/(1-q)])
    result = results[np.where(test)[0][0]]
    return result

def polylog1(z):
    """
    **Summary:**
    Computes polylog 1.

    Args:
        z (np.ndarray): Values of the complex structure moduli.
    
    Returns:
        np.complex128: Value of polylog 1.
    """
    q = np.exp(2*np.pi*1j*z)
    test = np.array([abs(z)<1e-5,abs(z)>=1e-5])
    results = np.array([polylog1_pert(z),-np.log(1-q)])
    result = results[np.where(test)[0][0]]    
    return result

def polylog2(z):
    """
    **Summary:**
    Computes polylog 2.
    
    Args:
        z (np.ndarray): Values of the complex structure moduli.
    
    Returns:
        np.complex128: Value of polylog 2.
    """
    q = np.exp(2*np.pi*1j*z)
    test = np.array([abs(z)<1e-5,abs(z)>=1e-5])
    results = np.array([polylog2_pert(z),float(polylog_mpmath(2,q).real)+1j*float(polylog_mpmath(2,q).imag)])
    result = results[np.where(test)[0][0]]   
    return result

def polylog3(z):
    """
    **Summary:**
    Computes polylog 3.
    
    Args:
        z (np.ndarray): Values of the complex structure moduli.
    
    Returns:
        np.complex128: Value of polylog 3.
    """
    q = np.exp(2*np.pi*1j*z)
    test = np.array([abs(z)<1e-5,abs(z)>=1e-5])
    results = np.array([polylog3_pert(z),float(polylog_mpmath(3,q).real)+1j*float(polylog_mpmath(3,q).imag)])
    result = results[np.where(test)[0][0]]   
    return result

# Vectorizing the polylogs
polylog0 = np.vectorize(polylog0)
polylog1 = np.vectorize(polylog1)
polylog2 = np.vectorize(polylog2)
polylog3 = np.vectorize(polylog3)
polylogm1 = np.vectorize(polylogm1)

