# ===================================================================================
# Copyright 2024 Liam McAllister, Jakob Moritz, Richard Nally, and Andreas Schachner
#
#   This script provides useful wrapper functions to validate the various
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
import pandas as pd

# Import scipy libraries
import scipy as sc
from scipy.special import zeta

# CYTools imports
from cytools import Polytope

# custom imports
from flux_vacua import *
from kahler_stabilisation import *

def call_my_name(example):

    paper_name = str(example["paper name"]).capitalize()

    if "name" in example.keys():
        name = str(example["name"])

        if name == "manwe":
            call_sign = "Manwe Sulimo, High King of Arda"
        elif name == "lorien":
            call_sign = "Lorien, The Dreamer"
        elif name == "aule":
            call_sign = "Aule, The Smith"
        elif name == "orome":
            call_sign = "Orome, The Hunter"
        elif name == "tulkas":
            call_sign = "Tulkas, The Strong"
        else:
            print("I have not heard of this LOTR character before, please tell me more about them...")
            call_sign = "..."
        print(paper_name+" aka "+call_sign)
    else:
        print(paper_name.capitalize())

    print("")

def get_nonpert_superpotential_AdS(example):

    # Construct polytope
    p = Polytope(example["points"])

    # Grab triangulations
    t = p.triangulate(heights = np.array(example["heights AdS"]))

    # Get CY threefold
    cy = t.get_cy()

    # Grab values for W0 and gs for the SUSY minimum
    gs = example["SUSY gs"]
    W0 = example["SUSY W0"]
    

    # Kahler
    kahler = np.array(example["kahler AdS"])
    phi = np.array(example["phi AdS"])

    # Make consistency check for kahler:
    if len(kahler)!=cy.h11():
        raise ValueError(f"Kahler parameters do not have the right shape.\
                        Expected {cy.h11()}, but go {len(kahler)}. Please check input!")

    # Make consistency check for phi:
    if len(phi)!=cy.h11():
        raise ValueError(f"Axion values do not have the right shape.\
                        Expected {cy.h11()}, but go {len(phi)}. Please check input!")


    # Define information about superpotential and worldsheet instantons
    GVs_X = np.array(example["GVs AdS"])
    rigid_list = np.array(example["rigid divisors"])
    dual_cox = np.array(example["dual coxeter numbers"])
    Pfaffians = np.array(example["Pfaffians"])
    bfield = np.array(example["bfield"])

    W0phase = W0/abs(W0)

    Wnp_terms = terms_non_pert_superpotential(cy,kahler,phi,rigid_list,dual_cox,gs,Pfaffians,bfield=bfield,GV=GVs_X)
    
    return W0phase*sum(Wnp_terms)

def verify_complex_structure_stabilisation(example):
    
    print("----------------------")
    print("")

    call_my_name(example)
    
    # Grab dual polytope points
    dual_points = np.array(example["dual points"])

    # Construct dual Polytope
    dual_p = Polytope(dual_points)

    # Get Hodge numbers
    p = dual_p.dual()
    h11 = p.h11(lattice="N")
    h21 = p.h21(lattice="N")

    print("Hodge numbers (h11,h21)=",(h11,h21))

    # Grab heights from dataframe
    mirror_heights = np.array(example["mirror heights"])

    # Construct mirror CY from heights
    dual_cy = dual_p.triangulate(heights = mirror_heights).get_cy()

    # Grab conifold curve
    coni_curve0 = np.array(example["conifold curve"]).astype(int)

    # Get basis change
    basis_change = np.array(example["basis transformation"]).astype(int)

    # Transfor conifold curve to new basis (should be (1,0,...,0))
    coni_curve = basis_change@coni_curve0

    print("Conifold curve: ",coni_curve0)
    print("Conifold curve in basis (should be (1,0,...,0)): ",coni_curve)


    # Define triple intersection numbers for mirror CY
    dual_intnums = dual_cy.intersection_numbers(in_basis=True,format='dense')

    # Define second Chern class for mirror CY
    dual_c2 = dual_cy.second_chern_class(in_basis=True)


    # Change triple intersection numbers to conifold basis
    dual_intnums=np.einsum('ai,ibc->abc', basis_change, np.einsum('bj,ijc->ibc', basis_change,np.einsum('ck,ijk->ijc', basis_change, dual_intnums)))

    # Change c2 to conifold basis
    dual_c2=np.matmul(basis_change,dual_c2)

    # grab tau
    tau = example["SUSY tau"]

    # grab complex structure moduli
    zvec = np.array(example["SUSY zvec"])

    # transform complex structure moduli to basis
    zvec_in_coni_basis = zvec@np.linalg.inv(basis_change)

    zcf = np.abs(zvec_in_coni_basis[0])

    print("Check that computed zcf agrees with the saved value: ", np.abs(example["SUSY zcf"]-zcf)<1e-5)

    print("")

    print("Flux choice...")

    print("")

    Mvec = np.array(example["M vector"])
    Kvec = np.array(example["K vector"])
    pvec = np.array(example["P vector"])

    print("M-vector: ",Mvec)
    print("K-vector: ",Kvec)
    print("p-vector: ",pvec)

    print("")

    ## Grab GVs
    GVs = np.array(example["mirror GVs"])

    ## Split GVs into invariants and curve charges
    GV_charges = GVs[:,:-1]
    GV_invariants = GVs[:,-1]

    # Change the basis of curve charges to the conifold basis
    GV_charges_basis = np.matmul(basis_change,GV_charges.T)

    ## Transform curve charges in the right basis
    GV_basis = np.append(GV_charges_basis,[GV_invariants],axis=0).T

    ## Find the index of the conifold curve (which by definition is a nilpotent curve with GV=n_cf)
    coni_index = np.where(np.all(GV_basis[:,:-1] == coni_curve,axis=1))[0][0]

    ## Grab the GV invariant associated with the conifold curve... this sets the number of conifolds
    coni_GV = GV_basis[coni_index][-1]

    ## Modify the second Chern class by shifting it accordingly
    dual_c2_prime = dual_c2+coni_curve*coni_GV

    # Define N-matrix (above eq. (3.13))
    Nmatrix = dual_intnums@Mvec

    # Define A-matrix (eq. (2.53))
    amatrix = np.mod(np.array([[dual_intnums[a][b][b] if a<=b else dual_intnums[b][a][a] for b in range(h21)] for a in range(h21)]),2)/2

    # Define c2' (eq. (3.5))
    dual_c2_prime = dual_c2+coni_curve*coni_GV

    # Define M-dual scalar (eq. (3.17))
    Mdualscalar = Mvec@dual_c2_prime/24

    # Define M-dual vector (eq. (3.16))
    Mdualvec = np.rint(amatrix@Mvec).astype(int)

    print("Tests at the PFV level...")

    print("")

    flag1 = sum(abs(Nmatrix@pvec-Kvec)[1:])<1e-10
    flag2 = np.mod(Mdualscalar,1)==0
    flag3 = sum(abs((amatrix@Mvec)-Mdualvec))==0
    flag4 = np.all(np.where(GV_basis[:,:-1]@pvec<=0)[0] == np.array([coni_index]))

    flags = [flag1,flag2,flag3,flag4]

    print("PFV condition 1 satisfied (flat direction: N.p=K (eq. (3.15))): ", flag1)

    print("PFV condition 2 satisfied (first quantization condition: b.M is integer (eq. (3.17))): ",flag2)

    print("PFV condition 3 satisfied (second quantization condition (eq. (3.16)): ",flag3)

    print("PFV condition 4 satisfied (Kahler cone condition (eq. (3.14)): ",flag4)

    if not all(flags):
        failures = np.where(list(map(lambda x: not x,flags)))[0]+1
        raise ValueError(f"PFV conditions {str(list(failures))[1:-1]} failed!")



    print("")

    print("Test the F-term conditions...")
    print("")

    Wnp = get_nonpert_superpotential_AdS(example)

    fterms,dw,dk,w=fterms_flux(zvec_in_coni_basis,tau,Mvec,Kvec,pvec,dual_intnums,amatrix,GV_basis,h11,h21,wconst = Wnp)

    print("F-terms: ",np.abs(fterms))
    print("")

    print("Compare computed values with saved values...")
    print("")

    gs = 1/np.imag(tau)

    print("Value of the flux superpotential W0 agrees with expected value from file: ", np.abs(example["SUSY W0"]-w)<1e-5)
    print("Value of the string coupling gs agrees with expected value from file: ", np.abs(example["SUSY gs"]-gs)<1e-5)
    print("")
    print("Print some quantities...")
    print("")
    print("W0: ",abs(w))
    print("")
    print("gs: ",1/np.imag(tau))

    print("")
    print("----------------------")


def verify_SUSY_AdS_minimum(example):

    print("----------------------")

    print("")

    call_my_name(example)

    # Construct polytope
    p = Polytope(example["points"])

    # Grab triangulations
    t = p.triangulate(heights = np.array(example["heights AdS"]))

    # Get CY threefold
    cy = t.get_cy()

    # Grab values for W0 and gs for the SUSY minimum
    gs = example["SUSY gs"]
    W0 = example["SUSY W0"]
    W0 = np.abs(W0)
    print("Input values for gs and W0 at the SUSY minimum:")
    print(f"gs = {gs}")
    print(f"W0 = {W0}")

    # Kahler
    kahler = np.array(example["kahler AdS"])
    phi = np.array(example["phi AdS"])

    # Make consistency check for kahler:
    if len(kahler)!=cy.h11():
        raise ValueError(f"Kahler parameters do not have the right shape.\
                        Expected {cy.h11()}, but go {len(kahler)}. Please check input!")

    # Make consistency check for phi:
    if len(phi)!=cy.h11():
        raise ValueError(f"Axion values do not have the right shape.\
                        Expected {cy.h11()}, but go {len(phi)}. Please check input!")


    # Define information about superpotential and worldsheet instantons
    GVs = np.array(example["GVs AdS"])
    rigid_list = np.array(example["rigid divisors"])
    dual_cox = np.array(example["dual coxeter numbers"])
    Pfaffians = np.array(example["Pfaffians"])
    bfield = np.array(example["bfield"])

    print("")

    # Compute the gradient of the scalar potential and the value of the scalar potential itself in one go...
    dV,V = gradient_potential_km(cy,kahler,phi,rigid_list,dual_cox,W0,gs,Pfaffians,return_potential=True,GV=GVs,bfield=bfield,vtilde=example["SUSY volTilde"])

    print("Maximum |dV|: ",np.max(np.abs(dV)))
    
    print("Maximum ratio |dV/V|: ",np.max(np.abs(dV)/np.abs(V)))

    print("")

    print("Computed vacuum energy: ",V)
    print("Value expected: ",example["V AdS"])
    flag = np.abs(V-example["V AdS"])<1e-5
    print("AdS vacuum energy matches expected value: ", flag)

    if not flag:
        raise ValueError("AdS vacuum energies do not match!")


    print("")
    print("Comparing volumes at the minimum...")

    print("")

    corrected_divisor_volumes = compute_corrected_divisor_volumes(cy,kahler,gs,bfield=bfield,GV=GVs)

    flag = np.abs(np.max(corrected_divisor_volumes-np.array(example["corrected divisor volumes AdS"])))<1e-5
    print("Corrected divisor volumes match expected value: ", flag)

    if not flag:
        raise ValueError("Corrected divisor volumes do not match!")

    CY_volume_str_frame = compute_corrected_cy_volume(cy,kahler,gs,bfield=bfield,GV=GVs)

    flag = np.abs(np.max(example["corrected CY volume AdS"]*gs**(3/2)-CY_volume_str_frame))<1e-5
    print("Corrected CY volume matches expected value", np.abs(np.max(example["corrected CY volume AdS"]*gs**(3/2)-CY_volume_str_frame))<1e-5)

    if not flag:
        raise ValueError("Corrected CY volumes do not match!")

    print("")
    print("Corrected CY volume in Einstein frame: ",CY_volume_str_frame/gs**(3/2))

    return None


def verify_uplifted_minimum(example,vacuum_type=None):

    if vacuum_type not in ["dS","AdS uplifted"]:
        raise ValueError(f"vacuum_type must be one of {['dS','AdS uplifted']}!")

    print("----------------------")

    print("")

    call_my_name(example)

    # Construct polytope
    p = Polytope(example["points"])

    # Grab triangulations
    t = p.triangulate(heights = np.array(example["heights "+vacuum_type]))

    # Get CY threefold
    cy = t.get_cy()

    # Grab values for W0 and gs for the SUSY minimum
    gs = example["gs"]
    W0 = example["W0"]
    W0 = np.abs(W0)
    print("Input values for gs and W0:")
    print(f"gs = {gs}")
    print(f"W0 = {W0}")

    # Kahler
    kahler = np.array(example["kahler "+vacuum_type])
    phi = np.array(example["phi "+vacuum_type])

    # Make consistency check for kahler:
    if len(kahler)!=cy.h11():
        raise ValueError(f"Kahler parameters do not have the right shape.\
                        Expected {cy.h11()}, but go {len(kahler)}. Please check input!")

    # Make consistency check for phi:
    if len(phi)!=cy.h11():
        raise ValueError(f"Axion values do not have the right shape.\
                        Expected {cy.h11()}, but go {len(phi)}. Please check input!")


    # Define information about superpotential and worldsheet instantons
    GVs = np.array(example["GVs "+vacuum_type])
    rigid_list = np.array(example["rigid divisors"])
    dual_cox = np.array(example["dual coxeter numbers"])
    Pfaffians = np.array(example["Pfaffians"])
    bfield = np.array(example["bfield"])


    print("")
    
    # Grab uplift information
    c_uplift = example["c uplift"]
    CSFterm = example["CSFterm"]
    
    dV,V = gradient_potential_km(cy,kahler,phi,rigid_list,dual_cox,W0,gs,Pfaffians,return_potential=True,GV=GVs,
                                 bfield=bfield,c_uplift=c_uplift,uplift=True,vtilde=example["volTilde"],CSFterm=CSFterm)
    
    
    print("Maximum |dV|: ",np.max(np.abs(dV)))
    
    print("Maximum ratio |dV/V|: ",np.max(np.abs(dV)/np.abs(V)))

    print("")

    print("Computed vacuum energy: ",V)
    print("Value expected: ",example["V "+vacuum_type])
    flag = np.abs(V-example["V "+vacuum_type])<1e-5
    print(vacuum_type+" vacuum energy matches expected value: ", flag)

    if not flag:

        raise ValueError(vacuum_type+" vacuum energies do not match!")


    print("")

    print("Comparing volumes at the minimum...")

    print("")

    corrected_divisor_volumes = compute_corrected_divisor_volumes(cy,kahler,gs,bfield=bfield,GV=GVs)

    flag = np.abs(np.max(corrected_divisor_volumes-np.array(example["corrected divisor volumes "+vacuum_type])))<1e-5
    print("Corrected divisor volumes match expected value: ", flag)

    if not flag:
        raise ValueError("Corrected divisor volumes do not match!")

    CY_volume_str_frame = compute_corrected_cy_volume(cy,kahler,gs,bfield=bfield,GV=GVs)

    flag = np.abs(np.max(example["corrected CY volume "+vacuum_type]*gs**(3/2)-CY_volume_str_frame))<1e-5
    print("Corrected CY volume matches expected value", np.abs(np.max(example["corrected CY volume "+vacuum_type]*gs**(3/2)-CY_volume_str_frame))<1e-5)

    if not flag:
        raise ValueError("Corrected CY volumes do not match!")

    print("")
    print("Corrected CY volume in Einstein frame: ",CY_volume_str_frame/gs**(3/2))

    

    return None



















