# ===================================================================================
# Copyright 2024 Stefano Lanza, and Andreas Schachner
#
#   This script provides code to perform optimisation on flux vacua on the rigid CY 
#   using Genetic Algorithms.
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
import os, sys, warnings, time, tqdm, glob
import numpy as np

# plotting
# --------
import seaborn as sn
import matplotlib.pylab as plt
import matplotlib as mpl
import time
from IPython import display
from matplotlib.gridspec import GridSpec

plt.rcParams["font.family"] = "Times New Roman"
sn.set(rc={"figure.dpi":200, 'savefig.dpi':200})
sn.set_context('notebook')
sn.set_style("ticks")
cmap=sn.color_palette("viridis", as_cmap=True)
sn.set_theme()

# typing
# --------
from typing import Any, Callable, Sequence, Tuple
from numpy.typing import ArrayLike

# custom
# --------
from flux_vacua import *
from utils import * 



# generic
# -------
def lift_population_to_flux_vacua(
                                  pop: ArrayLike, 
                                  Qmax: int = 100
                                  ) -> Array:
    r"""
    **Description:**
    Lifts choice of DNAs to flux vacua.
    
    Args:
        pop (Array): Population of DNAs.
        Qmax (int, optional): Maximum allowed tadpole. Defaults to `Qmax=100`.
    
    Returns:
        Array: Lifted choice of DNAs.
    
    """

    nflux = np.array(vNflux(pop))

    tau_values = np.array(vtau_val(pop))

    flag = (nflux<=Qmax)&(tau_values.imag>0)

    tmp = np.array(pop[flag])

    tau_values = tau_values[flag]

    pop_size = len(tmp)
    fluxes = [] 
    for i in range(pop_size):

        tau,flux=map_to_FD_tau(tau_values[i],tmp[i])

        fluxes.append(flux)

    return fluxes


def prepare_initial_population(
                               population_size: int, 
                               allele_bound: int, 
                               Qmax: int = 100
                               ) -> Array:
    r"""
    **Description:**
    Prepares initial population by sampling flux vacua.
    
    Args:
        population_size (int): Population size.
        allele_bound (int): Maximum allowed flux number.
        Qmax (int, optional): Maximum allowed tadpole. Defaults to `Qmax=100`.
        
    Returns:
        Array: Array of DNAs.
    
    """

    fluxes, _ = sample_vacua(population_size,allele_bound,Qmax=Qmax)
    
    return fluxes

def construct_next_generation_population(
                                         mutants: ArrayLike,
                                         population_size: int,
                                         print_progress: bool = False, 
                                         Qmax: int = 100
                                         ) -> Array:
    r"""
    **Description:**
    Construct next population from list of fluxes.
    
    Args:
        mutants (Array): Array of DNAs.
        population_size (int): Population size.
        print_progress (bool, optional): If `True`, prints progress.
        Qmax (int, optional): Maximum allowed tadpole. Defaults to `Qmax=100`.
    
    Returns:
        Array: Array of DNAs constituting new population.
    
    Raises:
        ValueError: If not sufficiently many choices of DNA can be obtained, the code reports error.
    """
    
    
    count=0
    population_nn=[]
    failed_lifts = 0
    while len(population_nn)<population_size:

        if print_progress:
            print(f"Flux vacua: {len(population_nn)}     ",flush=False,end="\r")

        population_new = mutants[count*population_size:(count+1)*population_size]

        if len(population_new)==0:
            break
        
        lifted_population = lift_population_to_flux_vacua(population_new,Qmax=Qmax)
        
        if len(lifted_population)==0:
            continue

        if len(population_nn)==0:
            population_nn = lifted_population
        else:
            population_nn = np.append(population_nn,lifted_population,axis=0)
            
        population_nn = np.unique(population_nn,axis=0)

        count+=1
        
        
        
    if len(population_nn)<population_size:
        raise ValueError(f"Did not find sufficiently many Flux Vacua for the next generation! Only obtained {len(population_nn)} vacua, but required {population_size}!")
        
    return np.array(population_nn)

def select_roulette_wheel(
                          numsel: int,
                          dna_size: int, 
                          population: ArrayLike, 
                          fitness: ArrayLike
                          ) -> Array:
    r"""
    **Description:**
    Selection operator: Roulette Wheel. Selects parents by choosing random individuals with the normalised fitness as a weighting function. 
    
    Args:
        numsel (int): Number of pairs to be selected.
        dna_size (int): Length of the chromosomes or DNA sequence.
        population (Array): Population of DNAs.
        fitness (array): Values of the fitness across population.
    
    Returns:
        array: pairs of parents used for crossover
    """
    if numsel==0:
        return np.array([[[]]])
    
    parentinds=np.random.choice(len(population), 2*numsel, p=fitness)

    return population[parentinds].reshape(numsel,2,dna_size)



def selection_tournament(
                         numsel: int,
                         dna_size: int, 
                         population: ArrayLike, 
                         fitness: ArrayLike,
                         toursize: int
                         ) -> Array:
    r"""
    **Description:**
    Selection operator: Tournament. Selects parents by performing a number of tournaments, i.e., choose a number 
    (=toursize < p = population size) of individuals and take only the fittest.
    
    Args:
        numsel (int): Number of pairs to be selected.
        dna_size (int): Length of the chromosomes or DNA sequence.
        population (Array): Population of DNAs.
        fitness (array): Values of the fitness across population.
        toursize (int): size of the tournaments
    
    Returns:
        array: pairs of parents used for crossover
    """
    if numsel==0:
        return np.array([[[]]])
    
    lengthpop=len(population)
    
    parentinds=[]
    
    num_tournaments = 2*numsel
    
    for i in range(num_tournaments):
        
        tourind=np.random.choice(lengthpop,toursize)
        
        fitnesstour=fitness[tourind]
        
        parentinds.append(np.random.choice(np.where(fitness==np.amax(fitnesstour))[0],1)[0])
        
   
    return population[parentinds].reshape(numsel,2,dna_size)


def crossover_n_point(
                      dna_size: int, 
                      numcuts: int, 
                      pairs: ArrayLike
                      ) -> Array:
    r"""
    **Description:**
    Crossover operator: N-point crossover. Performs crossover by cutting the chromosome of one parent at N random positions replacing the 
    intermediate substring by the alleles of the second parent.
    
    Args:
        dna_size (int): Length of the chromosomes or DNA sequence.
        numcuts (int): Number of cuts for n-point crossover.
        pairs (Array): Array of pairs to perform crossover on.
    
    Returns:
        Array: Array of children.
    """

    
    if dna_size<=numcuts:
        raise ValueError('THERE ARE NOT ENOUGH ALLELES FOR N-POINT CROSSOVER!')
       
    len_pairs=len(pairs)
    population_co=[]
    
    for ind_pair in range(len_pairs):
        
        indlist=np.random.choice(np.arange(dna_size),numcuts) #indices at which to cut
        indlist.sort() #sort the list
        
        parent1=pairs[ind_pair][0].copy()
        parent2=pairs[ind_pair][1].copy()
        child1=pairs[ind_pair][0].copy()
        child2=pairs[ind_pair][1].copy()
        
        for i in range(0,numcuts,2):#only want to replace alleles very second cut
            if i>len(indlist):
                break
            if numcuts>1 and i<numcuts-1:
                child1[indlist[i]:indlist[i+1]]=parent2[indlist[i]:indlist[i+1]]
                child2[indlist[i]:indlist[i+1]]=parent1[indlist[i]:indlist[i+1]]
                
            
            elif numcuts>1 and i==numcuts-1:
                child1[indlist[i]:dna_size]=parent2[indlist[i]:dna_size]
                child2[indlist[i]:dna_size]=parent1[indlist[i]:dna_size]
                
            
            elif numcuts==1:
                child1[indlist[i]:dna_size]=parent2[indlist[i]:dna_size]
                child2[indlist[i]:dna_size]=parent1[indlist[i]:dna_size]
                    
        population_co.append(child1)
        population_co.append(child2)
        
    
    return np.array(population_co)

def crossover_uniform(
                      dna_size: int, 
                      pairs: ArrayLike
                      ) -> Array:
    r"""
    **Description:**
    Crossover operator: Uniform crossover. Performs crossover by deciding for each allele whether it is obtained from parent A or parent B.
    
    Args:
        dna_size (int): Length of the chromosomes or DNA sequence.
        pairs (Array): Array of pairs to perform crossover on.
    
    Returns:
        Array: Array of children.
    """
    
    
    len_pairs=len(pairs) 
    
    population_co=[]
    
    for ind_pair in range(len_pairs):
        
        parent1=pairs[ind_pair][0].copy()
        parent2=pairs[ind_pair][1].copy()
        child1=pairs[ind_pair][0].copy()
        child2=pairs[ind_pair][1].copy()
        
        for j in range(dna_size):
            
            KC=np.random.random()
            
            if KC<=0.5:
                child1[j]=parent2[j]
                child2[j]=parent1[j]
                    
        population_co.append(child1)
        population_co.append(child2)

    
    return np.array(population_co)



def mutation_random(
                    dna_size: int, 
                    allele_bound: int, 
                    children: ArrayLike, 
                    mutation_rate: float,
                    nummut: int = 1
                    ) -> Array:
    r"""
    **Description:**
    Mutation operator: Random mutation. Performs mutations on children with a random number of replacements. 
    The number of replacements can be different for each individual.
    
    Args:
        dna_size (int): Length of the chromosomes or DNA sequence.
        allele_bound (int): Maximum flux value.
        children (Array): Array of children.
        mutation_rate (float): Mutation rate.
        nummut (int, optional): Maximum number of mutations.
    
    Returns:
        array: array of mutants
    
    """
    
    len_children=len(children)
    population=[]
    
    for ind_children in range(len_children):
        
        q=np.random.random()
        
        if q<mutation_rate:
        
            # number of mutations to be performed
            if nummut is None:
                nummut=np.random.randint(0,dna_size+1)
        
            child=list(children[ind_children])
        
            # perform mutation for at most nummut alleles
            for i in range(nummut):
            
                KC=np.random.random()
            
                if KC<=0.5:
                    mutind=np.random.randint(0,dna_size)
                    mutval=np.random.randint(-allele_bound,allele_bound+1)
                    child[mutind]=mutval
            
            population.append(child)
        else:
            population.append(list(children[ind_children]))
        
    
    return np.array(population)

def make_ga_progess_plot(
                         fitness: ArrayLike,
                         generation: int,
                         num_features: int,
                         features: ArrayLike,
                         fig: object,
                         gs: object,
                         num_generations: int,
                         cmap: object=None,
                         log_scale_plot: bool = True,
                         run_folder: str = "./"
                         ) -> None:
    r"""
    **Description:**
    Makes progress plot for the GA.
    
    Args:
        fitness (Array): Value of fitness for population.
        generation (int): Current generation.
        num_features (int): Number of features.
        features (Array): Features.
        fig (plt.figure): Figure.
        gs (GridSpec): Grid.
        num_generations (int): Maximum number of generations.
        cmap (int, optional): Color map.
        log_scale_plot (bool, optional): Whether to u.se log-scale for the plot
        run_folder (str, optional): Folder to store plots.
    """
    
    if cmap is None:
        cmap = sn.color_palette("viridis", as_cmap=True)

    plot_color = cmap((generation+1e-4)/num_generations)
    
    num_bins = min([int(len(features)//4),100])

    if generation==0:
        ax2 = fig.add_subplot(gs[0, 0])
    else:
        ax2 = fig.axes[0]

    maxf = features[np.argmax(fitness)]

    #ax2.hist(fitness, bins=num_bins,color=plot_color,label=f"generation {generation}");
    ax2.plot(generation,maxf, 'go--', linewidth=2, markersize=7,color=plot_color)

    if log_scale_plot:
        ax2.set_yscale('log')

    ax2.set_title(f"Best feature for generation {generation}")

    ax2.set_ylabel("best feature")
    ax2.set_xlabel("generation")

    

    if generation==0:
        ax2 = fig.add_subplot(gs[1, 0])
    else:
        ax2 = fig.axes[1]
            
    if log_scale_plot:
        _, bins = np.histogram(np.log10(fitness + 1e-20), bins=num_bins)
        ax2.hist(fitness, bins=10**bins,color=plot_color,label=f"generation {generation}");

        ax2.loglog()
        
    else:
        ax2.hist(fitness, bins=num_bins,color=plot_color,label=f"generation {generation}");

    ax2.set_title(f"Fitness for generation {generation}")
    ax2.set_xlabel("fitness")
    ax2.set_ylabel("count")

    if num_features==1:

        if generation==0:
            ax2 = fig.add_subplot(gs[2, 0])
        else:
            ax2 = fig.axes[2]
            
        if np.max(np.abs(features))<1e-3:
            log_scale_plot=True
        
        if log_scale_plot:
            _, bins = np.histogram(np.log10(features + 1e-20), bins=num_bins)#bins='auto'
            ax2.hist(features, bins=10**bins,color=plot_color);

            ax2.loglog()
        else:
            ax2.hist(features, bins=num_bins,color=plot_color);

        ax2.set_title(f"Features for generation {generation}")

        ax2.set_ylabel("count")
        ax2.set_xlabel("feature")
        

    else:
        ft = features.T
        for f1 in range(num_features):

            if generation ==0:
                ax2 = fig.add_subplot(gs[f1+2, 0])
            else:
                ax2 = fig.axes[f1+2]

            data = ft[f1]
            
            if np.max(np.abs(data))<1e-3:
                log_scale_plot=True
            
            if log_scale_plot:
                _, bins = np.histogram(np.log10(data + 1e-20), bins=num_bins)#bins='auto'
                ax2.hist(data, bins=10**bins,color=plot_color);
            
                ax2.loglog()
            else:
                ax2.hist(data, bins=num_bins,color=plot_color);

            ax2.set_title(f"Feature {f1} for generation {generation}")
            ax2.set_ylabel("count")
            ax2.set_xlabel(f"feature {f1}")

    if generation>0:
        if generation == 1:
            ax3  = fig.add_axes([0.99,0.10,0.03,0.85])
        else:
            ax3  = fig.axes[-1]
            
        norm = mpl.colors.Normalize(vmin=0,vmax=num_generations)#generation
        cb1  = mpl.colorbar.ColorbarBase(ax3,cmap=cmap,norm=norm,orientation='vertical',label="generation")

    plt.tight_layout()
    
    plt.savefig(run_folder+"progress_plot.pdf",format="pdf",dpi=200, bbox_inches='tight')

    display.display(plt.gcf())
    display.clear_output(wait=True)
    
    
def construct_mutants(
                      population_size: int,
                      dna_size: int, 
                      population: ArrayLike,
                      fitness: ArrayLike,
                      allele_bound: int,
                      hyper_dict: dict = None
                      ) -> Array:
    r"""
    **Description:**
    Performs GA operations on input population based on fitness and provided hyperparameters.
    
    Args:
        population_size (int): Population size.
        dna_size (int): Length of the chromosomes or DNA sequence.
        population (Array): Population of DNAs.
        fitness (Array): Value of fitness for population.
        allele_bound (int): Maximum flux value.
        hyper_dict (dict, optional): Dictionary of hyperparameters.
    
    Returns:
        TYPE: Description
    
    Raises:
        ValueError: Description
    """

    # Grab hyperparameters
    num_survival_fittest = np.rint(hyper_dict["survival fittest"]).astype(int)
    mutation_rate = hyper_dict["mutation rate"]
    toursize = np.rint(hyper_dict["tournament size"]).astype(int)
    num_select_rw = np.rint(hyper_dict["selection rw"]).astype(int)
    num_select_tour = np.rint(hyper_dict["selection tour"]).astype(int)
    num_cross_triv = np.rint(hyper_dict["crossover triv"]).astype(int)
    num_cross_1pt = np.rint(hyper_dict["crossover 1pt"]).astype(int)
    num_cross_2pt = np.rint(hyper_dict["crossover 2pt"]).astype(int)
    num_cross_uni = np.rint(hyper_dict["crossover uni"]).astype(int)
    num_mutation_rand = np.rint(hyper_dict["mutation rand"]).astype(int)
    num_mutation_opt = np.rint(hyper_dict["mutation 1pt"]).astype(int)
    num_mutation_tpt = np.rint(hyper_dict["mutation 2pt"]).astype(int)
    num_mutation_triv = np.rint(hyper_dict["mutation triv"]).astype(int)

    ########### SELECTION ############

    # Perform roulette wheel selection
    pairs_rw=select_roulette_wheel(num_select_rw,dna_size, population, fitness)

    # Perform tournament selection
    pairs_tour=selection_tournament(num_select_tour,dna_size, population, fitness,toursize)
    
    # Combine all pairs
    try:
        pairs = np.append(pairs_rw, pairs_tour, axis=0)
    except:
        pairs = pairs_rw if len(pairs_rw) > len(pairs_tour) else pairs_tour
        print(pairs_rw)
        print(pairs_tour)

    # Shuffle pairs
    np.random.shuffle(pairs)

    ########### CROSSOVER ############

    ### NO CROSSOVER AT ALL!
    pairs_flat = pairs.reshape(pairs.shape[1]*pairs.shape[0],pairs.shape[-1])

    children_triv = pairs_flat[:num_cross_triv]

    np.random.shuffle(pairs)

    children_op = crossover_n_point(dna_size, 1, pairs[:num_cross_1pt])

    np.random.shuffle(pairs)

    children_tp = crossover_n_point(dna_size, 2, pairs[:num_cross_2pt])

    np.random.shuffle(pairs)

    children_u = crossover_uniform(dna_size, pairs[:num_cross_uni])

    # Combine children
    try:
        children = np.append(children_tp,children_u,axis=0)
    except:
        children = children_tp if len(children_tp) > len(children_u) else children_u
        print(children_tp)
        print(children_u)
        
    try:
        children = np.append(children_op,children,axis=0)
    except:
        print(children_op)
        print(children)
    
    try:
        children = np.append(children_triv,children,axis=0)
    except:
        print(children_triv)
        print(children)

    # Shuffle the children to reduce bias
    np.random.shuffle(children)

    ########### MUTATION ############

    # Perform random mutation
    mutants_rand = mutation_random(dna_size, allele_bound, children[:num_mutation_rand], mutation_rate)

    # Random shuffle children
    np.random.shuffle(children)

    # Perform 1-point random mutation
    mutants_opt = mutation_random(dna_size, allele_bound, children[:num_mutation_opt], mutation_rate,nummut=1)

    # Random shuffle children
    np.random.shuffle(children)

    # Perform 2-point random mutation
    mutants_tpt = mutation_random(dna_size, allele_bound, children[:num_mutation_tpt], mutation_rate,nummut=2)

    # Random shuffle children
    np.random.shuffle(children)

    # Perform no mutation
    mutants_triv = children[:num_mutation_triv] 

    # Combine mutants
    try:
        mutants = np.append(mutants_rand,mutants_opt,axis=0)
    except:
        mutants = mutants_rand if len(mutants_rand) > len(mutants_opt) else mutants_opt
        print(mutants_rand)
        print(mutants_opt)
        
    try:
        mutants = np.append(mutants_tpt,mutants,axis=0)
    except:
        print(mutants_tpt)
        print(mutants)
        
    try:
        mutants = np.append(mutants_triv,mutants,axis=0)
    except:
        print(mutants_triv)
        print(mutants)

    # REMOVE DUPLICATES
    mutants = np.unique(mutants,axis=0)

    # If there are not enough mutants left to construct new population, raise error!
    if len(mutants)<population_size:
        raise ValueError("Have not found sufficiently many mutants! Check input!")

    # We random shuffle the mutants to reduce biasses
    np.random.shuffle(mutants)
    
    return mutants


    
    
def perform_survival_fittest(
                             population: ArrayLike,
                             mutants: ArrayLike,
                             fitness: ArrayLike,
                             num_survival_fittest: int
                             ) -> Array:
    r"""
    **Description:**
    
    Args:
        population (Array): Population of DNAs.
        mutants (Array): Array of DNAs.
        fitness (Array): Value of fitness for population.
        num_survival_fittest (int): Number of fit individuals surviving.
    
    Returns:
        Array: Updated DNAs after survival.
    """
    _,index_fit = np.unique(fitness,return_index=True)

    for ii in range(num_survival_fittest):

        index_fittest = index_fit[-(ii+1)]

        mutants[ii] = population[index_fittest]
        
    return mutants

def generate_random_hyperparameters(
                                    population_size: int,
                                    num_generations: int,
                                    scaling_factor: int = 2
                                    ) -> dict:
    r"""
    **Description:**
    Generates random hyperparameters.

    Args:
        population_size (int): Population size.
        num_generations (int): Maximum number of generations.
        scaling_factor (int, optional): Scaling factor used for generation of offsprings.
    
    Returns:
        dict: Dictionary of hyperparameters.
    
    """

    hyper_keys = ["survival fittest","mutation rate","tournament size","selection rw","selection tour",
                  "crossover triv","crossover 1pt","crossover 2pt","crossover uni","mutation rand",
                  "mutation 1pt","mutation 2pt","mutation triv"]


    if population_size>=1000:
        num_survival_fittest = max([np.random.randint(population_size/1000,population_size/100),1])
        tour_size = max([np.random.randint(population_size/1000,population_size/100),2])
    else:
        num_survival_fittest = 1
        tour_size = 2
        
    mutation_rate = np.random.uniform(0.1,1)
    hyper_values = [num_survival_fittest,mutation_rate,tour_size]

    num_selection_operators = 2

    selection = np.random.uniform(0.1,2,num_selection_operators)
    selection = np.rint(scaling_factor*selection/np.linalg.norm(selection)*population_size).astype(int)


    num_crossover_operators = 4

    crossover = np.random.uniform(0.1,2,num_crossover_operators)
    crossover = np.rint(scaling_factor*crossover/np.linalg.norm(crossover)*population_size).astype(int)

    num_mutation_operators = 4

    mutation = np.random.uniform(0.1,2,num_mutation_operators)
    mutation = np.rint(scaling_factor*mutation/np.linalg.norm(mutation)*population_size).astype(int)


    hyper_values = np.append(np.append(np.append(hyper_values,selection),crossover),mutation)

    hyper_dict = dict(zip(hyper_keys,hyper_values))

    # Complete dictionary for the hyperparameters
    hyper_dict["population size"] = population_size
    hyper_dict["number generations"] = num_generations
    
    return hyper_dict

# MAIN METHODS
# ============
def run_GA(
    population_size: int,
    num_generations: int,
    fitness_function: Callable = None, 
    optimisation_target: Callable = None, 
    hyper_dict: dict = None, 
    initial_population: ArrayLike = None, 
    plot_progress: bool = False,
    cmap: object = None,
    save_files: bool = True,
    run_folder: str = None,
    log_scale_plot: bool = True,
    Qmax: int = None,
    allele_bound: int = None,
    sigma: float = 3, 
    mu: float = 7, 
    update_sigma: bool = True, 
    verbosity: int = 0,
    dynamical_pop_size: bool = False,
    max_population_size: int = 100):
    r"""
    **Description:**
    Run GA.

    Args:
        population_size (int): Population size.
        num_generations (TYPE): Description
        fitness_function (TYPE): Description
        optimisation_target (Callable, optional): Description
        hyper_dict (dict, optional): Description
        initial_population (Array, optional): Description
        plot_progress (bool, optional): Description
        cmap (??, optional): Description
        save_files (bool, optional): Description
        run_folder (str, optional): Description
        log_scale_plot (bool, optional): Description
    
    Returns:
        Array: Final population.
        Array: Features of final population.
        Array: GA history, i.e., all the populations (flux choices) encountered during the run.
    
    """
    
    
    if run_folder is None:
        raise ValueError("Please specify a folder where data can be stored!")
    else:
        if os.path.isdir(run_folder)==False:
            os.mkdir(run_folder)
        else:
            files = glob.glob(run_folder+"*")
            for file in files:
                if os.path.isfile(file):
                    os.remove(file)

    if hyper_dict is None:
        raise ValueError("Please provide dictionary for the hyperparameters!")
        
    else:
        hyper_keys = ["survival fittest","mutation rate","tournament size","selection rw","selection tour",
                      "crossover triv","crossover 1pt","crossover 2pt","crossover uni","mutation rand","mutation 1pt","mutation 2pt","mutation triv"]
        keys = list(hyper_dict.keys())
        
        if not all(x in keys for x in hyper_keys):
            raise ValueError("Key missing in dictionary for hyperparameters! Please check input of hyperparameters!")



        hyper_dict['sigma'] = sigma
        hyper_dict['mu'] = mu
        save_zipped_pickle(hyper_dict,run_folder+f"/hyperparameters.p")

        
        
    num_survival_fittest = int(hyper_dict['survival fittest'])
        
    if optimisation_target is None:
        raise ValueError("Please provide target function for the optimisation!")
    
    
    output = []
    history = []
    
    
    dna_size = 4
    
    if Qmax is None:
        raise ValueError("Please provide value for maximum tadpole!")
        
    if allele_bound is None:
        raise ValueError("Please provide value for the maximum flux values!")
    
    features_tmp,_ = optimisation_target(np.array([[1,2,-3,4]]))
    
    if type(features_tmp) != np.ndarray and type(features_tmp) != list:
        num_features = 1
    else:
        num_features = len(features_tmp)
        
    
    features = [0]
    
    if plot_progress:
        fig = plt.figure(figsize=(8, 3*(num_features+2)),dpi=100)

        gs = GridSpec(nrows=num_features+2, ncols=1)

        if cmap is None:
            cmap = sn.color_palette("viridis", as_cmap=True)
    
    
    
    
    
    
    count_freeze = 0
    best_individual = 0
    
    if num_features>1:
        print("NEED TO ADAPT THE CODE BELOW FOR MORE THAN ONE FEATURE FOR OPTIMISATION!")
        
    

    if verbosity>=1:
        print(f"Running {num_generations} generations...")
        
    try:
    #if True:
        for generation in range(num_generations+1):
            if verbosity>=2:
                print(f"Generation #{generation}...")


            if generation==0:

                if initial_population is None:
                    if verbosity>=1:
                        print("Generating initial population...")

                    population = prepare_initial_population(population_size,allele_bound,Qmax=Qmax)

                    if verbosity>=1:
                        print("Done generating initial population...")

                    if len(population)<population_size:
                        raise ValueError("Initial population is too small! Something must have gone wrong!")

                else:

                    population = construct_next_generation_population(initial_population,population_size,Qmax=Qmax)

                    if len(population)<population_size:
                        raise ValueError("Please check initial input population! It contains non-liftable 2-face FRTs!")

                population = population[:population_size]
                history.append(population)

            if verbosity>=1:
                print("Saving files...")

            if save_files:
                save_zipped_pickle(population,run_folder+f"/population_{generation}.p")

            if verbosity>=1:
                print("Computing features...")

            # HOW DO WE WANT TO HANDLE THIS???
            # # # #
            features, extras = optimisation_target(population)#get_features(population,optimisation_target=optimisation_target)

            if save_files:
                save_zipped_pickle(features,run_folder+f"/features_{generation}.p")
                if extras!=dict():
                    save_zipped_pickle(extras,run_folder+f"/output_{generation}.p")



            # Compute fitness
            fitness = fitness_function(features,sig=sigma,mu=mu)

            if update_sigma:
                if num_features==1:
                    maxf = features[np.argmax(fitness)]

                    if maxf == best_individual:

                        count_freeze+=1
                        if count_freeze%5==0 and generation>15:
                            count_freeze=0
                            sigma = sigma/2
                            if verbosity >= 1:
                                print("Updated sigma: ",sigma)

                    else:
                        best_individual = maxf


            if save_files:
                save_zipped_pickle(fitness,run_folder+f"/fitness_{generation}.p")


            if plot_progress:
                try:

                    make_ga_progess_plot(fitness,generation,num_features,features,fig,gs,num_generations,
                                         cmap=cmap,log_scale_plot=log_scale_plot,run_folder=run_folder)

                except KeyboardInterrupt:
                    break

            # WE END HERE!!!
            if generation == num_generations:
                break

            mutants = construct_mutants(population_size,dna_size, population, fitness,allele_bound,hyper_dict=hyper_dict)

            # We keep the best CYs from the previous population
            # To this end, we simply replace some of the mutants
            if num_survival_fittest is not None:

                mutants = perform_survival_fittest(population,mutants,fitness,num_survival_fittest)


            population = construct_next_generation_population(mutants,population_size,Qmax=Qmax)

            population = population[:population_size]
            
            history.append(population)

            if dynamical_pop_size:
                if population_size<max_population_size:
                    population_size = population_size*2
                    
    except KeyboardInterrupt: 
        return population,features,history
    except Exception as e:
        print("Something went wrong...") 
        print(e)
        return population,features,history
    return population,features,history



    
        
        


