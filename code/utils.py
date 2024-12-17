# ===================================================================================
# Copyright 2024 Stefano Lanza, and Andreas Schachner
#
#   This script provides utility functions.
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

import pickle
import gzip

import jax

from tqdm.auto import tqdm


def load_zipped_pickle(filen):
    r"""
    
    **Description:**
    Returns content of zipped pickle file.
    
    
    Args:
       filen (string): Filename of zipped file to be read.
        
    Returns:
       data (array/dictionary): Data contained in file.
    
    """
    
    with gzip.open(filen, 'rb') as f:
        loaded_object = pickle.load(f)
            
    f.close()
            
    return loaded_object



def save_zipped_pickle(obj, filen, protocol=-1):
    r"""
    
    **Description:**
    Saves data in a zipped pickle file.
    
    
    Args:
       obj (array/dict): Data to be stored in file.
       filen (string): Filename of file to be read.
        
    Returns:
        
    
    """
    with gzip.open(filen, 'wb') as f:
        pickle.dump(obj, f, protocol)
        
    f.close()
    
    
# Set random seed for reproducible results
class PRNGSequence:
    r"""
    **Description:**
    PRNG sequence.
    
    Args:
        (): .
        
    Returns:
        (): .
    
    """
    
    _key = None

    def __init__(self, seed: 42):
        r"""
        Random key sequence.

        Use as follows:
        >>> rns = PRNGSequence(42)
        >>> key = next(rns)
        """
        if isinstance(seed, int):
            self._key = jax.random.PRNGKey(seed)
        else:
            self._key = seed

    def __next__(self):
        r"""Get the next random key."""
        k, self._key = jax.random.split(self._key)
        return k


def sort_lists_together(list1, list2):
    r"""
    **Description:**
    Utility function for sorting lists together
    
    Args:
        list1 (list): First list.
        list2 (list): Second list.
        
    Returns:
        list: Sorted first list.
        list: Sorted second list.
        
    """
    
    # Check that the lists are the same length
    if len(list1) != len(list2):
        raise ValueError("Both lists should have the same length")

    # Create a list of indices sorted by the corresponding value in list1
    sorted_indices = sorted(range(len(list1)), key=lambda i: list1[i])

    # Use the sorted list of indices to reorder both lists
    sorted_list1 = [list1[i] for i in sorted_indices]
    sorted_list2 = [list2[i] for i in sorted_indices]

    return sorted_list1, sorted_list2

def group_tuples_by_index(tuples, index):
    r"""
    **Description:**
    This function groups a list of tuples by the value of the element at the given index.
    We need this to track the effects of different hyperparameters with the other ones held constant.
    
    Args:
        tuples (tuple): Tuple of hyperparameter names.
        index (list): Indices.
        
    Returns:
        dict: Dictionary.
    """
    
    # Initialize an empty dictionary to store the groups
    groups = {}

    # Iterate over each tuple in the input list
    for t in tuples:
        # Create a copy of the tuple as a list so we can modify it
        mutable_t = list(t)
        
        # Remove the element at the given index
        del mutable_t[index]

        # Convert the list back into a tuple to use it as a dictionary key
        key = tuple(mutable_t)

        # If this key is not yet in the dictionary, add an empty list
        if key not in groups:
            groups[key] = []

        # Append the current tuple to the list for this key
        groups[key].append(t)

    # Return the groups as a list of lists
    dep_list = []
    for i in groups.keys():
        dep_list.append(groups[i])
    return groups




