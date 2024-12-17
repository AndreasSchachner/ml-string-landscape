# ===================================================================================
# Copyright 2024 Stefano Lanza, and Andreas Schachner
#
#   This script provides code to construct NNs using `flax`.
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
import sys, os, warnings
import numpy as np
from tqdm.auto import tqdm

# plotting
# --------
import seaborn as sn
import matplotlib.pyplot as plt
import matplotlib as mpl
from IPython.display import clear_output
cmap=sn.color_palette("viridis", as_cmap=True)

# jax
# --------
from jax import jit, vmap
import jax 
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from jax import Array
from functools import partial

# optax
# --------
import optax

# flax
# --------   
import flax
from flax import linen as nn
from flax.training import checkpoints
from flax.training import train_state

# typing
# --------
from typing import Any, Callable, Sequence, Tuple, Generator
from jax.typing import ArrayLike

# custom
# --------
from utils import PRNGSequence



class nn_model(nn.Module):
    r"""
    **Description:**
    NN model.
    
    Args:
        (): .
        
    Returns:
        (): .
    
    """
    
    features: Sequence[int]
    activation: str
    activations: Sequence[int]

    def setup(self):
        # we automatically know what to do with lists, dicts of submodules
        self.layers = [nn.Dense(feat) for feat in self.features]
        # for single submodules, we would just write:
        # self.layer1 = nn.Dense(feat1)
        
        if self.activation not in ["tanh","relu","sigmoid"]:
            raise ValueError("Could not determine activiation input!")
            
        

    def __call__(self, inputs):
        x = inputs
        for i, lyr in enumerate(self.layers):
            x = lyr(x)
            if i != len(self.layers) - 1:
                
                if self.activations is None:
                    act = self.activation
                else:
                    act = self.activations[i]
                
                if act=="tanh":
                    x = nn.tanh(x)
                elif act=="relu":
                    x = nn.relu(x) 
                elif act=="sigmoid":
                    x = nn.sigmoid(x) 
            
        return x
    

class train_state_model():
    
    def __init__(self,state):
        
        self.state = state
        
    def predict(self,data):
        
        return self.state.apply_fn(self.state.params,data)
    
    


def random_samples(
                   a: ArrayLike, 
                   rns_key=None,
                   seed: int=42,
                   shape: Tuple[int,int]=(1,)
                   ) -> Array:
    r"""
    **Description:**
    Generates random samples from given input Array.
    
    Args:
        a (Array): Input array.
        rns_key (PNRGSequence, optional): Key chain for random number generation.
        seed (int, optional): Seed for random number generation.
        shape (tuple, optional): Shape to random sample from.
        
    Returns:
        Array: Random sample.
    
    """

    if rns_key is None:
        rns_key = PRNGSequence(seed)
        
    if np.prod(shape)>len(a):
        raise ValueError("Shape incompatible for taking random samples!")
        
    return jax.random.choice(next(rns_key),a,shape=shape,replace=False)


def get_batches(
                input_data: ArrayLike, 
                output_data: ArrayLike, 
                batch_size: int = 16, 
                rns_key=None, 
                seed: int = 42
                ) -> Generator[ArrayLike, None, None]:
    r"""
    **Description:**
    Splits input into batches of given size.
    
    Args:
        input_data (): .
        output_data (): .
        batch_size (int, optional): Batch size.
        rns_key (PNRGSequence, optional): Key chain for random number generation.
        seed (int, optional): Seed for random number generation.
        
    Returns:
        (): .
    
    """
    
    # Random sequence generator  
    if rns_key is None:
        rns_key = PRNGSequence(seed)
    
    # Generate indices
    indices = jnp.arange(0,len(input_data))
    
    indices = jax.random.permutation(next(rns_key),indices,independent=True)
    
    num_batches = np.floor(input_data.shape[0]/batch_size).astype(int)
    
    indices_batched = random_samples(indices,shape=(num_batches,batch_size))
    
    for ii in range(num_batches):
        yield input_data[indices_batched[ii]],output_data[indices_batched[ii]]
        



def train_model(
                state,
                data: Tuple[ArrayLike,ArrayLike],
                labels: Tuple[ArrayLike,ArrayLike],
                model = None,
                optimizer: str = "adam",
                batch_size: int = 16,
                num_epochs: int = 100,
                learning_rate: float = 1e-3,
                learning_rate_schedule: list = None,
                schedule: list = [],
                make_plot: bool = False,
                print_progress: bool = False,
                rns_key=None
                ) -> Tuple[object,Array,Array]:
    r"""
    **Description:**
    Trains NN model for given input and output data.
    
    Args:
        state (): .
        data (): .
        labels (): .
        model (, optional): .
        optimizer (,optional): .
        batch_size (int, optional): Batch size.
        num_epochs (int, optional): .
        learning_rate (float, optional): .
        learning_rate_schedule (list, optional): .
        schedule (list, optional): .
        make_plot (boolean, optional): .
        print_progress (boolean, optional): .
        rns_key (PNRGSequence, optional): Key chain for random number generation.
        
    Returns:
        state, losses, test_losses: .
    
    """
    
    if learning_rate_schedule is not None:
        if len(learning_rate_schedule)!=len(schedule):
            raise ValueError(f"Learning rate updates and schedule should have same length, but have {len(learning_rate_schedule)} and {len(schedule)} respectively.")
    
    if len(data)!=2:
        raise ValueError(f"Input data needs to be 2-dimensional (tain and test data), but have {len(data)}.")
        
    if len(labels)!=2:
        raise ValueError(f"Labels input needs to be 2-dimensional (tain and test labels), but have {len(labels)}.")
    
    train_data, test_data = data
    train_label, test_label = labels
    
    try:
        
        losses = []
        test_losses = []
        cc=0
        if print_progress:
            ranger = tqdm(range(num_epochs))
        else:
            ranger = range(num_epochs)
            
        # Training loop
        for epoch in ranger:

            # Updating learning rate
            if epoch in schedule:
                
                if learning_rate_schedule is None:
                    learning_rate = learning_rate/2
                else:
                    
                    if cc>len(learning_rate_schedule)-1:
                        raise ValueError(f"Failed to update learning rate. Counter value exceeded number of scheduled updates {len(learning_rate_schedule)}.")
                        
                    learning_rate = learning_rate_schedule[cc]
                    cc+=1
                    
                if optimizer=="adam":
                    optimizer_fct = optax.adam(learning_rate=learning_rate)
                elif optimizer=="sgd":
                    optimizer_fct = optax.sgd(learning_rate=learning_rate)
                else:
                    raise ValueError("Could not determine optimizer!")
                state = train_state.TrainState.create(apply_fn=model.apply,params=state.params,tx=optimizer_fct)
                
            
            data_loader = get_batches(train_data,train_label,batch_size=batch_size,rns_key=rns_key)
            
            for b in  data_loader:
                state, loss = train_step(state, b[0],b[1])
                
            loss = eval_step(state, train_data,train_label)
            losses.append(loss)
            
            
            tloss = eval_step(state, test_data,test_label)
            test_losses.append(tloss)
            
            
            print(f"Epoch: {epoch}       Train loss: {loss}         Test loss: {tloss}                 ",flush=True,end="\r")
            
            if make_plot:

                #if (epoch%10==0 and epoch>0 and epoch<100) or (epoch%100==0 and epoch>0) or (epoch <10 and epoch>0):
                if (epoch>0 and epoch%10==0):
                    clear_output(wait=True)
                    fig = plt.figure(dpi=100)
                    sn.lineplot(x = np.arange(0,len(losses)),y = np.array(losses),lw=4,label="train")
                    sn.lineplot(x = np.arange(0,len(test_losses)),y = np.array(test_losses),lw=4,label="test")
                    plt.ylabel("loss")
                    plt.xlabel("epoch")
                    plt.yscale("log")
                    plt.show();
            
                
    except KeyboardInterrupt:
        return state, losses, test_losses
    except Exception as e: print(e)
        
    return state, losses, test_losses


def flux_vacua_model(
                     data: Tuple[ArrayLike,ArrayLike],
                     labels: Tuple[ArrayLike,ArrayLike],
                     epochs: int = 1000,
                     batch_size: int = 16,
                     learning_rate: float = 1e-2,
                     optimizer: str = "adam",
                     num_layers: int = 1,
                     num_outputs: int = 2, 
                     layer_size: int = 8,
                     activation: str = "tanh",
                     make_plot: bool = False,
                     learning_rate_schedule: list = None, 
                     schedule: list = [],
                     input_size: int = 1,
                     print_progress: bool = False,
                     rns_key = None,
                     model_state = None,
                     features=None,
                     activations=None
                     ) -> Tuple[object,Array,Array]:
    r"""
    **Description:**
    Configures NN model and starts training.
    
    Args:
        (): .
        
    Returns:
        state (): .
        losses (): .
        test_losses (): .
    
    """
    
    if optimizer=="adam":
        optimizer_fct = optax.adam(learning_rate=learning_rate)
    elif optimizer=="sgd":
        optimizer_fct = optax.sgd(learning_rate=learning_rate)
    else:
        raise ValueError("Could not determine optimizer!")
    
    
    if features is None:
        features = []
        features += [layer_size]*num_layers
        
    features.append(num_outputs)
    
    model = nn_model(features = features,activation = activation,activations = activations)

    if model_state is None:
        
        rng = jax.random.PRNGKey(42)#next(rns_key)
        rng, inp_rng, init_rng = jax.random.split(rng, 3)
        inp = jax.random.normal(inp_rng, (batch_size, input_size))  # shape = (batch size, input size)
        # Initialize the model
        params = model.init(init_rng, inp)

        model_state = train_state.TrainState.create(apply_fn=model.apply,params=params,tx=optimizer_fct)

    
    return train_model(model_state, data, labels, batch_size=batch_size, num_epochs=epochs, learning_rate=learning_rate, make_plot=make_plot, optimizer = optimizer,
                       learning_rate_schedule=learning_rate_schedule, schedule=schedule, print_progress=print_progress, rns_key=rns_key,model=model)



    
@partial(jit, static_argnums = ())
def predictions(
                state: object, 
                params: dict, 
                input_data: ArrayLike
                ) -> Array:
    r"""
    **Description:**
    Computes the predictions for given NN state.
    
    Args:
        state (object): Training state.
        params (dict/pytree): Dictionary/pytree containing the values of hyperparameters.
        input_data (Array): Input data.
        
    Returns:
        ArrayLike: Predictions of NN.
    
    """
    
    return state.apply_fn(params,input_data)

@partial(jit, static_argnums = ())
def predictions_vmap(
                     state: object, 
                     params: dict, 
                     input_data: ArrayLike
                     ) -> Array:
    r"""
    **Description:**
    Vmapped version of predictions.
    
    Args:
        state (object): Training state.
        params (dict/pytree): Dictionary/pytree containing the values of hyperparameters.
        input_data (Array): Input data.
        
    Returns:
        ArrayLike: Vmapped predictions of NN.
    
    """
    
    return vmap(predictions,in_axes=(None,None,0))(state,params,input_data)
    
    
@partial(jit, static_argnums = ())
def compute_loss(
                 state: object, 
                 params: dict, 
                 input_data: ArrayLike, 
                 output_data: ArrayLike
                 ) -> float:
    r"""
    **Description:**
    Computes loss.
    
    Args:
        state (object): Training state.
        params (dict/pytree): Dictionary/pytree containing the values of hyperparameters.
        input_data (Array): Input data.
        output_data (Array): Output data.
        
    Returns:
        float: Loss.
    
    """
    
    values = predictions_vmap(state, params, input_data)
    
    # Compute phi^r
    r = values-output_data
    
    return jnp.mean(r**2)


@partial(jit, static_argnums = ())
def grad_loss(
              state: object, 
              params: dict, 
              input_data: ArrayLike, 
              output_data: ArrayLike
              ) -> Array:
    r"""
    **Description:**
    Computes gradient of loss function.
    
    Args:
        state (object): Training state.
        params (dict/pytree): Dictionary/pytree containing the values of hyperparameters.
        input_data (Array): Input data.
        output_data (Array): Output data.
        
    Returns:
        ArrayLike: Gradient of loss wrt network parameters.
    
    """
    
    return jax.grad(compute_loss,argnums=1)(state, params, input_data,output_data)
    
    

@partial(jit, static_argnums = ())
def get_grad(
             state: object, 
             params: dict, 
             input_data: ArrayLike, 
             output_data: ArrayLike
             ) -> float:
    r"""
    **Description:**
    Returns loss and gradient of loss wrt to NN parameters.
    
    Args:
        state (object): Training state.
        params (dict/pytree): Dictionary/pytree containing the values of hyperparameters.
        input_data (Array): Input data.
        output_data (Array): Output data.
        
    Returns:
        float: Loss.
        ArrayLike: Gradient of loss wrt network parameters.
    
    """
    
    return compute_loss(state, params, input_data,output_data), grad_loss(state, params, input_data,output_data)


@partial(jit, static_argnums = ())
def train_step(
               state: object, 
               input_data: ArrayLike, 
               output_data: ArrayLike
               ) -> Tuple[object,float]:
    r"""
    **Description:**
    Performs training step.
    
    Args:
        state (object): Training state.
        input_data (Array): Input data.
        output_data (Array): Output data.
        
    Returns:
        object: Training state.
        float: Loss.
    
    """
    # Gradient function
    loss, grad_loss = get_grad(state, state.params, input_data,output_data)
    
    # Perform parameter update with gradients and optimizer
    state = state.apply_gradients(grads=grad_loss)
    
    # Return state and any other value we might want
    return state, loss

@partial(jit, static_argnums = ())
def eval_step(
              state: object, 
              input_data: ArrayLike, 
              output_data: ArrayLike
              ) -> float:
    r"""
    **Description:**
    Evaluates the loss at a given step.
    
    Args:
        state (object): Training state.
        input_data (Array): Input data.
        output_data (Array): Output data.
        
    Returns:
        float: Loss.
    
    """
    
    # Determine the accuracy
    return compute_loss(state, state.params, input_data,output_data)
    