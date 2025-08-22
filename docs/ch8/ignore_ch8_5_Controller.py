"""
==========================
8.5 Controller
==========================
Let us introduce the Heuristic-controlled Kernel algorithm.  
"""
# Importing necessary modules
import os
import sys
import time 

from matplotlib import pyplot as plt
import numpy as np

import codpy.core as core
import codpy.KQLearning as KQLearning
from codpy.core import get_matrix
import codpy.optimization as optimization
#########################################################################
# Main Class
#########################################################################

#########################################################################
#The main Controlelr class can be found at :class:`codpy.KQLearning.KController`. 
#
#The specificities of this algorithm is that it uses a heuristic controller to be tuned. Therefore, we can see in the call function that it directly outputs the value of the controller. 
#
#The parameters $\theta$ of the controller are found at :func:`self.controller.get_thetas()`. 
#
#The algorithm is solving for: 
#
#$$\theta_{n+1} = \arg \max_{\theta \in \Theta_n} \mathcal{L}(R_{k,\lambda_e},\theta), \quad \Theta_{n} = \bar{\theta_e} \cup \Theta_{N,n}$$
#where $\Theta_{N,n}$ is a screening around the last $\theta_n$ and is defined as follow:
#
#$$\Theta_{N,n} = (\theta_n+\alpha^n \Theta_N) \cap \Theta$$
#And ${L}(R_{k,\lambda_e},\theta)$ is an optimization function which can be defined and tuned based on your needs. 
#
#We get the last controller parameters: 
#
#>>> last_theta = get_matrix(self.controller.get_thetas()).T
#
#And the rewards: 
#
#>>> reward = get_matrix(rewards)
#
#${L}(R_{k,\lambda_e},\theta)$ is defined in the :func:`get_function`: 
#
#>>> def function(x):
#>>>    expectation = self.expectation_estimator(
#>>>      x
#>>>    )  
#>>>    distance = self.expectation_estimator.distance(x)
#>>>    return expectation + distance
#
#We sample on the controller distribution with 
#
#>>> self.controller.get_distribution()
#
#And the optimization code, called here: 
#
#>>> max_val, new_theta = optimization.continuous_optimizer(
#>>>    function,
#>>>    self.controller.get_distribution(),
#>>>    include=self.x,
#>>>    **kwargs,
#>>> )
#
#$\Theta_{N,n} = (\theta_n+\alpha^n \Theta_N) \cap \Theta$ is built in the :class:`contract_distrib` :func:`__call__` method:
#
#>>> class contract_distrib:
#>>>     def __init__(self, contracting_factor, mean):
#>>>         self.contracting_factor = contracting_factor
#>>>         self.mean = mean
#>>> 
#>>>     def __call__(self, n, *args, **kwds):
#>>>         samples = distribution(n)
#>>>         if self.mean is not None:
#>>>             samples = (samples - samples.mean()) * self.contracting_factor + self.mean
#>>>             samples = distribution.support(samples)
#>>>         return samples
#
#This make a distribution centered around $\theta_n$. 
#The function to be optimized is called in the :func:`screen_optimize`
#
#>>> y = function(samples).flatten()
#
#We then return the extremum thetas, and we keep iterating. 

class KController(KQLearning.KAgent):
    def __init__(self, state_dim, actions_dim, controller, **kwargs):
        self.controller = controller
        self.x, self.y = None, None
        self.expectation_estimator = None
        self.label = kwargs.get("label", None)
        super().__init__(state_dim=state_dim, actions_dim=actions_dim, **kwargs)

    def __call__(self, z, **kwargs):
        return self.controller(z, **kwargs)
    
    def get_function(self, **kwargs):
        self.expectation_estimator = self.get_expectation_estimator(
            self.x, self.y, **kwargs
        )
        def function(x):
            expectation = self.expectation_estimator(
                x
            )  
            distance = self.expectation_estimator.distance(x)
            return expectation + distance

        return function
    
    def get_expectation_estimator(self, x, y, **kwargs):
        class explore_kernel(KQLearning.Kernel):
            def distance(self, z, **kwargs):
                out = get_matrix(self.dnm(x=self.get_x(), y=z).min(axis=0))
                return out

        params = kwargs.get("KController", {})
        self.expectation_kernel = explore_kernel(x=x, fx=y, **params)
        return self.expectation_kernel
    
    def train(self, game, **kwargs):
        states, actions, next_states, rewards, dones = self.format(game, **kwargs)
        self.replay_buffer.push(states, actions, next_states, rewards, dones, **kwargs)
        reward = get_matrix(rewards)
        last_theta = get_matrix(self.controller.get_thetas()).T
        if self.x is None:
            self.x = get_matrix(last_theta)
            self.y = get_matrix(reward)
        else:
            if (
                self.expectation_estimator is None
                or self.expectation_estimator.distance(last_theta) > 1e-9
            ):
                self.x = np.concatenate([self.x, last_theta])
                self.y = np.concatenate([self.y, reward])

        if self.x.shape[0] > 2:
            function = self.get_function(**kwargs)
            last_vals = function(self.x)
            last_val = last_vals.max()
            last_val_max_min = last_val - last_vals.min()
            max_val, new_theta = optimization.continuous_optimizer(
                function,
                self.controller.get_distribution(),
                include=self.x,
                **kwargs,
            )
            new_theta = get_matrix(new_theta).T
            if last_val < max_val:  # debug to see if parameters are updated
                tot = 1
                pass
            self.controller.set_thetas(new_theta)
        else:
            self.controller.set_thetas(self.controller.get_distribution()(1))

#########################################################################
# Optimizer
# ------------------------
def screen_optimize(function, distribution, n=1000, maximize=True):
    if n == 0:
        return None
    samples = distribution(n)
    y = function(samples).flatten()

    if maximize:
        y_extrem_idx = y.argmax()
    else:
        y_extrem_idx = y.argmin()

    extremum = y[y_extrem_idx], samples[y_extrem_idx]

    return extremum


def continuous_optimizer(
    function,
    distribution,
    n=1000,
    n_iter=5,
    contracting_factor=0.3,
    maximize=True,
    include=None,
    **kwargs
):
    if maximize == False:
        return continuous_optimizer(
            lambda x: -function(x) - function.distance(x),
            distribution,
            n,
            n_iter,
            contracting_factor,
            True,
            **kwargs
        )

    class contract_distrib:
        def __init__(self, contracting_factor, mean):
            self.contracting_factor = contracting_factor
            self.mean = mean

        def __call__(self, n, *args, **kwds):
            samples = distribution(n)
            if self.mean is not None:
                samples = (
                    samples - samples.mean()
                ) * self.contracting_factor + self.mean
                samples = distribution.support(samples)
            return samples

    mean = None
    cf = 1.0
    extremum_y, extremum_x = -sys.float_info.max, None
    if include is not None:
        values = function(include)
        extremum_y, extremum_x = values.max(), include[values.argmax()]

    for i in range(min(n, n_iter)):
        temp_y, temp_x = screen_optimize(
            function, contract_distrib(cf, mean), n, maximize
        )
        if temp_y > extremum_y:
            extremum_y = temp_y
            extremum_x = temp_x
        mean = extremum_x
        cf *= contracting_factor

    return extremum_y, extremum_x

#########################################################################
# Cartpole
# ------------------------

class heuristic_ControllerCP:
    """
    This class defines an expert-based heuristic controller for the CartPole environment.
    """
    # This is the number of parameters to be optimized
    dim = 4

    def __init__(self, w=None, **kwargs):
        if w is None:
            self.w = np.ones([self.dim]) * 0.5
        else:
            self.w = w
        pass

    def get_distribution(self):
        """
        This will be called by the optimizer. You need to define a way to sample from the parameters distribution, and get the support. 
        """
        class uniform:
            def __init__(self, shape1):
                self.shape1 = shape1

            def __call__(self, n):
                return 2 * np.random.uniform(size=[n, self.shape1]) - 1

            def support(self, v):
                return v

        return uniform(self.w.shape[0])

    def get_thetas(self):
        return self.w

    def set_thetas(self, w):
        self.w = w.flatten()

    def __call__(self, s, **kwargs):
        """
        Will be used to make inference. This is where you define the action to be taken. 

        Parameters: 
        - s : state of the environment, a numpy array of shape (n, state_dim).

        Returns: 
        - prod: int, action to be taken
        """
        prod = (self.w * s).sum()
        prod = int((np.sign(prod) + 1) / 2)
        return prod
    
class KControllerCP(KQLearning.KController):
    """
    This is the main class which will optimize the heuristic controller. 
    """
    def __init__(self, state_dim, actions_dim, **kwargs):
        # This is where you would pass any other custom controller
        controller = heuristic_ControllerCP(state_dim=state_dim, **kwargs)
        super().__init__(state_dim, actions_dim, controller, **kwargs)

    def get_function(self, **kwargs):
        """
        The optimizer will find the best parameters which maximizes this function. 

        This is where you would tweak the function to be maximized.
        """
        self.expectation_estimator = self.get_expectation_estimator(self.x, self.y, **kwargs)
        def function(x):
            expectation = self.expectation_estimator(x)
            distance = self.expectation_estimator.distance(x)
            return expectation * distance
        return function 


    def format(self, sarsd, **kwargs):
        """
        In the case of the controller, the agent only sees the sum of the rewards for an entire episode. 
        All other game data won't be used for training. The format function still need to output a tuple. 
        """
        state, action, next_state, reward, done = [
            core.get_matrix(e) for e in sarsd
        ]
        reward[done.astype(bool)] = 0

        action = KQLearning.rl_hot_encoder(action, self.actions_dim)
        action = core.get_matrix(self.controller.get_thetas()).T
        done = core.get_matrix(done, dtype=bool)
        return (
            core.get_matrix(state.mean(axis=0)).T,
            core.get_matrix(action.mean(axis=0)).T,
            core.get_matrix(next_state.mean(axis=0)).T,
            core.get_matrix(reward.sum(axis=0)).T,
            core.get_matrix(done.mean(axis=0)).T,
        )

    def train(self, game, **kwargs):
        # Similarily, you can skip training if the game is too long to save training time.
        states, actions, next_states, rewards, dones = game
        if len(states) >= kwargs.get("max_game", 1e12):
            print("no training")
            return
        super().train(game, **kwargs)

#########################################################################
# LunarLander
# ------------------------
# For this we need a heuristic function taking parameters as input and returning actions. Action dims needs to match with the environment.  

class heuristic_ControllerLN:
    """
    Defines the heuristic controller for LunarLander. We choose to use 12 parameters to be tweaked. 
    """
    dim = 12

    def __init__(self, w=None, **kwargs):
        if w is None:
            self.w = np.ones([self.dim]) * 0.5
        else:
            self.w = w
        pass

    def get_distribution(self):
        class uniform:
            def __init__(self, shape1):
                self.shape1 = shape1

            def __call__(self, n):
                return np.random.uniform(size=[n, self.shape1])

            def support(self, v):
                out = np.clip(v, 0, 1)
                return out

        return uniform(self.w.shape[0])

    def get_thetas(self):
        return self.w

    def set_thetas(self, w):
        self.w = w.flatten()

    def __call__(self, s, **kwargs):
        angle_targ = s[0] * self.w[0] + s[2] * self.w[1]
        if angle_targ > self.w[2]:
            angle_targ = self.w[2]
        if angle_targ < -self.w[2]:
            angle_targ = -self.w[2]
        hover_targ = self.w[3] * np.abs(s[0])

        angle_todo = (angle_targ - s[4]) * self.w[4] - (s[5]) * self.w[5]
        hover_todo = (hover_targ - s[1]) * self.w[6] - (s[3]) * self.w[7]

        if s[6] or s[7]:
            angle_todo = self.w[8]
            hover_todo = -(s[3]) * self.w[9]

        a = 0
        if hover_todo > np.abs(angle_todo) and hover_todo > self.w[10]:
            a = 2
        elif angle_todo < -self.w[11]:
            a = 3
        elif angle_todo > +self.w[11]:
            a = 1
        return a
    

class KControllerLN(KQLearning.KController):
    """
    Defines the class for optimizing the controller. 

    The class inherit from KQLearning.KController. You can then extend any method from the main class to fit your needs. 

    Parameters:
    - state_dim: Dimension of the environment's state space.
    - actions_dim: Dimension of the environment's action space.
    """
    def __init__(self, state_dim, actions_dim, **kwargs):
        controller = heuristic_ControllerLN(state_dim=state_dim, **kwargs)
        super().__init__(state_dim, actions_dim, controller, **kwargs)

    def get_function(self, **kwargs):
        """
        The optimizer will find the best parameters which maximizes this function. 

        This is where you would tweak the function to be maximized.
        """
        self.expectation_estimator = self.get_expectation_estimator(
            self.x, self.y, **kwargs
        )

        def function(x):
            expectation = self.expectation_estimator(x)
            distance = self.expectation_estimator.distance(x)
            return expectation + distance

        return function 

    def format(self, sarsd, **kwargs):
        """
        This formats the game data to be used in the train method
        """
        state, action, next_state, reward, done = [core.get_matrix(e) for e in sarsd]

        action = KQLearning.rl_hot_encoder(action, self.actions_dim)
        action = core.get_matrix(self.controller.get_thetas()).T
        done = core.get_matrix(done, dtype=bool)
        return (
            core.get_matrix(state.mean(axis=0)).T,
            core.get_matrix(action.mean(axis=0)).T,
            core.get_matrix(next_state.mean(axis=0)).T,
            core.get_matrix(reward.mean(axis=0)).T,
            core.get_matrix(done.mean(axis=0)).T,
        )