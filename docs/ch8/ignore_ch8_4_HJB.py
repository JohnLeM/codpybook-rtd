"""
==========================
8.4 HJB
==========================
Let us introduce the Kernalized Non-Parametric Hamilton-Jacobi-Bellman algorithm. 
"""
# Importing necessary modules
import os
import sys
import time 

from matplotlib import pyplot as plt
import numpy as np

import codpy.core as core
import codpy.KQLearning as KQLearning
import codpy.conditioning as conditioning
from codpy.kernel import Kernel
#########################################################################
# Main Class
#########################################################################

#########################################################################
#The main HJB class can be found at :class:`codpy.KQLearning.KQLearningHJB`. 
#The main difference with KQLearning lies in the optimal_state_values_function. Here, it solves:
#$$Q^{\pi}(s_t,a_t) = R(s_t,a_t) + \gamma \int \left[ \sum_{a \in \mathcal{A}} \pi^a(s_t) Q^{\pi}(s',a)\right] d \mathbb{P}_S(s',s_t,a_t).$$
#Where the main difference lies in the presence of the transition probability matrix. 
#
#Numerically, we effectively solve for the set of parameters $\theta$ of the kernel $K$ such that:
#$$\theta = \Big( K(Z, Z) - \gamma \sum_{a} \pi^a(S)\Gamma(P^a) K(P, Z)\Big)^{-1} R, \quad P = \{ S+F_k(S,a), a \}$$
#
#Where: 
#
#* $K(Z,Z)$ is the kernel matrix of the states and actions.
#
#>>> Z = np.concatenate([states, actions], axis=1)
#>>> kernel = self.kernel_type(x=states_actions, fx=returns, **kwargs)
#>>> _knm_ = kernel.knm(x=states_actions, y=kernel.get_x())
#
#* $P$ is the set of the predicted next state actions possibilities
#
#>>> next_expected_states_actions = expectation_kernel_(states_actions)
#>>> next_expected_all_states_actions = self.all_states_actions(next_expected_states_actions)
#
#* $K(P,Z)$ is the kernel matrix of the predicted next state actions and the states and actions.
#
#>>> _projection_ = kernel.knm(x=next_expected_all_states_actions, y=kernel.get_x())
#>>> _projection_[modif] = 0.0
#
#Note here that we again handle the case of the terminal state Q-values.
#
#We then use the exact same process defined :ref:`kqlearning` to solve the Bellman equation using the :func:`optimal_bellman_solver`.
#
#>>> thetas, bellman_error, indices = self.optimal_bellman_solver(
#>>>     thetas=kernel.get_theta(),
#>>>     next_states_projection=_projection_,
#>>>     knm=_knm_,
#>>>     rewards=rewards,
#>>>     maxiter=10,
#>>>     games=games,
#>>>     **kwargs,
#>>> )
#
#The main difference being that the next_state_projection now contains the transition probability matrix. 


def get_conditioned_kernel(
    games,
    expectation_kernel,
    base_class=conditioning.PiKernel,
    **kwargs,
):
    class ConditionerKernel(base_class):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

    states, actions, next_states, rewards, returns, dones = games
    # Predict expected next states: s + F(s,a)
    expected_states = expectation_kernel(np.concatenate([states, actions], axis=1))

    params = kwargs.get("HJBModel", kwargs)
    # Form x = (s + F(s,a), a) tuples — this is the P set in K(P,Z)
    # Instantiate the kernel with:
    #   - x: expected transitions P = (s + F(s,a), a)
    #   - y: actual next states s'
    #   - expectation_kernel: the transition model
    out = ConditionerKernel(
        x=np.concatenate([expected_states, actions], axis=1),
        y=next_states,
        expectation_kernel=expectation_kernel,
        **params,
    )
    return out


def get_expectation_kernel(games, **kwargs):
    class expectation_kernel(Kernel):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.state_dim = kwargs.get("state_dim", None)

        def __call__(self, z, **kwargs):
            # Predict expected next state by: s + F(s,a)
            # Here, `super().__call__(z)` gives the kernel-based estimate of F(s,a)
            out = super().__call__(z) + z[:, : self.state_dim]
            return out

    states, actions, next_states, rewards, returns, dones = games
    states_actions = np.concatenate([states, actions], axis=1)
    params = kwargs.get("HJBModel", kwargs)
    # Set up kernel model on input (s,a), output: delta = s' - s
    # This models F(s,a) ≈ s' - s
    out = expectation_kernel(x=states_actions, fx=next_states - states, **params)
    # Compute noise: difference between true s' and predicted s + F(s,a)
    noise = next_states - out(states_actions)
    # Add average noise to model output
    out.set_fx(out.get_fx() + noise.mean(axis=0))
    return out

class KQLearningHJB(KQLearning.KQLearning):

    def optimal_states_values_function(
        self, games, kernel=None, full_output=False, maxiter=5, reorder=False, **kwargs
    ):

        states, actions, next_states, rewards, returns, dones = games
        states_actions = np.concatenate([states, actions], axis=1)
        if kernel is None or not kernel.is_valid():
            kernel = self.kernel_type(x=states_actions, fx=returns, **kwargs)

        expectation_kernel_ = self.get_expectation_kernel(games, **kwargs)
        conditioned_kernel_ = self.get_conditioned_kernel(
            games=games, expectation_kernel=expectation_kernel_, **kwargs
        )
        states_actions = np.concatenate([states, actions], axis=1)
        next_expected_states_actions = expectation_kernel_(states_actions)
        next_expected_all_states_actions = self.all_states_actions(
            next_expected_states_actions
        )
        _projection_ = kernel.knm(x=next_expected_all_states_actions, y=kernel.get_x())
        _knm_ = kernel.knm(x=states_actions, y=kernel.get_x())

        def helper(i):
            if dones[i] == True:
                return [i * self.actions_dim + j for j in range(self.actions_dim)]

        modif = [
            item
            for i in range(dones.shape[0])
            if dones[i] == True
            for item in helper(i)
        ]
        _projection_[modif] = 0.0

        thetas, bellman_error, indices = self.optimal_bellman_solver(
            thetas=kernel.get_theta(),
            next_states_projection=_projection_,
            knm=_knm_,
            rewards=rewards,
            maxiter=10,
            games=games,
            **kwargs,
        )

        #This is making a correction on the thetas
        thetas = conditioned_kernel_.get_transition(
            y=next_expected_all_states_actions[indices, : self.state_dim],
            x=np.concatenate([next_expected_states_actions, actions], axis=1),
            fx=thetas,
        )
        kernel.set_theta(thetas)
        kernel.bellman_error = bellman_error

        if full_output:
            return kernel, bellman_error, indices
        return kernel
    
#########################################################################
# Cartpole
# ------------------------

class KQLearningHJBCP(KQLearning.KQLearningHJB):

    def format(self, sarsd, max_training_game_size=None, **kwargs):
        states, actions, next_states, rewards, dones = [
            core.get_matrix(e) for e in sarsd
        ]

        actions = KQLearning.rl_hot_encoder(actions, self.actions_dim)
        returns = self.compute_returns(
            states, actions, next_states, rewards, dones, **kwargs
        )
        dones = core.get_matrix(dones, dtype=bool)
        if max_training_game_size is not None:
            states, actions, next_states, rewards, returns, dones = (
                states[:max_training_game_size],
                actions[:max_training_game_size],
                next_states[:max_training_game_size],
                rewards[:max_training_game_size],
                returns[:max_training_game_size],
                dones[:max_training_game_size],
            )

        return states, actions, next_states, rewards, returns, dones


    def train(self, game, max_training_game_size =sys.maxsize,tol=1e-4,**kwargs):
        states, actions, next_states, rewards, dones = game

        if len(states) >= kwargs.get("max_game", 1e12):
            print("no training")
            return
        states, actions, next_states, rewards, returns, dones = self.format(game, max_training_game_size=max_training_game_size,**kwargs)

        self.replay_buffer.push(states, actions, next_states, rewards, returns, dones)
        games = self.replay_buffer.memory
        states, actions, next_states, rewards, returns, dones = games
        if self.critic.is_valid(): #This function returns False if the kernel hasn't be properly initialized, i.e x and fx haven't been set.
            # We compute returns using the critic instead of MC returns.
            returns = self.critic(np.concatenate([states,actions],axis=1))
            games = states, actions, next_states, rewards, returns, dones
        
        self.critic = self.optimal_states_values_function(games,verbose=True,**kwargs)        
        return 
    
#########################################################################
# LunarLander
# -------------------------

class KQLearningHJBLN(KQLearning.KQLearningHJB):

    def __call__(self, state, **kwargs):
        self.eps_threshold *= 0.999
        if np.random.random() > self.eps_threshold and self.critic.is_valid() == True:
            z = self.all_states_actions(core.get_matrix(state).T)
            # z = self.all_states_actions(self.get_expectation_kernel(z))
            q_values = self.critic(z)
            q_values += np.random.random(q_values.shape) * 1e-9
            return np.argmax(q_values)
        return np.random.randint(0, self.actions_dim)

    def get_conditioned_kernel(self, games, **kwargs):
        return KQLearning.get_conditioned_kernel(
            games, base_class=conditioning.ConditionerKernel, **kwargs
        )

    def train(
        self,
        game,
        max_training_game_size=None,
        format=True,
        replay_buffer=True,
        tol=1e-2,
        **kwargs
    ):
        # return super().train(game, max_training_game_size,format,replay_buffer, tol,**kwargs)
        # l = len(game[0])
        # self.gamma = np.exp(-np.log(l) / l)
        game = self.format(
            game, max_training_game_size=max_training_game_size, **kwargs
        )
        kernel = self.optimal_states_values_function(game, verbose=True, **kwargs)
        kernel.games = game
        # kernel.gamma = self.gamma
        self.critic.add_kernel(kernel, **kwargs)
        delete_kernels = []
        for i, k in self.critic.kernels.items():
            # self.gamma = k.gamma
            error = self.critic.kernels[i].bellman_error
            if error > tol and not hasattr(self.critic.kernels[i], "flag_kill_me"):
                kernel = self.optimal_states_values_function(
                    self.critic.kernels[i].games,
                    kernel=self.critic.kernels[i],
                    verbose=True,
                    **kwargs,
                )
                kernel.games = self.critic.kernels[i].games
                # kernel.gamma = self.critic.kernels[i].gamma
                if kernel.bellman_error >= error - tol:
                    # delete_kernels.append(i)
                    kernel.flag_kill_me = "please"
                else:
                    self.critic.kernels[i] = kernel
        if (
            len(delete_kernels) > 0
            and len(self.critic.kernels) - len(delete_kernels) > 1
        ):
            new_kernels = {}
            count = 0
            for i in range(len(self.critic.kernels)):
                if i not in delete_kernels:
                    new_kernels[count] = self.critic.kernels[i]
                    count = count + 1
            self.critic.kernels = new_kernels