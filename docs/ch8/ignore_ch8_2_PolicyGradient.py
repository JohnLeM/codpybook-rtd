"""
==========================
8.2 Policy Gradient
==========================
Let us introduce the Kernel-Bases Q-value gradient estimator
"""
# Importing necessary modules
import os
import sys
import time 

from matplotlib import pyplot as plt
import numpy as np

import codpy.core as core
import codpy.KQLearning as KQLearning
from codpy.lalg import LAlg
from codpy.kernel import get_tensor_probas, Kernel
#########################################################################
# Main Class
#########################################################################

#########################################################################
#The main PolicyGradient class can be found at :class:`codpy.KQLearning.PolicyGradient`. 
#It uses a :class:`GamesKernelClassifier` as the actor, which is a specific type of kernel used for stochastic policies, and outputs probabilities over action_dim.
#The :func:`get_advantage` method is what is used in the train method to solve for 
# $$A^{\pi}(s) = \nabla_{y} Q^\pi_k(\cdot) = K(\cdot, Z) \nabla_{y} \theta^\pi.$$
#Where:
# $$\nabla_{y} \theta^\pi = \gamma \Big(K(Z,Z) - \gamma \sum_a \pi^a(S) K(W,Z) \Big)^{-1} \sum_a\Big(Q^{\pi}_k(W) \pi^a(\delta_b(a)-\pi^b)\Big)$$
#
#$Z$ are the state actions: 
#
#>>> Z = np.concatenate([states, actions], axis=1)
# 
#$W$ is the next state actions:	
#
#>>> W = self.all_states_actions(next_states)
#
#$K(\cdot, Z)$ is the kernel between new samples and training points: 
#
#>>> derivative_estimator = Kernel()
#>>> derivative_estimator.set_x(states_actions)
#
#$K(Z,Z)$ is the Gram matrix of the training points:
#
#>>> knm = value_function.knm(x=states_actions, y=states_actions)
#
#$\gamma \sum_a \pi^a(S) K(W,Z)$ is the weighted projection operator onto the next state-actions. We reshape it to dinstinguish actions, and use :func:`np.einsum` to compute the sum over actions:
#
#>>> projection_op = value_function.knm(x=next_states_actions, y=states_actions)
#>>> projection_op = projection_op.reshape([states_actions.shape[0], self.actions_dim, states_actions.shape[0]])
#>>> projection_op[[bool(d) for d in dones]] = 0.0
#>>> sum_policy = np.einsum("...ji,...j", projection_op, policy)
#
#We set the projection to 0 for the terminal states, so the Q value is actually the reward, as per the definition. 
#
#We then have the $\Big(K(Z,Z) - \gamma \sum_a \pi^a(S) K(W,Z) \Big)^{-1}$ finally computed with
#
#>>> projection_op = LAlg.lstsq(knm - sum_policy * self.gamma)
#
#$Q^{\pi}_k(W)$ is the critic evaluated at :data:`next_states_actions`:
#
#>>> next_states_actions_values = value_function(next_states_actions).reshape([states_actions.shape[0], self.actions_dim])
#
#$\pi^a(\delta_b(a)-\pi^b)$ is an adjustment based on probability differences. In code, this is: 
#
#>>> coeffs = get_tensor_probas(policy)
#>>> second_member = np.einsum("...i,...ij", next_states_actions_values, coeffs)
#
#Therefore, we have :data:`second_member` to be $\sum_a\Big(Q^{\pi}_k(W) \pi^a(\delta_b(a)-\pi^b)\Big)$.
#
#We then combine all to solve for $\nabla_{y} \theta^\pi$:
#
#>>> derivative_estimator.set_theta(LAlg.prod(projection_op, second_member))

class PolicyGradient(KQLearning.KActorCritic):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        params = kwargs.get("KCritic", {})
        self.actor = KQLearning.GamesKernelClassifier(**params)

    def __call__(self, state, **kwargs):
        if self.actor.get_x() is not None and self.actor.get_x().shape[0] > 1:
            action_probs = self.actor(core.get_matrix(state).T)
            action_probs = action_probs.squeeze()
            action = np.random.choice(len(action_probs), p=action_probs)
            # action = action_probs.argmax()
            return action
        else:
            return np.random.randint(0, self.actions_dim)

    def get_advantages(self, games, policy, **kwargs):
        states, actions, next_states, rewards, returns, dones = games
        derivative_estimator = self.get_derivatives_policy_state_action_value_function(
            games, policy, **kwargs
        )
        states_actions = np.concatenate([states, actions], axis=1)
        derivative_estimations = derivative_estimator(states_actions)
        return derivative_estimations, derivative_estimator
    
    def get_derivatives_policy_state_action_value_function(
        self, games, policy, output_value_function=False, **kwargs
    ):
        states, actions, next_states, rewards, returns, dones = games
        states_actions = np.concatenate([states, actions], axis=1)
        next_states_actions = self.all_states_actions(next_states)
        value_function = Kernel()
        value_function.set(x=states_actions)
        knm = value_function.knm(x=states_actions, y=states_actions)
        projection_op = value_function.knm(
            x=next_states_actions, y=states_actions
        ).reshape([states_actions.shape[0], self.actions_dim, states_actions.shape[0]])
        projection_op[[bool(d) for d in dones]] = 0.0
        sum_policy = np.einsum("...ji,...j", projection_op, policy)
        projection_op = LAlg.lstsq(knm - sum_policy * self.gamma)
        thetas = LAlg.prod(projection_op, rewards)
        value_function.set_theta(thetas)

        next_states_actions_values = value_function(next_states_actions).reshape(
            [states_actions.shape[0], self.actions_dim]
        )
        coeffs = get_tensor_probas(policy)
        second_member = np.einsum("...i,...ij", next_states_actions_values, coeffs)

        derivative_estimator = Kernel()
        derivative_estimator.set_x(states_actions)
        derivative_estimator.set_theta(LAlg.prod(projection_op, second_member))
        if output_value_function:
            return derivative_estimator, value_function
        return derivative_estimator
#########################################################################
# Cartpole
# ------------------------

class PolicyGradientCP(KQLearning.PolicyGradient):

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


    def train(self, game, **kwargs):
        states, actions, next_states, rewards, dones = game
        if len(states) >= kwargs.get("max_game", 1e12):
            print("no training")
            return
        super().train(game,clip=1., **kwargs)

#########################################################################
# LunarLander

class PolicyGradientLN(KQLearning.PolicyGradient):
    def train(self, game, max_training_game_size=None, **kwargs):
        if self.actor.is_valid() and self.actor.get_x().shape[0] > self.replay_buffer.capacity:
            return
        params = kwargs.get("KCritic", {})
        state, action, next_state, reward, return_, done = self.format(
            game, max_training_game_size=max_training_game_size, **kwargs
        )
        if len(self.replay_buffer):
            states, actions, next_states, rewards, returns, dones = (
                self.replay_buffer.memory
            )
        else:
            states, actions, next_states, rewards, returns, dones = state, action, next_state, reward, return_, done
            # dones[0] = True
        games = [states, actions, next_states, rewards, returns, dones]

        if self.actor.is_valid():
            last_policy = self.actor(states)
        else:
            last_policy = np.full(
                [states.shape[0], self.actions_dim], 1.0 / self.actions_dim
            )
        last_policy = np.where(last_policy < 1e-9, 1e-9,last_policy)
        last_policy = np.where(last_policy > 1.-1e-9,1.- 1e-9,last_policy)
        # update probabilities
        if not self.actor.is_valid() or self.actor.get_x().shape[0] < self.replay_buffer.capacity:
            advantages, self.value_function = self.get_advantages(games, policy=last_policy, **kwargs)
            self.actor = self.update_probabilities(
                advantages, games, last_policy=last_policy, clip=.1, **kwargs
            )
        else:
            pass
            # advantages, self.value_function = self.get_advantages(games, policy=last_policy, kernel = self.value_function,**kwargs)
            # kernel = self.update_probabilities(
            #     advantages, games, last_policy=last_policy,kernel = self.actor, clip=.1, **kwargs
            # )
        if not hasattr(self,"scores"):
            self.scores = [rewards.sum()]
        else:
            self.scores.append(rewards.sum())
        # if len(self.replay_buffer)+states.shape[0] < self.replay_buffer.capacity:
        is_pushed = self.replay_buffer.push(
            state, action, next_state, reward, return_, done, worst_game=False,**kwargs
        )

    def format(self, sarsd, max_training_game_size=None, **kwargs):
        states, actions, next_states, rewards, dones = [
            core.get_matrix(e) for e in sarsd
        ]
        actions = KQLearning.rl_hot_encoder(actions, self.actions_dim)
        dones = core.get_matrix(dones, dtype=bool)
        len_game=states.shape[0]
        if max_training_game_size is not None :
            # indices = [int(n*len_game/max_training_game_size) for n in range(0, max_training_game_size)]
            states, actions, next_states, rewards, dones = (
                states[-max_training_game_size:],
                actions[-max_training_game_size:],
                next_states[-max_training_game_size:],
                rewards[-max_training_game_size:],
                dones[-max_training_game_size:],
                # states[:max_training_game_size],
                # actions[:max_training_game_size],
                # next_states[:max_training_game_size],
                # rewards[:max_training_game_size],
                # dones[:max_training_game_size],
            )
        returns = self.compute_returns(
            states, actions, next_states, rewards, dones, **kwargs
        )
        # dones[0]=True
        return states, actions, next_states, rewards, returns, dones