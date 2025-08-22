"""
==========================
8.3 KActorCritic
==========================
Let us introduce the Kernel Actor-Critic algorithm. 
"""
# Importing necessary modules
import os
import sys
import time 

from matplotlib import pyplot as plt
import numpy as np

curr_f = os.path.join(os.getcwd(), "codpy-book", "utils")
sys.path.insert(0, curr_f)

import codpy.core as core
import codpy.KQLearning as KQLearning
import codpy.lalg as LAlg
#########################################################################
# Main Class
#########################################################################

#########################################################################
#The main KActorCritic class can be found at codpy.KQLearning.KActorCritic. 
#The advantage function here is solving for 
#$$A^{\pi^a}(s) = R(s,a) + \gamma V^{\pi}(s') - V^{\pi}(s), \quad s'=S(s,a).$$
# 
#Where $R(s,a)$ are the rewards, $V^{\pi}(s)$ is the value function, and $S(s,a)$ is the next state.
#
#>>> value_function = self.get_state_action_value_function(games, policy, max_y=None,**kwargs)
#
#The :func:`get_state_action_value_function` is described below. It solves the Bellman equation for the value function by fitting a kernel to solve $V^{\pi}(s) = r + \gamma V^{\pi}(s')$. 
#
#We normalize the advantages and set the advantages to 0 when the game is done.
#
#>>> advantages -= core.get_matrix((advantages).mean(1))
#>>> advantages[dones.flatten()] = 0.0

class KActorCritic(KQLearning.KAgent):

    def get_advantages(self, games, policy, **kwargs):
        states, actions, next_states, rewards, returns, dones = games
        value_function = self.get_state_action_value_function(
            games, policy, max_y=None,**kwargs
        )
        advantages = (
           value_function(self.all_states_actions(next_states)).reshape(actions.shape)
            * self.gamma
            + rewards
        )
        advantages -= value_function(np.concatenate([states, actions], axis=1))
        advantages -= core.get_matrix((advantages).mean(1))

        advantages[dones.flatten()] = 0.0
        return advantages,value_function
    
    def get_state_action_value_function(self, games, policy=None, max_y=None,kernel=None,**kwargs):
        """
        Solves the Bellman equation for the value function by fitting a kernel.
        The code is very similar to KQLearning, but we solve for the state value function instead of the state-action value function.
        """

        states, actions, next_states, rewards, returns, dones = games
        if policy is None:
            policy = actions
        if max_y is None: max_y = sys.maxsize
        states_, actions_, next_states_, rewards_, returns_, dones_ = states, actions, next_states, rewards, returns, dones
        policy_ = policy
        states_actions = np.concatenate([states_, actions_], axis=1)
        next_states_actions = self.all_states_actions(next_states_)
        value_function = self.kernel_type(**kwargs.get("KActor", {}))
        if kernel is None:
            value_function.set(x=states_actions, y=states_actions,fx=returns_)
        else:
            value_function.copy(kernel)
        # This is the kernel matrix K(Z,Z) where Z is the state-action pair. 
        knm = value_function.knm(x=states_actions, y=value_function.get_y()) 
        # This is the projection operator K(x',y) where x' is the next state and y is the state-action pair.
        projection_op = value_function.knm(x=next_states_actions, y=value_function.get_y()) 

        def helper(i):
            if dones_[i] == True:
                return [i * self.actions_dim + j for j in range(self.actions_dim)]

        modif = [
            item
            for i in range(dones_.shape[0])
            if dones_[i] == True
            for item in helper(i)
        ]
        projection_op[modif] = 0.0 # We set the projection to 0 when the game is done.

        projection_op = projection_op.reshape(
            [
                states_actions.shape[0],
                projection_op.shape[0] // states_actions.shape[0],
                value_function.get_y().shape[0]
            ]
        )

        sum_policy = np.einsum("...ji,...j", projection_op, policy_)
        mat = knm - sum_policy * self.gamma
        thetas = LAlg.lstsq(mat, rewards)
        value_function.set_theta(thetas)
        value_function.games=(states_, actions_, next_states_, rewards_, returns_, dones_)
        return value_function
    
#########################################################################
# CartPole  

class KActorCriticCP(KQLearning.KActorCritic):

    def format(self, sarsd, max_training_game_size=None, **kwargs):
        """
        Format the game data by keeping only up to max_trainin_game_size timesteps. 

        Parameters:
        - sarsd: tuple collection of game data (states, actions, next_states, rewards, dones).
        - max_training_game_size: maximum number of timesteps to keep for training.

        Returns:
        - states, actions, next_states, rewards, returns, dones: formatted game data.
        """
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
        """
        Skips training if the game was too long. (for cartpole, this means we already solved the environment.)
        """
        states, actions, next_states, rewards, dones = game
        if len(states) >= kwargs.get("max_game", 1e12):
            print("no training")
            return
        super().train(game, clip=1.,**kwargs)

#########################################################################
# LunarLander  

class KActorCriticLN(KQLearning.KActorCritic):
    """
    Defines the main KActorCritic class.

    This inherits from KQLearning.KActorCritic. You can then extend any method from the main class to fit your needs.

    """

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
            )
        returns = self.compute_returns(
            states, actions, next_states, rewards, dones, **kwargs
        )
        # dones[0]=True
        return states, actions, next_states, rewards, returns, dones