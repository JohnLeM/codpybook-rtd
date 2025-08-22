""".. _kqlearning:

==========================
8.1 KQLearning
==========================
Let us introduce the KQLearning Algorithm. 
"""
# Importing necessary modules
import os
import sys
import time 

from matplotlib import pyplot as plt
import numpy as np
from scipy import optimize

curr_f = os.path.join(os.getcwd(), "codpy-book", "utils")
sys.path.insert(0, curr_f)

import codpy.core as core
import codpy.KQLearning as KQLearning
import codpy.lalg as LAlg
from codpy.utils import gather
#########################################################################
# Main Class
#########################################################################

#########################################################################
#The main KQLearning class can be found at :class:`codpy.KQLearning.KQLearning`. 
#This will run out of the box, but can be tweaked and tuned to fit your needs and environment constraints. 
#Let us define the main class important methods and details here, altough any custom class should inherit from the main one and modify the methods in their own definition. 
#
#You can see below how this works, when we define two KQLearning subclasses, one for the Cartpole and one for the LunarLander environment. 
#
#This class will be solving the optimal Bellman equation: 
# $$Q(s,a) = r + \gamma \max_{a'} Q(s',a')$$
#It does so calling the :func:`optimal_states_values_function` in an iterative scheme, where at each loop it does two things: 
# 
# 1- Solve: 
# $$\theta^{\pi}_{n+1/2} = \Big( K(Z, Z) - \gamma \sum_a \pi_{n+1/2}^a(S) K(W^a,Z)\Big)^{-1} R$$
# 2- It then refines the parameters through an interpolation coefficient 
# $$\theta_{n+1}^{\pi} = \lambda \theta^{\pi}_{n+1/2} + (1 - \lambda) \theta_{n}^{\pi}.$$
#
#Z is the concatenation of the states and actions, which is defined: 
#
#>>> Z = np.concatenate([states, actions], axis=1)
#
#$K(Z,Z)$ is the gram matrix of current state actions pairs, defined with
#
#>>> _knm_ = kernel.knm(x=Z, y=kernel.get_x())
# 
#$K(W^a,Z)$ is the gram matrix of the next states and actions, defined with	
#
#>>> _projection_ = kernel.knm(x=next_states_actions, y=kernel.get_x())
#
#The function then assures a limit condition on the Q-values by setting the last Q-values equal to the rewards. 
#
#In KQLearning, $\pi_{n+1/2}^a(S)$ is the maxt of the next Q-values. We get all next states and actions with: 
# 
#>>> next_states_actions = self.all_states_actions(next_states)
#
#After that, we pass those to :func:`optimal_bellman_solver`. The first $\theta$ is computed with
#
#>>> next_theta = LAlg.lstsq(knm - max_projection * self.gamma, rewards, reg)
#
#Where reg is the regularization parameter. From there we interpolate the $\theta$, and we optimize for the $\lambda$ parameter using the Brent method.
#
#>>> def f(x):
#>>>    interpolated_thetas = theta * x + next_theta * (1.0 - x)
#>>>    out = bellman_error(interpolated_thetas)
#>>> return out
#>>> xmin, fval, iter, funcalls = optimize.brent(f, brack=(0.0, 1.0), maxiter=maxiter, full_output=True)
#
#Here you can see that we run this in a loop and compute the Bellman error each time. Practically, we iterate a certain number of times, and stop either when the error is below a certain threshold, or when we reach the maximum number of iterations.
class KQLearning_main(KQLearning.KActorCritic):

    def __init__(
        self, actions_dim, state_dim, gamma=0.99, kernel_type=KQLearning.GamesKernel, **kwargs
    ):
        """
        This class has at its core a kernel, self.critic, which will be used to output the Q-values.

        Parameters:
        - actions_dim: int, the dimension of the action_space in the environment
        - state_dim: int, the dimension of the state_space in the environment
        - gamma: float, the discount factor 
        - kernel_type: class, the kernel class to be used for the critic.
        - kwargs: dict, additional parameters for the kernel class
        """
        self.kernel_type = kernel_type 
        self.actions_dim = actions_dim
        self.state_dim = state_dim
        self.gamma = gamma
        params = kwargs.get("KCritic", {})
        self.critic = self.kernel_type(gamma=gamma, **params)

        super().__init__(actions_dim, state_dim, gamma, kernel_type, **kwargs)

    def __call__(self, state, **kwargs):
        """
        This is what will be used at inference. 

        Parameters:
        - state: np.ndarray, the state of the environment of shape (d, state_dim) with d the batch dimension

        Returns: 
        - int, the action to be taken in the environment
        """
        # Epsilon greedy decay mechanism
        self.eps_threshold *= 0.999
        if np.random.random() > self.eps_threshold and self.critic.is_valid() == True:
            z = self.all_states_actions(core.get_matrix(state).T)
            q_values = self.critic(z)
            q_values += np.random.random(q_values.shape) * 1e-9
            return np.argmax(q_values)
        return np.random.randint(0, self.actions_dim)
    
    def optimal_states_values_function(
            self, games, kernel=None, full_output=False, **kwargs
        ):
            """
            Solves the Bellman equation on the games
            """

            states, actions, next_states, rewards, returns, dones = games

            states_actions = np.concatenate([states, actions], axis=1)
            if kernel is None or not kernel.is_valid():
                kernel = self.kernel_type(x=states_actions, fx=returns, **kwargs)
            else:
                states_actions = np.concatenate([kernel.get_x(), states_actions], axis=0)
                rewards = np.concatenate([kernel.games[3], rewards], axis=0)
                next_states = np.concatenate([kernel.games[2], next_states], axis=0)

            next_states_actions = self.all_states_actions(next_states)
            _projection_ = kernel.knm(x=next_states_actions, y=kernel.get_x())

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

            _knm_ = kernel.knm(x=states_actions, y=kernel.get_x())
            thetas, bellman_error, indices = self.optimal_bellman_solver(
                thetas=kernel.get_theta(),
                next_states_projection=_projection_,
                knm=_knm_,
                rewards=rewards,
                games=games,
                **kwargs,
            )
            kernel.set_theta(thetas)
            kernel.bellman_error = bellman_error
            if full_output:
                return kernel, bellman_error, indices
            return kernel
    
    def optimal_bellman_solver(
            self,
            thetas,
            next_states_projection,
            knm,
            rewards,
            maxiter=5,
            reg=1e-9,
            tol=1e-6,
            verbose=False,
            **kwargs,
        ):
            """
            Called by the optimal_states_values_function, this solves the Bellman equation on the games and minimize the error. 
            """
            theta = thetas.copy()
            shape = [next_states_projection.shape[0] // self.actions_dim, self.actions_dim]

            def bellman_error(theta, full_output=False):
                error = (
                    LAlg.prod(next_states_projection, theta).reshape(shape) * self.gamma
                    + rewards
                )
                max_indices = error.argmax(1)
                error = gather(error, max_indices)
                error -= LAlg.prod(knm, theta)
                if full_output == True:
                    return np.fabs(error).mean(), max_indices
                return np.fabs(error).mean()

            error, max_indices = bellman_error(theta, full_output=True)
            count = 0
            if error < tol:
                return thetas, error, max_indices
            while count < maxiter and error > tol:
                indices = [
                    self.actions_dim * i + max_indices[i] for i in range(len(max_indices))
                ]
                max_projection = next_states_projection[indices]
                next_theta = LAlg.lstsq(knm - max_projection * self.gamma, rewards, reg)

                def f(x):
                    interpolated_thetas = theta * x + next_theta * (1.0 - x)
                    out = bellman_error(interpolated_thetas)
                    return out

                xmin, fval, iter, funcalls = optimize.brent(
                    f, brack=(0.0, 1.0), maxiter=maxiter, full_output=True
                )
                if fval >= error:
                    break
                theta = theta * xmin + next_theta * (1.0 - xmin)
                error, max_indices = bellman_error(theta, full_output=True)
                count = count + 1

            max_indices = LAlg.prod(next_states_projection, theta).reshape(shape).argmax(1)
            indices = [
                self.actions_dim * i + max_indices[i] for i in range(len(max_indices))
            ]
            if verbose:
                print("Computed global error Bellman mean: ", fval, " iter: ", count)
            return theta, fval, indices

    def train(
        self,
        game,
        max_training_game_size=None,
        format=True,
        tol=1e-2,
        **kwargs,
    ):
        """
        Training algorithm for the KQLearning. 

        Parameters: 
        - game: tuple, previous game data (states, actions, next_states, rewards, returns, dones)
        - max_training_game_size: int, the maximum size of the training game to be used
        - format: bool
        - tol: float, Bellman error tolerance threshold for updating the kernel
        """

        # First we format the game data, and push it to the replay buffer.
        if format:
            states, actions, next_states, rewards, returns, dones = self.format(
                game, max_training_game_size=max_training_game_size, **kwargs
            )
        else:
            states, actions, next_states, rewards, returns, dones = game

        if self.critic.is_valid():
            returns = self.critic(np.concatenate([states, actions], axis=1))
            games = states, actions, next_states, rewards, returns, dones

        self.replay_buffer.push(
            states, actions, next_states, rewards, returns, dones, capacity=sys.maxsize
        )

        if (
            len(self.replay_buffer) <= self.replay_buffer.capacity
        ):
            # As long as the replay buffer is not full, we can train on the entire replay buffer.
            # self.optimal_states_values_function solves for Bellman equation on the entire buffer and return a kernel of type self.kernel_type fit on the data.

            games = self.replay_buffer.memory
            kernel = self.optimal_states_values_function(games, verbose=True, **kwargs)
            kernel.games = games
            if len(self.critic.kernels) == 0:
                # This allow us to perform clustering later on
                self.critic.add_kernel(kernel)
            else:
                # As long as the replay buffer isn't full, we only keep 1 kernel.
                self.critic.kernels[len(self.critic.kernels) - 1] = kernel
            return
        else:
            # Once full, we fit a new kernel on half of the latest data. 
            # This is still done through optimal_state_values_function
            # And we then add the kernel, which will be used for clustering. 
            kernel = self.critic.kernels[len(self.critic.kernels) - 1]
            kernel.set(
                x=kernel.get_x()[: self.replay_buffer.capacity // 2],
                fx=kernel.get_fx()[: self.replay_buffer.capacity // 2],
            )
            kernel.games = [
                elt[: self.replay_buffer.capacity // 2]
                for elt in self.replay_buffer.memory
            ]
            games = self.replay_buffer.memory
            games = [
                elt[self.replay_buffer.capacity // 2 :]
                for elt in self.replay_buffer.memory
            ]
            kernel = self.optimal_states_values_function(games, verbose=True, **kwargs)
            kernel.games = games
            self.critic.add_kernel(kernel)
            self.replay_buffer.memory = games

        # We then check if the kernels are still valid, and if not, we delete them.
        # We see if we can delete the kernels, based on a Bellman error tolerance threshold. 
        # This help cope with growing number of kernels and ones which don't give good results.
        delete_kernels = []
        for i, k in self.critic.kernels.items():
            error = self.critic.kernels[i].bellman_error
            if error > tol and not hasattr(self.critic.kernels[i], "flag_kill_me"):
                kernel = self.optimal_states_values_function(
                    k.games, kernel=k, verbose=True, **kwargs
                )
                kernel.games = self.critic.kernels[i].games
                if error <= self.critic.kernels[i].bellman_error:
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
            
#########################################################################
# CartPole
# ------------------------
# We know look at how you should subclass KQLearning and override methods to fit your needs and environment constraints.
class KQLearningCP(KQLearning.KQLearning):

    def format(self, sarsd, max_training_game_size=None, **kwargs):
        """
        In Cartpole, we only want to keep a certain amount of timesteps for each episode. The original format approach keep all the data.
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

    def train(self, game, max_training_game_size =sys.maxsize,tol=1e-4,**kwargs):
        """
        In cartpole we don't want clustering so we override the train method. 
        """
        states, actions, next_states, rewards, dones = game

        # In cartpole we skip training if we already solved the environment.
        if len(states) >= kwargs.get("max_game", 1e12):
            print("no training")
            return
        states, actions, next_states, rewards, returns, dones = self.format(game, max_training_game_size=max_training_game_size,**kwargs)
        if self.critic.is_valid():
            returns = self.critic(np.concatenate([states,actions],axis=1))

        self.replay_buffer.push(states, actions, next_states, rewards, returns, dones)
        games = self.replay_buffer.memory

        # self.critic here is a kernel, and it fit on the entire replay buffer to solve for Bellman equations.
        self.critic = self.optimal_states_values_function(games,verbose=True,**kwargs)        
        return 
    
#########################################################################
# LunarLander
# ------------------------

class KQLearningLN(KQLearning.KQLearning):

    def train(
        self,
        game,
        max_training_game_size=None,
        format=True,
        replay_buffer=True,
        tol=1e-2,
        **kwargs
    ):
        """
        For LunarLander, we want to fit one kernel per game. So again, we override the train method.
        """
        game = self.format(
            game, max_training_game_size=max_training_game_size, **kwargs
        )
        # Here the kernel is fit on the latest game only. 
        kernel = self.optimal_states_values_function(game, verbose=True, **kwargs)
        kernel.games = game
        self.critic.add_kernel(kernel, **kwargs)
        delete_kernels = []
        for i, k in self.critic.kernels.items():
            error = self.critic.kernels[i].bellman_error
            if error > tol and not hasattr(self.critic.kernels[i], "flag_kill_me"):
                kernel = self.optimal_states_values_function(
                    self.critic.kernels[i].games,
                    kernel=self.critic.kernels[i],
                    verbose=True,
                    **kwargs,
                )
                kernel.games = self.critic.kernels[i].games
                if kernel.bellman_error >= error - tol:
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

# Instanciate your gymnasium environment
import gymnasium as gym

env = gym.make("CartPole-v1")
agent = KQLearningCP(env.action_space.n, env.observation_space.shape[0], gamma=0.99)
obs, _ = env.reset()
steps = 0
while steps < 1000:
    action = agent(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    steps += 1
    if terminated or truncated:
        obs, _ = env.reset()
env.close()
