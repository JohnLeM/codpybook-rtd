"""
==========================
8.2 Experiments - Cartpole
==========================
We use the OpenAI Gym library to instanciate the gymnasium CartPole-v1 environment and reproduce the figure from chapter 8_XXX. 

We train the following agents: 

- PPO 
- DQN 
- Controller-based
- Kernel Actor-Critic
- Kernel Q-Learning
- Kernel Q-Learning HJB
- Kernel Policy-Gradient

We show how you can tweak some methods in each algorithm to tune them to the environment. For a detailed documentation on KAgents, see **codpy documentation**.
"""

# Importing necessary modules
import sys

from matplotlib import pyplot as plt
import numpy as np

import codpy.core as core
import codpy.KQLearning as KQLearning

from ignore_utils import * 

#########################################################################
# KQLearning
# ------------------------
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
# PolicyGradient
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
# KActorCritic
# ------------------------
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
# HJB
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
# KController
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
        
if __name__ == "__main__":
    # Define agents here, which will be trained in the benchmark. If game_dictionnary is empty, the benchmark will try to load data from the .pkl file
    game_dictionary = {
        "PPOAgent": PPOAgent,
        "PolicyGradient": PolicyGradientCP,
        "Controller-based": KControllerCP,
        "KACAgent": KActorCriticCP,
        "DQNAgent": DQNAgent,
        "KQLearningHJBCP": KQLearningHJBCP,
        "KQLearning": KQLearningCP,
    }

    # Define your agent's parameters here. This dict will be passed in each agent's __init__() method.
    extras = {
        # "D":4,
        "KActor": {"n_batch": 1000000, "max_nystrom": 1000, "reg": 1e-9, "order": None},
        "KCritic": {
            "n_batch": 1000000,
            "max_nystrom": 1000,
            "reg": 1e-9,
            "order": None,
        },
        "Rewards": {
            "n_batch": 1000000,
            "max_nystrom": 1000,
            "reg": 1e-9,
            "order": None,
        },
        "DQNAgent": {
            # 'reward_function': mc_reward_function,
            "episodes": 500,
            "policy_param": 64,
            "target_param": 64,
        },
        "KController": {
            "reg": 1e-3,
            "order": None,
        },
        "HJBModel": {
            # "latent_shape":[100,50],
            "max_size": 100000,
            "n_batch": 1000000,
            "max_nystrom": 1000,
            "reg": 1e-9,
            "order": None,
            "state_dim": 4,
        },
        "max_game": 1000,
        "max_training_game_size": 1000,
        "gamma": 0.99,
        "capacity": 200000000,
        # "seed": 42,
    }
    seed = extras.get("seed", None)
    np.random.seed(seed)

    Benchmark()(
        game_dictionary,
        "CartPole-v1",
        num_games=100,
        num_repeats=3,
        max_time=3,
        axis="episode",
        # file_name="results_CP_final.pkl",
        **extras,
    )
    plt.show()
    pass