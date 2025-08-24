"""
==================================
8.3 Experiments - LunarLander
==================================
We use the OpenAI Gym library to instanciate the gymnasium LunarLander-v3 environment and reproduce the figure from chapter 8_XXX.

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
import codpy.conditioning as conditioning
from ignore_utils import * 

#########################################################################
# KQLearning
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
#########################################################################
# PolicyGradient
# ------------------------
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
#########################################################################
# KActorCritic
# ------------------------
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

#########################################################################
# HJB
# ------------------------
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

#########################################################################
# KController
# ------------------------
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

def main():
    # Define agents here, which will be trained in the benchmark. If game_dictionnary is empty, the benchmark will try to load data from the .pkl file
    game_dictionary = {
        "PPOAgent": PPOAgent,
        "Controller-based": KControllerLN,
        "KACAgent": KActorCriticLN,
        "PolicyGradient": PolicyGradientLN,
        "DQNAgent": DQNAgent,
        # "KQLearningHJBCP": KQLearningHJBLN, #bug to solve get_transition
        "KQLearning": KQLearningLN,
    }

    # Define your agent's parameters here. This dict will be passed in each agent's __init__() method.
    extras = {
        "KActor": {
            # "latent_shape":[100,50],
            "max_size": 1000,
            "n_batch": 1000000,
            "max_nystrom": 1000,
            "reg": 1e-6,
            "order": None,
        },
        "KCritic": {
            "max_size": 1000000,
            "n_batch": 1000000,
            "max_nystrom": 1000,
            "reg": 1e-9,
            "order": None,
        },
        "HJBModel": {
            # "latent_shape":[100,50],
            "max_size": 100000,
            "n_batch": 1000000,
            "max_nystrom": 1000,
            "reg": 1e-9,
            "order": None,
            "state_dim": 8,
        },
        "Rewards": {
            "max_size": 1000000,
            "n_batch": 100000,
            "max_nystrom": 1000,
            "reg": 1e-9,
            "order": None,
        },
        "NextStates": {
            "max_size": 1000000,
            "n_batch": 100000,
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
        "ACAgent": {"reward_function": None},
        "QAgent": {
            0
            # 'reward_function': mc_reward_function
        },
        "KController": {
            "reg": 1e-9,
            "order": 2,
        },
        "Conditionner": {
            "reg": 1e-4,
            "order": 3,
        },
        "max_game": 2000,
        "gamma": 0.99,
        "capacity": 10000,
        "max_training_game_size": 1000,
        # "max_kernel": 40
        # "seed": 42,
    }
    seed = extras.get("seed", None)
    np.random.seed(seed)
    softmax = lambda x: np.exp(x) / np.sum(np.exp(x), axis=0)
    test = softmax([1,0])
    Benchmark()(
        game_dictionary,
        "LunarLander-v3",
        num_games=100,
        eps_threshold=0.1,
        num_repeats=3,
        max_time=50,
        axis="episodes",
        # file_name="results_LN_final.pkl",
        **extras,
    )
    plt.show()
    pass

main()