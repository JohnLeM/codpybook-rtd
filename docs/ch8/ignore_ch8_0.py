"""
==========================
8.0 Required Classes
==========================
Let us introduce the required classes & algorithms for this chapter: 

- KAgent 
- PPO
- DQN
- Benchmark

The first one is the main kernel class which will be used later in our examples. 
All other Kernel algorithms depend on the base one, with some modifications.

PPO and DQN are well known Reinforcement Learning algorithms. They both use Neural Networks as function approximator, and will be used as baselines for our benchmarks. 

The last one is an utility class, which will be used to run the environment and collect metrics. 

We use gymnasium to run the environments. We will be benchmarking on the Cartpole-v1 and LunlarLander-v3. 
"""
# Importing necessary modules
import os
import sys
import time 
import math 
import random 
import time 
from collections import namedtuple, deque 

import gymnasium as gym
from matplotlib import pyplot as plt
import numpy as np
import pickle
from scipy import optimize

import torch
import torch.nn as nn
from torch.distributions import Categorical

import codpy.KQLearning as KQLearning
import codpy.lalg as LAlg
from codpy.utils import gather

######################################################
# **Main Kernel class: KAgent**
# -------------------------------
# This class is the main kernel class for all Kernel Reinforcement Learning Algorithms. It has a couple of kernel defined, some of them which will be used depending on the specific algorithm implemented.
# :param x: sdlfkj. 

"""
==========================
8.014 Required Classes
==========================
Let us introduce the required classes & algorithms for this chapter: 

- KAgent 
- PPO
- DQN
- Benchmark

:param x: amlskdjfam
:param y: malskdjf
"""
class KAgent:
    def __init__(
        self, actions_dim, state_dim, gamma=0.99, **kwargs
    ):
        """
        Test phrase exemple description 

        Args: 
            x (float): test x value
            y (int): test y value

        """
        #: this is  diff comm
        # than this


    
#########################################################################
# Benchmarking Class
# ------------------------
# We define the class which will benchmark the agents. 

# This function is used to get a reliable marker style on plots, so each algorithm is tied to a specific marker.
def get_style_cycler():
    markers = ["o", "v", "s", "D", "^", "P", "*", "X", "H"]
    linestyles = ["-", "--", "-.", ":"]
    colors = plt.cm.tab10.colors 
    styles = [(m, "-", c) for m, c in zip(markers, colors)]
    return styles


class Benchmark:
    """
    You can define this class as you see fit. However, Codpy main Reinforcement Learning algorithms expect the be called following a specific schema, which you will need to use here. 
    Specifically, the __call__() and train() methods are expected to be implemented. 
    The __call__() is used for inference and will be called to get the action used to step the environment. 
    The train() will be called with the episode data, and this is where the training of the agent will be performed. 

    By default, the agent trains at the end of each episode. If your agent trains after each timestep, you need to set the attribute 'online_train' to your agent's class.
    """
    def __init__(self, verbose=1):
        self.verbose = verbose
        self.training_times = {}
        self.prediction_times = {}

    def update_rewards(self, cumulative_rewards, rewards):
        for label, reward in rewards.items():
            if label in cumulative_rewards and cumulative_rewards[label]:
                cumulative_reward = cumulative_rewards[label][-1] + reward
            else:
                cumulative_reward = reward
                cumulative_rewards[label] = []
            cumulative_rewards[label].append(cumulative_reward)

    def reset_rewards(self, cumulative_rewards):
        for label in cumulative_rewards.keys():
            cumulative_rewards[label].clear()

    def benchmark_agents(
        self, agent_classes, game_name, num_games=100, max_game=1000, max_time=sys.float_info.max,**kwargs
    ):
        """
        This is how we pass multiple agents to the class. 

        Parameters: 
        - agent_classes: dict of agent classes to be benchmarked.
        - game_name: str name of the gymnasium environment. 
        - num_games: int number of episodes to be run. 
        - max_game: int maximum number of timesteps per episode. 
        - max_time: float maximum time allowed per train() call.
        """
        env = gym.make(game_name)
        state_dim = env.observation_space.shape[0]
        if type(env.action_space) is gym.spaces.Box:
            actions_dim = env.action_space.shape[0]

        else:
            actions_dim = env.action_space.n
        cumulative_rewards = {label: [] for label in agent_classes.keys()}
        success_rates = {label: [] for label in agent_classes.keys()}
        training_times = {label: [] for label in agent_classes.keys()}
        prediction_times = {label: [] for label in agent_classes.keys()}
        steps_number = {label: [] for label in agent_classes.keys()}

        for label, AgentClass in agent_classes.items():
            steps_number[label] = []
            training_times[label] = []
            prediction_times[label] = []
            success_rates[label] = []
            kwargs["label"] = label
            # This is the init signature that all agents must follow for consistency. Every Kernel RL algorithms expect this.s
            agent = AgentClass(state_dim=state_dim, actions_dim=actions_dim, **kwargs)
            training_time,prediction_time, success = 0.0,0.0, 0

            for i in range(num_games):
                env_copy = env.env
                start_time = time.time()
                game = play(env_copy, agent, max_game=max_game)
                prediction_time += time.time()-start_time
                rewards = np.sum(game[3])
                self.update_rewards(cumulative_rewards, {label: rewards})

                start_time = time.time()
                if hasattr(agent, "train") and training_time < max_time:
                    agent.train(game, max_game=max_game, **kwargs)
                if hasattr(agent, "online_train") and training_time < max_time:
                    training_time = prediction_time
                else:
                    training_time += time.time() - start_time

                prediction_times[label].append(prediction_time)
                training_times[label].append(training_time)

                # Calculate success rate
                success += 1 if rewards >= env.spec.reward_threshold else 0
                success_rates[label].append(success)

                steps = len(game[0])
                steps_number[label].append(steps)

                if self.verbose == 1:
                    print(
                        f"label {label}, Reward {i}: {np.mean(rewards):.3f}, Len(game): {len(game[3])}, Training Time: {training_time:.3f}s, Prediction Time: {prediction_time:.3f}s"
                    )

        return (
            cumulative_rewards,
            training_times,
            prediction_times,
            success_rates,
            steps_number,
        )

    def plot_rewards(
        self, cumulative_rewards, steps_number, num_games, axis="steps", **kwargs
    ):
        plt.figure(figsize=(12, 5))

        styles = get_style_cycler()
        agent_labels = sorted(cumulative_rewards.keys())

        for idx, label in enumerate(agent_labels):
            marker, linestyle, color = styles[idx % len(styles)]

            rewards = cumulative_rewards[
                label
            ]
            mean_rewards = np.mean(rewards, axis=0)
            std_rewards = np.std(rewards, axis=0)

            if axis == "steps":
                episodes = np.cumsum(steps_number[label][0])
            else:
                episodes = np.arange(len(mean_rewards))

            plt.plot(
                episodes,
                mean_rewards,
                label=label,
                linestyle=linestyle,
                marker=marker,
                markevery=max(1, len(episodes) // 15),
                markersize=7,
                markerfacecolor=color,
                markeredgewidth=1.2,
                linewidth=2,
                color=color,
            )
            plt.fill_between(
                episodes,
                mean_rewards - std_rewards,
                mean_rewards + std_rewards,
                alpha=0.2,
                color=color,
            )

        plt.xlabel("Steps" if axis == "steps" else "Episodes")
        plt.ylabel("Cumulative Reward")
        plt.title(f"Cumulative Reward over {num_games} Games")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

    def plot_times(
        self, training_times, steps_number, num_games, axis="steps", **kwargs
    ):
        plt.figure(figsize=(12, 5))
        styles = get_style_cycler()
        agent_labels = sorted(training_times.keys())

        for idx, label in enumerate(agent_labels):

            marker, linestyle, color = styles[idx % len(styles)]
            times = training_times[label]
            mean_times = np.mean(times, axis=0)

            if axis == "steps":
                cum_steps = np.cumsum(steps_number[label])
            else:
                cum_steps = list(range(len(mean_times)))

            plt.plot(
                cum_steps,
                mean_times,
                label=label,
                linestyle=linestyle,
                marker=marker,
                markevery=max(1, len(cum_steps) // 20),
                markersize=5,
                linewidth=1.5,
                color=color,
            )

        plt.xlabel("Steps" if axis == "steps" else "Episodes")
        plt.ylabel("Training Time (seconds)")
        plt.title(f"Training Time per Game over {num_games} Games")
        plt.legend()
        plt.tight_layout()

    def __call__(
        self,
        agent_classes,
        game_id,
        num_games=10000,
        num_repeats=10,
        file_name=None,
        **extras,
    ):
        """
        This is what is called by the external scripts. If a pkl file is provided and agent_classes is empty, it will load the file data instead of re-training the agents.

        Parameters: 
        - agent_classes: dict of agent classes to be benchmarked.
        - game_id: str name of the gymnasium environment.
        - num_games: int number of episodes to be run.
        - num_repeats: int number of repeats for the benchmark. This runs each agent n times.
        - file_name: str name of the file to save or load the results from. If None, it will train agents and not save the results. If a file name is provided and agent_classes is empty, this will try to load previous data. Otherwise, it trains the agent and save the results to the file.
        - extras: dict of extra parameters to be passed for each agent. 
        """

        file_path = None
        if file_name is not None:
            file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),file_name)

        if file_path and os.path.exists(file_path):
            file = open(file_path, "rb")
            debug = pickle.load(file)
            (
                all_cumulative_rewards,
                all_training_times,
                all_prediction_times,
                all_success_rates,
                all_steps_number,
            ) = debug
        else:
            (
                all_cumulative_rewards,
                all_training_times,
                all_prediction_times,
                all_success_rates,
                all_steps_number,
            ) = (
                {},
                {},
                {},
                {},
                {},
            )

        all_cumulative_rewards = {
            **all_cumulative_rewards,
            **{label: [] for label in agent_classes.keys()},
        }
        all_training_times = {
            **all_training_times,
            **{label: [] for label in agent_classes.keys()},
        }
        all_prediction_times = {
            **all_prediction_times,
            **{label: [] for label in agent_classes.keys()},
        }
        all_success_rates = {
            **all_success_rates,
            **{label: [] for label in agent_classes.keys()},
        }
        all_steps_number = {
            **all_steps_number,
            **{label: [] for label in agent_classes.keys()},
        }

        for j in range(num_repeats):
            (
                cumulative_rewards,
                training_times,
                prediction_times,
                success_rates,
                steps_number,
            ) = self.benchmark_agents(agent_classes, game_id, num_games, **extras)
            for label in agent_classes.keys():
                all_cumulative_rewards[label].append(cumulative_rewards[label])
                all_training_times[label].append(training_times[label])
                all_prediction_times[label].append(prediction_times[label])
                all_success_rates[label].append(success_rates[label])
                all_steps_number[label].append(steps_number[label])
            print(j)

        if file_name is not None:
            pickle.dump(
                [
                    all_cumulative_rewards,
                    all_training_times,
                    all_prediction_times,
                    all_success_rates,
                    all_steps_number,
                ],
                open(file_path, "wb"),
            )
        self.plot_rewards(all_cumulative_rewards, steps_number, num_games, **extras)
        self.plot_times(all_training_times, steps_number, num_games, **extras)
        return all_cumulative_rewards

#########################################################################
# Play
# ------------------------
# This plays an episode for max_game and break if flagged done. 

def play(env, agent, seed=None, max_game=1000, **kwargs):
    state, _ = env.reset(seed=seed)

    states, actions, next_states, rewards, dones = [], [], [], [], []
    for t in range(max_game):
        action = agent(state, **kwargs)
        next_state, reward, done, _, _ = env.step(action)
        (
            states.insert(0, state),
            actions.insert(0, action),
            next_states.insert(0, next_state),
            rewards.insert(0, reward),
            dones.insert(0, done),
        )
        if hasattr(agent, "online_train"):
            agent.online_train(state, action, reward, next_state, done, **kwargs)
        state = next_state

        if done:
            break
    return states, actions, next_states, rewards, dones


#########################################################################
# PPO
# ------------------------
# This is the memory used by PPO to accumulate experience before training. 

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = [] 
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        self.actions.clear()
        self.states.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.is_terminals.clear()

#########################################################################
# The Actor / Critic Networks using Pytorch and Fully Connected layers. 

class ActorCritic(nn.Module):
    """
    Actor-critic network for discrete action space
    """

    def __init__(self, state_dim, action_dim, hidden_size=128):
        super(ActorCritic, self).__init__()

        # Actor network
        self.actor_network = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1),
        )

        # Critic network
        self.critic_network = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def select_action(self, state, memory):
        state = torch.from_numpy(state).float()
        action_probabilities = self.actor_network(state)
        action_distribution = Categorical(action_probabilities)
        action = action_distribution.sample()

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_distribution.log_prob(action))

        return action.item()

    def compute_logprobs_values_entropy(self, state, action):
        action_probabilities = self.actor_network(state)
        action_distribution = Categorical(action_probabilities)

        action_logprobs = action_distribution.log_prob(action)
        dist_entropy = action_distribution.entropy()

        state_value_estimate = self.critic_network(state)

        return action_logprobs, torch.squeeze(state_value_estimate), dist_entropy

#########################################################################
# Main PPO class.
# ------------------------
# This implements the PPO algorithm. 
class PPOAgent:
    def __init__(
        self,
        state_dim,
        actions_dim,
        hidden_size=128,
        lr=0.002,
        betas=(0.9, 0.999),
        gamma=0.99,
        num_epochs=4,
        eps_clip=0.2,
        update_interval=1200,
        **kwargs,
    ):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.num_epochs = num_epochs
        self.current_timestep = 0
        self.update_interval = update_interval
        self.memory = Memory()

        # Init policy and optimizer
        self.policy = ActorCritic(state_dim, actions_dim, hidden_size)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.policy_old = ActorCritic(state_dim, actions_dim, hidden_size)
        self.policy_old.load_state_dict(self.policy.state_dict())

        # loss function for value estimation
        self.mse_loss = nn.MSELoss()

    def update_policy(self):
        # MC estimate of state rewards:
        discounted_rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(
            reversed(self.memory.rewards), reversed(self.memory.is_terminals)
        ):
            if reward is None:
                reward = 0.0

            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            discounted_rewards.insert(0, discounted_reward)

        # rewards normalization:
        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32)
        if len(discounted_rewards) > 1:
            discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (
                discounted_rewards.std() + 1e-5
            )

        old_states = torch.stack(self.memory.states).detach()
        old_actions = torch.stack(self.memory.actions).detach()
        old_logprobs = torch.stack(self.memory.logprobs).detach()

        # policy optimization for num_epochs epochs:
        for _ in range(self.num_epochs):
            # Evaluate old actions and values:
            logprobs, state_values, dist_entropy = (
                self.policy.compute_logprobs_values_entropy(old_states, old_actions)
            )

            # (pi_theta / pi_theta_old):
            importance_ratio = torch.exp(logprobs - old_logprobs.detach())

            # advantages and surrogate loss:
            advantages = discounted_rewards - state_values.detach()
            unclipped_surrogate_loss = importance_ratio * advantages
            clipped_surrogate_loss = (
                torch.clamp(importance_ratio, 1 - self.eps_clip, 1 + self.eps_clip)
                * advantages
            )
            loss = (
                -torch.min(unclipped_surrogate_loss, clipped_surrogate_loss)
                + 0.5 * self.mse_loss(state_values, discounted_rewards)
                - 0.01 * dist_entropy
            )

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # new weights loaded into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

    def online_train(self, state, action, reward, next_state, done, **kwargs):
        self.current_timestep += 1
        # save reward and if is_terminal:
        self.memory.rewards.append(reward)
        self.memory.is_terminals.append(done)

        # update policy:
        if self.current_timestep % self.update_interval == 0:
            self.update_policy()
            self.memory.clear()
            self.current_timestep = 0

    def __call__(self, state, **kwargs):
        return self.policy_old.select_action(state, self.memory)
    
#########################################################################
# DQN
# ------------------------
# This is the DQN agent. It uses a feedforward neural network to approximate the Q function, and a target Network to stabilize training. 
# It uses a replay buffer to store experience and sample batches for training.
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
class ReplayBuffer(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
class DQN_FFN(nn.Module):
    """
    This is the basic Network structure.
    """
    def __init__(self, n_observations, n_actions, **kwargs):
        super().__init__()
        size = kwargs.get("size", 128)
        self.layer1 = nn.Linear(n_observations, size)
        self.layer2 = nn.Linear(size, size)
        self.layer3 = nn.Linear(size, n_actions)

    def forward(self, x):
        x = nn.functional.relu(self.layer1(x))
        x = nn.functional.relu(self.layer2(x))
        return self.layer3(x)


class DQNAgent:
    def __init__(
        self,
        state_dim,
        actions_dim,
        epsilon=0.1,
        alpha=0.001,
        gamma=0.99,
        buffer_size=100000,
        batch_size=64,
        tau=0.005,
        **kwargs
    ):
        self.actions_dim = actions_dim
        self.state_dim = state_dim

        self.epsilon = epsilon 
        self.alpha = alpha 
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        DQN_param = kwargs.get("QNAgent", {})
        policy_param = DQN_param.get("policy_param", 128)
        target_param = DQN_param.get("target_param", 128)

        self.policy_net = DQN_FFN(self.state_dim, self.actions_dim, size=policy_param)
        self.target_net = DQN_FFN(self.state_dim, self.actions_dim, size=target_param)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=self.alpha)
        self.memory = ReplayBuffer(buffer_size)
        self.steps_done = 0
        self.i = 0

        self.path = r"{}".format(os.path.expanduser("~/Desktop/Q"))
        self.global_max_score = -1e10
        self.global_max_height = -1e10
        self.save_freq = 100
        self.checkpoint_dir = r"{}".format(os.path.expanduser("~/Desktop/Q"))

    def saveQ(self, path):
        torch.save(self.policy_net.state_dict(), path + ".pth")

    def loadQ(self, path):
        self.policy_net.load_state_dict(torch.load(path + ".pth"))

    def __call__(self, state, **kwargs):
        kwargs = kwargs.get("DQNAgent", {})
        nature = kwargs.get("nature_a", "d")
        EPS_START = kwargs.get("EPS_START", 0.9)
        EPS_END = kwargs.get("EPS_END", 0.05)
        EPS_DECAY = kwargs.get("EPS_DECAY", 1000)
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(
            -1.0 * self.steps_done / EPS_DECAY
        )
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                state = torch.tensor([state], dtype=torch.float32)
                return self.policy_net(state).max(1)[1].view(1, 1).item()
        else:
            return self.action_sample()

    def action_sample(self):
        return random.randint(0, self.actions_dim - 1)

    def compute_loss(self, batch):
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool
        )
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None]
        )
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(
                non_final_next_states
            ).max(1)[0]

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        return loss

    def optimize(self, state, action, reward, next_state, done, **kwargs):
        reward_function = kwargs.get("reward_function", None)
        if reward_function:
            reward = reward_function(state, next_state, reward)

        if done:
            next_state = None
        else:
            next_state = torch.tensor([next_state], dtype=torch.float32)

        state = torch.tensor([state], dtype=torch.float32)
        action = torch.tensor([[action]], dtype=torch.int64)
        reward = torch.tensor([reward], dtype=torch.float32)

        self.memory.push(state, action, next_state, reward)

        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        loss = self.compute_loss(batch)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 1000)
        self.optimizer.step()

        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[
                key
            ] * self.tau + target_net_state_dict[key] * (1 - self.tau)
        self.target_net.load_state_dict(target_net_state_dict)

    def train(self, episode, **kwargs):
        kwargs = kwargs.get("DQNAgent")
        verbose = kwargs.get("verbose", 0)
        episodes_num = kwargs.get("episodes", None)
        states, actions, next_states, rewards, dones = episode
        self.epsilon -= (
            5 * self.epsilon / episodes_num if self.epsilon > 0 else 0
        )  
        for state, action, reward, next_state, done in zip(
            states, actions, rewards, next_states, dones
        ):
            self.optimize(state, action, reward, next_state, done, **kwargs)
        self.i += 1