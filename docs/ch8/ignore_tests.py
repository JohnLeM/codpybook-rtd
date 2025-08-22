import codpy.KQLearning as KQLearning
import sys 
import numpy as np
import gymnasium as gym
import codpy.core as core

if __name__ == "__main__":
    # Instanciate your environment
    env = gym.make("CartPole-v1", render_mode="human")

    # The agent requires action and observation space dimensions
    agent = KQLearning.KQLearning(env.action_space.n, env.observation_space.shape[0], gamma=0.99)
    steps = 0
    games = 0

    while games < 500:
        # The agent requires to be passed the game trajectory
        states, actions, next_states, rewards, dones = [], [], [], [], []
        state, _ = env.reset()
        while steps < 1000:
            action = agent(state)
            next_state, reward, done, _, _ = env.step(action)
            # The agent expects the game to be passed in reverse order
            (
                states.insert(0, state),
                actions.insert(0, action),
                next_states.insert(0, next_state),
                rewards.insert(0, reward),
                dones.insert(0, done),
            )
            steps += 1
            state = next_state
            if done:
                break
        # You train your agent once at the end of every episode
        agent.train((states, actions, next_states, rewards, dones))
        games += 1
    env.close()

#########################################################################
# CartPole version
# ------------------------
#You can overwrite specific methods to customize the training process. 
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
# LunarLander Version
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