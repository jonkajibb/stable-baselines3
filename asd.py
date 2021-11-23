import gym
import numpy as np
from numpy.core.fromnumeric import std
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy

'''
In this script I will try out Proximal Policy Optimization. PPO is an Actor-Critic method which uses a value function to
improve the policy gradient descent (by reducing the variance). It is an on-policy algorithm, meaning that it always uses the latest
policy to update the networks. It is usually less sample efficient than off-policy algs like DQN, SAC or TD3, but is faster in regards
to wall-clock time.
'''

def evaluate(model, num_episodes=100):
    '''
    Evaluate a RL agent
    @param model: the RL agent
    :param num_episodes: number of episodes to evaluate it
    :return: mean reward for the last num_episodes
    '''
    env = model.get_env()
    all_episodes_rewards=[]
    for i in range(num_episodes):
        episode_rewards=[]
        done=False
        obs = env.reset() #getting state/observation
        while not done:
            # states are only useful when using LSTM policies (?)
            action, _states = model.predict(obs)
            #action, states, rewards and dones are arrays
            # because we are using vectorized env
            obs, reward, done, info = env.step(action)
            episode_rewards.append(reward)
        
        all_episodes_rewards.append(sum(episode_rewards))
    
    mean_episode_rewards = np.mean(all_episodes_rewards)
    print("Mean reward: ", mean_episode_rewards, "Num episodes: ", num_episodes)
    return mean_episode_rewards

env = gym.make('CartPole-v1')
model = PPO(MlpPolicy, env, verbose=0) # MlpPolicy = multi-layered perceptron?

# Random agent, before training
mean_reward_before_train = evaluate(model, num_episodes=100)
# Mean reward:  23.1 Num episodes:  100, trash...

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
# also trash

# Lets train the agent for 10k steps
model.learn(total_timesteps=10000)

# Evaluation
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
