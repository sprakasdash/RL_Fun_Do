import gym

from a2c import A2C_Agent

env = gym.make('CartPole-v0')
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.n 
MAX_EPISODE = 1500
MAX_STEPS = 500

lr = 1e-3
gamma = 0.99
agent = (A2C_Agent(env, gamma, lr))

def run():
    for episode in range(MAX_EPISODE):
        state = env.reset()
        trajectory = []
        eps_reward = 0
        for steps in range(MAX_STEPS):
            action = agent.get_action(state)
            obs, rew, done, info = env.step(action)
            trajectory.append([state, action, rew, obs, done])
            eps_reward += rew
            if done:
                break

            state = obs
        if episode % 10 == 0:
            print('Episode '+ str(episode) + ': ' + str(eps_reward))
        agent.update(trajectory)

run()
