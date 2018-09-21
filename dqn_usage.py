from dqn_keras import DQN
import gym

env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = DQN(state_size, action_size,epsilon = 0.9, epsilon_decrease=True, epsilon_min=0.2)
done = False
batch_size = 32
max_time=250
episodes=5
scores=[]
for e in range(episodes):
    state = env.reset()
    for time in range(max_time):
        env.render()
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        if done:
            reward = -10   #if done we have to punish the agent with -10
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            scores.append(time)
            print("episode: {}/{}, score: {}"
                    .format(e+1, episodes, time))
            break
        agent.learn_from_memory(batch_size)
print(scores)





model=agent.get_model()
agent = DQN(state_size, action_size,model_test =True)
agent.set_model(model)

for e in range(episodes):
    state = env.reset()
    for time in range(max_time):
        env.render()
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        if done:
            scores.append(time)
            print("episode: {}/{}, score: {}"
                    .format(e+1, episodes, time))
            break