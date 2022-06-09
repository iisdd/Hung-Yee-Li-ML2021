# 平均reward: 266.3
# 训练&保存模型
import gym
import numpy as np
import gc # 清理内存的
import torch
import train
import buffer
import random
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

seed = 543 # Do not change this
def fix(env, seed):
  env.seed(seed)
  env.action_space.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  np.random.seed(seed)
  random.seed(seed)



ENV = 'LunarLander-v2'     # 'CartPole-v0', 'MountainCar-v0', 'BipedalWalker-v2', 'LunarLander-v2'
env = gym.make(ENV)
env = env.unwrapped     # 还原env的原始设置，env外包了一层防作弊层

fix(env, seed) # fix the environment Do not revise this !!!

MAX_BUFFER = 10000

S_DIM = env.observation_space.shape[0]
A_DIM = env.action_space.n


print(' Env: ', ENV)
print(' State Dimension: ', S_DIM)
print(' Number of Action(discrete) : ', A_DIM)


ram = buffer.ReplayBuffer(MAX_BUFFER)
trainer = train.Trainer(S_DIM, A_DIM, ram, epsilon=0.01)
trainer.load_models(episode=999, env=ENV)


total_reward = []
for ep in range(10):
    ep_r = 0
    ep_step = 0
    s = env.reset()

    while 1:
        env.render()                          # 太费时间了...
        ep_step += 1
        s = np.float32(s)
        # a = trainer.get_exploration_action(s)  # 这里不随机的话会卡平台
        a = trainer.get_exploitation_action(s)
        s_, r, done, info = env.step(a)

        # 修改一下reward

        s = s_
        ep_r += r

        if done:
            print('ep: ', ep, '  reward: %.2f' % ep_r, ' steps: ', ep_step)
            total_reward.append(ep_r)
            break

    gc.collect()                                # 清内存

print('demo completed')
print('平均reward: ', np.mean(total_reward))
import matplotlib.pyplot as plt
plt.plot(total_reward)
plt.show()
