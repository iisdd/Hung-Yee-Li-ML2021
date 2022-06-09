# 训练&保存模型,不同的环境可能要修改一下reward,所以分开来写了
# 这个任务的关键是网络的size
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



MAX_EPISODES = 1001
MAX_BUFFER = 100000

S_DIM = env.observation_space.shape[0]
A_DIM = env.action_space.n


print(' Env: ', ENV)
print(' State Dimension: ', S_DIM)
print(' Number of Action(discrete) : ', A_DIM)

batch_size=64

ram = buffer.ReplayBuffer(MAX_BUFFER)
trainer = train.Trainer(S_DIM, A_DIM, ram, batch_size=batch_size, learning_rate=0.0003, reward_decay = 0.99,
                        eps_end=0.01, )

RENDER = False
best_r = 100
last_10_r = [0]*10
total_reward = []
for ep in range(MAX_EPISODES):
    ep_r = 0
    ep_step = 0
    s = env.reset()
    if ep > MAX_EPISODES - 10: RENDER = True

    while 1:
        ep_step += 1
        if RENDER: env.render()
        s = np.float32(s)
        a = trainer.get_exploration_action(s)

        s_, r, done, info = env.step(a)

        # if done:
        #     next_state = None
        # else:
        if ep_step > 2000:
            r -= 200
            done = True
        next_state = np.float32(s_)
        ram.add(s, a, r, next_state)

        s = s_
        ep_r += r
        if len(ram.buffer) > batch_size:
            trainer.optimize()
        if done:
            print('ep: ', ep, '  reward: %.2f' % ep_r, ' steps: ', ep_step)
            last_10_r.pop(0)
            last_10_r.append(ep_r)
            total_reward.append(np.mean(last_10_r))
            if np.mean(last_10_r) > best_r:
                best_r = np.mean(last_10_r)
                print('最近10个ep平均reward: ', np.mean(last_10_r), end='\t')
                trainer.save_models(episode=ep, env=ENV)
            break
    # 最后一个ep存一下
    # if ep % (MAX_EPISODES//5) == 0:
    #     trainer.save_models(episode=ep, env=ENV)
    gc.collect() # 清内存


print('training completed')

import matplotlib.pyplot as plt
plt.plot(total_reward)
plt.title('average reward')
plt.show()
print('后100eps平均rewards: ', np.mean(total_reward[-100: ]))