"""
Reinforcement learning maze example.

Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].

This script is the main part which controls the update method of this example.
The RL is in RL_brain.py.

View more on my tutorial page: htt ps://morvanzhou.github.io/tutorials/
"""

from maze_env import Maze
from RL_brain import QLearningTable


def update():
    for episode in range(100):  # 循环100个回合
        # initial observation
        observation = env.reset()

        while True:
            # fresh env
            env.render()    # 环境更新

            # RL choose action based on observation
            # QLearning只会估计下一时刻的状态，而不会采取状态
            action = RL.choose_action(str(observation))

            # RL take action and get next observation and reward
            # 观察量，observation_表示下一个状态
            observation_, reward, done = env.step(action)

            # RL learn from this transition
            # 需要考虑下一个action，从当前行动以及回报还有当前观测模型中学习下一个观测
            RL.learn(str(observation), action, reward, str(observation_))

            # swap observation
            # 把下一次观测值当成这一次的观测值
            observation = observation_

            # break while loop when end of this episode
            if done:
                break

    # end of game
    print('game over')
    env.destroy()


if __name__ == "__main__":
    env = Maze()
    RL = QLearningTable(actions=list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()
