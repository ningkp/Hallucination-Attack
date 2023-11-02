import gym
import torch
import argparse
import numpy as np
import torch.optim as optim
from RL.model import Actor, Critic
from RL.utils import get_action
from collections import deque
from RL.running_state import ZFilter
from RL.hparams import HyperParams as hp
import matplotlib.pyplot as plt
from LLME import LLMEnvironment

parser = argparse.ArgumentParser()
parser.add_argument('--algorithm', type=str, default='PPO',
                    help='select one of algorithms among Vanilla_PG, NPG, TPRO, PPO')
args = parser.parse_args()

if args.algorithm == "PG":
    from RL.vanila_pg import train_model
elif args.algorithm == "NPG":
    from RL.npg import train_model
elif args.algorithm == "TRPO":
    from RL.trpo import train_model
elif args.algorithm == "PPO":
    from RL.ppo import train_model

if __name__ == "__main__":
    target_text = 'This is an example text'
    model_name = 'vicuna'
    env = LLMEnvironment(model_name, target_text)
    env.seed(500)
    torch.manual_seed(500)

    num_inputs = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]

    print('state size:', num_inputs)
    print('action size:', num_actions)

    actor = Actor(num_inputs, num_actions)
    critic = Critic(num_inputs)

    actor_optim = optim.Adam(actor.parameters(), lr=hp.actor_lr)
    critic_optim = optim.Adam(critic.parameters(), lr=hp.critic_lr,
                              weight_decay=hp.l2_rate)

    running_state = ZFilter((num_inputs,), clip=5)
    episodes = 0
    xar = []
    yar = []
    for iter in range(50):
        actor.eval(), critic.eval()
        memory = deque()

        steps = 0
        scores = []
        while steps < 2048:
            episodes += 1
            state = env.reset()
            state = running_state(state)
            score = 0
            for _ in range(10000):

                steps += 1
                mu, std, _ = actor(torch.Tensor(state).unsqueeze(0))
                action = get_action(mu, std)[0]
                next_state, reward, done, _ = env.step(action)
                next_state = running_state(next_state)

                if done:
                    mask = 0
                else:
                    mask = 1

                memory.append([state, action, reward, mask])

                score += reward
                state = next_state

                if done:
                    break
            scores.append(score)

        score_avg = np.mean(scores)
        print('{} episode score is {:.2f}'.format(episodes, score_avg))

        with open('reward per iter.txt', 'w') as file:
            file.write(str(episodes) + "," + str(score_avg))
            file.write("\n")
        file.close
        xar.append(int(episodes))
        yar.append(int(score_avg))

        actor.train(), critic.train()
        train_model(actor, critic, memory, actor_optim, critic_optim)


    def plotting():
        plt.plot(xar, yar, linewidth=3)
        plt.title("Avg score/Episodes", fontsize=19)
        plt.xlabel("Episodes", fontsize=10)
        plt.ylabel("Avg score", fontsize=10)
        plt.tick_params(axis='both', labelsize=9)
        plt.show()


    plotting()
    print(xar, '\n', yar)
