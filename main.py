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
    env = LLMEnvironment(model_name, target_text, device='cuda:6')
    torch.manual_seed(500)
    state, state_ids, state_onehot, state_embeds = env.reset()

    num_inputs = state_embeds.shape[0]
    num_actions = state_embeds.shape[0] * len(env.state_space)

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
        while steps < 200:
            episodes += 1
            state, state_ids, state_onehot, state_embeds = env.reset()
            # state = running_state(state)
            score = 0
            for _ in range(10000):

                steps += 1
                mu, std, _ = actor(torch.Tensor(state_embeds).unsqueeze(0))
                action = get_action(mu, std)[0]
                next_state, next_state_ids, next_state_onehot, next_state_embeds, reward, done, _ = env.step(action)
                # next_state = running_state(next_state)

                if done:
                    mask = 0
                else:
                    mask = 1

                memory.append([state_embeds, action, reward, mask])

                score += reward
                state_embeds = next_state_embeds

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
