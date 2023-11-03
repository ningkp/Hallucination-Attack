import gym
from gym import spaces
import numpy as np
import random
from config import ModelConfig
import torch.nn.functional as F
from utils import load_model_and_tokenizer, complete_input, extract_model_embedding, random_init

class LLMEnvironment(gym.Env):
    def __init__(self, model_name, target_text, length=20, device='cuda:1'):
        super(LLMEnvironment, self).__init__()
        self.device = device

        try:
            self.model_config = getattr(ModelConfig, model_name)[0]
        except AttributeError:
            raise NotImplementedError

        self.model, self.tokenizer = load_model_and_tokenizer(
            self.model_config['path'], self.device, False
        )
        self.model_embed = extract_model_embedding(self.model)

        # 定义状态空间为tokenizer的空间
        self.initial_text = random_init(model_name, length=length)
        self.state_space = list(range(len(self.tokenizer.get_vocab())))
        self.action_space = spaces.Tuple((spaces.Discrete(length), spaces.Discrete(len(self.tokenizer.get_vocab()))))
        self.model_name = model_name

        # 初始化环境
        self.current_state = self.initial_text
        self.current_state_ids = self.tokenizer(
            self.current_state, truncation=True, return_tensors='pt'
        ).input_ids[0].to(self.device)
        self.current_state_onehot = F.one_hot(self.current_state_ids, num_classes=len(self.state_space))
        self.current_state_embeds = self.extract_current_state_embed()
        self.target_state = target_text
        self.current_loss = self.calculate_loss()

    def reset(self):
        # 重置环境为初始状态
        self.current_state = self.initial_text
        self.current_state_ids = self.tokenizer(
            self.current_state, truncation=True, return_tensors='pt'
        ).input_ids[0].to(self.device)
        self.current_state_onehot = F.one_hot(self.current_state_ids, num_classes=len(self.state_space))
        self.current_state_embeds = self.extract_current_state_embed()
        self.current_loss = self.calculate_loss()
        return self.current_state, self.current_state_ids, self.current_state_onehot, self.current_state_embeds

    def step(self, action):
        # 执行动作：替换token
        position, new_token_index = action
        self.current_state_ids[position+1] = self.state_space[new_token_index]  # 起始位置不能替换
        self.current_state = self.tokenizer.decode(
            self.current_state_ids,
            skip_special_tokens=True,
        )
        self.current_state_onehot = F.one_hot(self.current_state_ids, num_classes=len(self.state_space))
        self.current_state_embeds = self.extract_current_state_embed()

        # 计算新的损失
        new_loss = self.calculate_loss()

        # 计算奖励为损失变化
        reward = self.current_loss - new_loss
        self.current_loss = new_loss

        # 判断是否达到目标状态
        done = self.current_state == self.target_state

        # 返回新状态、奖励、是否结束和其他信息
        return self.current_state, self.current_state_ids, self.current_state_onehot, self.current_state_embeds, reward, done, {}

    def calculate_loss(self):
        # 切片
        self.slice()
        # 根据 current_state 和 target_state 来计算loss
        input_str = complete_input(self.model_config, self.current_state)
        if input_str.endswith(':'):
            input_str += ' '
        input_str += self.target_state
        input_ids = self.tokenizer(
            input_str, truncation=True, return_tensors='pt'
        ).input_ids[0].to(self.device)

        logits = self.model(input_ids=input_ids.unsqueeze(0))[0]
        if logits.dim() > 2:
            logits = logits.squeeze()
        try:
            assert input_ids.shape[0] >= self.target_slice.stop
        except AssertionError:
            self.target_slice = slice(self.target_slice.start, input_ids.shape[0])

        compute_logits = logits[self.target_slice.start - 1: self.target_slice.stop - 1]
        target = input_ids[self.target_slice]
        loss = F.cross_entropy(compute_logits, target)
        return loss

    def slice(self):
        prefix = self.model_config.get('prefix', '')
        prompt = self.model_config.get('prompt', '')
        suffix = self.model_config.get('suffix', '')
        temp_str = prefix+prompt
        temp_tokens = self.tokenizer(temp_str).input_ids
        len1 = len(temp_tokens)
        temp_str += self.current_state
        temp_tokens = self.tokenizer(temp_str).input_ids
        self.input_slice = slice(len1, len(temp_tokens))
        try:
            assert self.tokenizer.decode(temp_tokens[self.input_slice]) == self.current_state
        except AssertionError:
            self.input_slice = slice(self.input_slice.start-1, self.input_slice.stop)
            try:
                assert self.tokenizer.decode(temp_tokens[self.input_slice]) == self.current_state
            except AssertionError:
                if self.tokenizer.decode(temp_tokens[self.input_slice]).lstrip() != self.current_state:
                    ### Todo
                    raise NotImplementedError

        temp_str += suffix
        temp_tokens = self.tokenizer(temp_str).input_ids
        len2 = len(temp_tokens)
        if suffix.endswith(':'):
            temp_str += ' '
        temp_str += self.target_state
        temp_tokens = self.tokenizer(temp_str).input_ids
        self.target_slice = slice(len2, len(temp_tokens))

    def extract_current_state_embed(self):
        state_embeds = self.model_embed(self.current_state_ids.unsqueeze(0)).detach()
        return state_embeds.view(-1)




if __name__ == '__main__':
    target_text = 'This is an example text'
    model_name = 'vicuna'
    env = LLMEnvironment(model_name, target_text)

    # 重置环境
    state, state_ids, state_onehot, state_embeds = env.reset()
    print("初始状态:", state)

    # 执行两个动作
    action1 = (2, 6)  # 假设你要替换第三个token为第七个token
    action2 = (4, 1)  # 假设你要替换第五个token为第二个token
    state, state_ids, state_onehot, state_embeds, reward, done, _ = env.step(action1)
    print("新状态:", state)
    print("新状态ids:", state_ids)
    print(state_embeds)
    print("奖励:", reward)
    state, state_ids, state_onehot, state_embeds, reward, done, _ = env.step(action2)
    print("新状态:", state)
    print("新状态ids:", state_ids)
    print(state_embeds)
    print("奖励:", reward)
