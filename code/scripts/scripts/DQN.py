from __future__ import print_function
import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class BasicBuffer:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def push(self, experience):
        self.buffer.append(experience)

    def clear(self):
        self.buffer.clear()

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def sample_bc(self, batch_size):
        state_batch = []
        ref_action_batch = []
        batch = random.sample(self.buffer, batch_size)
        for experience in batch:
            state, ref_action = experience
            state_batch.append(state)
            ref_action_batch.append(ref_action)

        return state_batch, ref_action_batch

    def sample_sequence(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        min_start = len(self.buffer) - batch_size
        start = np.random.randint(0, min_start)

        for sample in range(start, start + batch_size):
            state, action, reward, next_state, done = self.buffer[sample]
            # state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def __len__(self):
        return len(self.buffer)


class AuxBuffer(BasicBuffer):
    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []
        density_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, done, density = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)
            density_batch.append(density)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch, density_batch

    def sample_bc(self, batch_size):
        state_batch = []
        ref_action_batch = []
        density_batch = []
        batch = random.sample(self.buffer, batch_size)
        for experience in batch:
            state, ref_action, density = experience
            state_batch.append(state)
            ref_action_batch.append(ref_action)
            density_batch.append(density)

        return state_batch, ref_action_batch, density_batch

    def sample_sequence(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []
        density_batch = []

        min_start = len(self.buffer) - batch_size
        start = np.random.randint(0, min_start)

        for sample in range(start, start + batch_size):
            state, action, reward, next_state, done, density = self.buffer[sample]
            # state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)
            density_batch.append(density)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch, density_batch


class DoubleBuffer(BasicBuffer):
    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []
        second_action_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, done, second_action = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)
            second_action_batch.append(second_action)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch, second_action_batch

    def sample_bc(self, batch_size):
        state_batch = []
        ref_action_batch = []
        second_action_batch = []
        batch = random.sample(self.buffer, batch_size)
        for experience in batch:
            state, ref_action, second_action = experience
            state_batch.append(state)
            ref_action_batch.append(ref_action)
            second_action_batch.append(second_action)

        return state_batch, ref_action_batch, second_action_batch

    def sample_sequence(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []
        second_action_batch = []

        min_start = len(self.buffer) - batch_size
        start = np.random.randint(0, min_start)

        for sample in range(start, start + batch_size):
            state, action, reward, next_state, done, second_action = self.buffer[sample]
            # state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)
            second_action_batch.append(second_action)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch, second_action_batch


class MultiBuffer(BasicBuffer):
    def sample(self, batch_size):
        state_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []
        depth_batch = []
        freq_batch = []
        cost_batch = []
        block_batch = []
        prefer_batch = []
        inflation_batch = []
        pose_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, reward, next_state, done, depth, freq, cost, block, prefer, inflation, pose = experience
            state_batch.append(state)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)
            depth_batch.append(depth)
            freq_batch.append(freq)
            cost_batch.append(cost)
            block_batch.append(block)
            prefer_batch.append(prefer)
            inflation_batch.append(inflation)
            pose_batch.append(pose)

        return state_batch, reward_batch, next_state_batch, done_batch, depth_batch, freq_batch, cost_batch, \
               block_batch, prefer_batch, inflation_batch, pose_batch

    def sample_bc(self, batch_size):
        state_batch = []
        ref_action_batch = []
        second_action_batch = []
        batch = random.sample(self.buffer, batch_size)
        for experience in batch:
            state, ref_action, second_action = experience
            state_batch.append(state)
            ref_action_batch.append(ref_action)
            second_action_batch.append(second_action)

        return state_batch, ref_action_batch, second_action_batch

    def sample_sequence(self, batch_size):
        state_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []
        depth_batch = []
        freq_batch = []
        cost_batch = []
        block_batch = []
        prefer_batch = []
        inflation_batch = []
        pose_batch = []

        min_start = len(self.buffer) - batch_size
        start = np.random.randint(0, min_start)

        for sample in range(start, start + batch_size):
            state, reward, next_state, done, depth, freq, cost, block, prefer, inflation, pose = self.buffer[sample]
            # state, action, reward, next_state, done = experience
            state_batch.append(state)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)
            depth_batch.append(depth)
            freq_batch.append(freq)
            cost_batch.append(cost)
            block_batch.append(block)
            prefer_batch.append(prefer)
            inflation_batch.append(inflation)
            pose_batch.append(pose)

        return state_batch, reward_batch, next_state_batch, done_batch, depth_batch, freq_batch, cost_batch, \
               block_batch, prefer_batch, inflation_batch, pose_batch


class SimpleCNN(nn.Module):
    def __init__(self, output_size):
        super(SimpleCNN, self).__init__()

        # kernel size for different CNN layers
        self._cnn_layers_kernel_size = [(8, 8), (4, 4), (3, 3)]

        # strides for different CNN layers
        self._cnn_layers_stride = [(4, 4), (2, 2), (1, 1)]
        cnn_dims = np.array([240, 320])
        for kernel_size, stride in zip(
            self._cnn_layers_kernel_size, self._cnn_layers_stride
        ):
            cnn_dims = self._conv_output_dim(
                dimension=cnn_dims,
                padding=np.array([0, 0], dtype=np.float32),
                dilation=np.array([1, 1], dtype=np.float32),
                kernel_size=np.array(kernel_size, dtype=np.float32),
                stride=np.array(stride, dtype=np.float32),
            )

        self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=self._cnn_layers_kernel_size[0],
                stride=self._cnn_layers_stride[0],
            ),
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=self._cnn_layers_kernel_size[1],
                stride=self._cnn_layers_stride[1],
            ),
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=64,
                out_channels=32,
                kernel_size=self._cnn_layers_kernel_size[2],
                stride=self._cnn_layers_stride[2],
            ),
            #  nn.ReLU(True),
            Flatten(),
            nn.Linear(32 * cnn_dims[0] * cnn_dims[1], output_size),
            nn.ReLU(True),
        )

        self.layer_init()

    def _conv_output_dim(
        self, dimension, padding, dilation, kernel_size, stride
    ):
        r"""Calculates the output height and width based on the input
        height and width to the convolution layer.

        ref: https://pytorch.org/docs/master/nn.html#torch.nn.Conv2d
        """
        assert len(dimension) == 2
        out_dimension = []
        for i in range(len(dimension)):
            out_dimension.append(
                int(
                    np.floor(
                        (
                            (
                                dimension[i]
                                + 2 * padding[i]
                                - dilation[i] * (kernel_size[i] - 1)
                                - 1
                            )
                            / stride[i]
                        )
                        + 1
                    )
                )
            )
        return tuple(out_dimension)

    def layer_init(self):
        for layer in self.cnn:
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(
                    layer.weight, nn.init.calculate_gain("relu")
                )
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)

    def forward(self, cnn_input):
        # cnn_input = F.avg_pool2d(cnn_input, 2)
        device = next(self.cnn.parameters()).device
        cnn_input = cnn_input.to(device)

        return self.cnn(cnn_input)


class DQN(nn.Module):

    def __init__(self, input_dim, output_dim, aux=False, double=False):
        super(DQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.aux = aux
        self.double = double

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 160),
            nn.ReLU(),
            nn.Linear(160, 128),
            nn.ReLU(),
            # nn.Linear(128, self.output_dim)
        )
        self.fc2 = nn.Sequential(
            # nn.Linear(128, 64),
            # nn.ReLU(),
            nn.Linear(128, self.output_dim)
        )
        if self.aux:
            self.density = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 4)
            )
        if self.double:
            self.second = nn.Sequential(
                # nn.Linear(128, 64),
                # nn.ReLU(),
                nn.Linear(128, 5)
            )
        for layer in self.fc:
            if hasattr(layer, 'weight'):
                nn.init.orthogonal_(layer.weight)
            if hasattr(layer, 'bias'):
                nn.init.constant_(layer.bias, 0)

    def forward(self, state):
        # state = self.fc(state)
        # qvals = self.fc2(state)
        state = self.fc(state)
        qvals = self.fc2(state)
        if self.aux:
            density = self.density(state)
            return qvals, density
        if self.double:
            second = self.second(state)
            return qvals, second
        return qvals


class MultiDQN(nn.Module):

    def __init__(self, input_dim):
        super(MultiDQN, self).__init__()
        self.input_dim = input_dim

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 160),
            nn.ReLU(),
            nn.Linear(160, 128),
            nn.ReLU(),
        )

        self.fcs = [nn.Linear(128, 11),]
        for _ in range(6):
            self.fcs.append(nn.Linear(128, 6))


        # for layer in self.fc:
        #     if hasattr(layer, 'weight'):
        #         nn.init.orthogonal_(layer.weight)
        #     if hasattr(layer, 'bias'):
        #         nn.init.constant_(layer.bias, 0)

    def forward(self, state):
        state = self.fc(state)
        qvals = []
        for layer in self.fcs:
            pred = layer(state)
            qval = pred[..., :-1] + pred[..., -1].unsqueeze(-1)
            qvals.append(qval)
        return qvals


class DQNAgent:
    def __init__(self, state_space, action_space, learning_rate=3e-4, gamma=0.99,
                 buffer_size=10000, eps=1, eps_decay=.99, aux=False, double=False):
        self.state_space = state_space
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.eps = eps
        self.eps_decay = eps_decay
        self.aux = aux
        self.double = double
        if aux:
            self.replay_buffer = AuxBuffer(max_size=buffer_size)
        elif double:
            self.replay_buffer = DoubleBuffer(max_size=buffer_size)
        else:
            self.replay_buffer = BasicBuffer(max_size=buffer_size)

        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device('cpu')

        self.model = DQN(state_space, action_space, aux, double).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.MSE_loss = nn.MSELoss()
        self.CE_loss = nn.CrossEntropyLoss()

    def act(self, state):
        if (np.random.random() < self.eps):
            action = np.random.choice(np.arange(self.action_space))
            if self.double:
                action2 = np.random.choice(np.arange(5))
                return action, action2
        else:
            state = torch.FloatTensor(state, device=self.device)
            if self.aux:
                qvals, _ = self.model(state)
            elif self.double:
                qvals1, qvals2 = self.model(state)
                action1 = torch.argmax(qvals1.squeeze()).item()
                action2 = torch.argmax(qvals2.squeeze()).item()
                return action1, action2
            else:
                qvals = self.model(state)
            # print(qvals.squeeze().shape)
            action = torch.argmax(qvals.squeeze()).item()
            self.eps *= self.eps_decay
        return action

    def compute_loss(self, batch):
        if self.aux:
            states, actions, rewards, next_states, dones, density = batch
            density = torch.LongTensor(density, device=self.device)
        elif self.double:
            states, actions, rewards, next_states, dones, second = batch
            second = torch.LongTensor(second, device=self.device)
        else:
            states, actions, rewards, next_states, dones = batch
        states = torch.FloatTensor(states, device=self.device)
        actions = torch.LongTensor(actions, device=self.device)
        rewards = torch.FloatTensor(rewards, device=self.device)
        next_states = torch.FloatTensor(next_states, device=self.device)
        dones = torch.FloatTensor(dones, device=self.device)

        if self.aux:
            curr_Q, pred_d = self.model(states)
            curr_Q = curr_Q.squeeze().gather(1, actions.unsqueeze(1))
            density_loss = self.CE_loss(pred_d.squeeze(), density)
            next_Q, _ = self.model(next_states)
            next_Q = next_Q.squeeze(1)
        elif self.double:
            curr_Q, curr_Q2 = self.model(states)
            curr_Q = curr_Q.squeeze().gather(1, actions.unsqueeze(1))
            curr_Q2 = curr_Q2.squeeze().gather(1, second.unsqueeze(1))
            next_Q, next_Q2 = self.model(next_states)
            next_Q = next_Q.squeeze(1)
            next_Q2 = next_Q2.squeeze(1)
            curr_Q2 = curr_Q2.squeeze(1)
            max_next_Q2 = torch.max(next_Q2, 1)[0]
            expected_Q2 = rewards.squeeze(1) + self.gamma * max_next_Q2 * (1 - dones)
            density_loss = self.MSE_loss(curr_Q2, expected_Q2)
        else:
            curr_Q = self.model(states).squeeze().gather(1, actions.unsqueeze(1))
            next_Q = self.model(next_states).squeeze(1)
            density_loss = 0
        curr_Q = curr_Q.squeeze(1)
        max_next_Q = torch.max(next_Q, 1)[0]
        expected_Q = rewards.squeeze(1) + self.gamma * max_next_Q * (1-dones)

        loss = self.MSE_loss(curr_Q, expected_Q) + density_loss
        return loss

    def replay(self, batch_size):
        batch = self.replay_buffer.sample(batch_size)
        loss = self.compute_loss(batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def remember(self, experience):
        self.replay_buffer.push(experience)

    def reset(self):
        self.replay_buffer.clear()
        self.optimizer = torch.optim.Adam(self.model.parameters())

    def behavior_loss(self, batch):
        if self.aux:
            states, ref_actions, density = batch
            density = torch.LongTensor(density, device=self.device)
        elif self.double:
            states, ref_actions, second = batch
            second = torch.LongTensor(second, device=self.device)
        else:
            states, ref_actions = batch
        states = torch.FloatTensor(states, device=self.device)
        ref_actions = torch.LongTensor(ref_actions, device=self.device)
        if self.aux:
            curr_Q, pred_d = self.model(states)
            density_loss = self.CE_loss(pred_d.squeeze(), density)
        elif self.double:
            curr_Q, curr_Q2 = self.model(states)
            density_loss = self.CE_loss(curr_Q2.squeeze(), second)
        else:
            curr_Q = self.model(states)
            density_loss = 0
        loss = self.CE_loss(curr_Q.squeeze(), ref_actions) + density_loss

        return loss

    def behavior_clone(self, batch_size):
        batch = self.replay_buffer.sample_bc(batch_size)
        loss = self.behavior_loss(batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_model(self, path, epoch=0):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'state_space': self.state_space,
            'action_space': self.action_space,
        }, path)
        print('model saved: ' + path)

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.model = DQN(checkpoint['state_space'], checkpoint['action_space'], double=self.double)
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()


class MultiDQNAgent:
    def __init__(self, state_space, learning_rate=3e-4, gamma=0.99,
                 buffer_size=6400, eps=1, eps_decay=.9993, aux=False, double=False):
        self.state_space = state_space
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.eps = eps
        self.eps_decay = eps_decay
        self.aux = aux
        self.double = double
        self.replay_buffer = MultiBuffer(max_size=buffer_size)

        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device('cpu')

        self.model = MultiDQN(state_space).to(self.device)
        self.target_model = MultiDQN(state_space).to(self.device)
        self.iter = 0
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.MSE_loss = nn.MSELoss()

    def act(self, state):
        if (np.random.random() < self.eps):
            action = [np.random.choice(np.arange(10))]
            for _ in range(6):
                action.append(np.random.choice(np.arange(5)))
        else:
            state = torch.FloatTensor(state, device=self.device)
            qvals = self.model(state)
            # print(qvals.squeeze().shape)
            action = [torch.argmax(q.squeeze()).item() for q in qvals]
            self.eps *= self.eps_decay
        return action

    def compute_loss(self, batch):
        states, rewards, next_states, dones, depths, freqs, costs, blocks, prefers, inflations, poses = batch
        states = torch.FloatTensor(states, device=self.device)
        rewards = torch.FloatTensor(rewards, device=self.device)
        next_states = torch.FloatTensor(next_states, device=self.device)
        dones = torch.FloatTensor(dones, device=self.device)
        depths = torch.LongTensor(depths, device=self.device)
        freqs = torch.LongTensor(freqs, device=self.device)
        costs = torch.LongTensor(costs, device=self.device)
        blocks = torch.LongTensor(blocks, device=self.device)
        prefers = torch.LongTensor(prefers, device=self.device)
        inflations = torch.LongTensor(inflations, device=self.device)
        poses = torch.LongTensor(poses, device=self.device)
        actions = [depths, freqs, costs, blocks, prefers, inflations, poses]
        curr_Qs = self.model(states)
        next_Qs = self.model(next_states)
        estimated_Qs = self.target_model(next_states)
        max_next_Qs = [0] * len(curr_Qs)
        expected_Qs = [0] * len(curr_Qs)
        loss = 0
        for i in range(len(curr_Qs)):
            curr_Qs[i] = curr_Qs[i].squeeze().gather(1, actions[i].unsqueeze(1)).squeeze(1)
            next_Qs[i] = next_Qs[i].squeeze(1)
            max_as = torch.argmax(next_Qs[i], 1)
            ind = torch.arange(len(estimated_Qs[i]))
            max_next_Qs[i] = estimated_Qs[i][ind, 0, max_as]
            # max_next_Qs[i] = torch.max(next_Qs[i], 1)[0]
            expected_Qs[i] = rewards.squeeze(1) + self.gamma * max_next_Qs[i] * (1 - dones)
            # print(curr_Qs[i].shape, expected_Qs[i].shape)
            loss += self.MSE_loss(curr_Qs[i], expected_Qs[i])
        return loss

    def update(self):
        # import rospy
        # rospy.logwarn('update')
        self.target_model.load_state_dict(self.model.state_dict())

    def replay(self, batch_size):
        batch = self.replay_buffer.sample(batch_size)
        loss = self.compute_loss(batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.iter += 1

    def remember(self, experience):
        self.replay_buffer.push(experience)

    def reset(self):
        self.replay_buffer.clear()
        self.optimizer = torch.optim.Adam(self.model.parameters())

    # def behavior_loss(self, batch):
    #     if self.aux:
    #         states, ref_actions, density = batch
    #         density = torch.LongTensor(density, device=self.device)
    #     elif self.double:
    #         states, ref_actions, second = batch
    #         second = torch.LongTensor(second, device=self.device)
    #     else:
    #         states, ref_actions = batch
    #     states = torch.FloatTensor(states, device=self.device)
    #     ref_actions = torch.LongTensor(ref_actions, device=self.device)
    #     if self.aux:
    #         curr_Q, pred_d = self.model(states)
    #         density_loss = self.CE_loss(pred_d.squeeze(), density)
    #     elif self.double:
    #         curr_Q, curr_Q2 = self.model(states)
    #         density_loss = self.CE_loss(curr_Q2.squeeze(), second)
    #     else:
    #         curr_Q = self.model(states)
    #         density_loss = 0
    #     loss = self.CE_loss(curr_Q.squeeze(), ref_actions) + density_loss
    #
    #     return loss
    #
    # def behavior_clone(self, batch_size):
    #     batch = self.replay_buffer.sample_bc(batch_size)
    #     loss = self.behavior_loss(batch)
    #
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()

    def save_model(self, path, epoch=0):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'state_space': self.state_space,
        }, path)
        print('model saved: ' + path)

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.model = MultiDQN(checkpoint['state_space'])
        self.target_model = MultiDQN(checkpoint['state_space'])
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()


class CNNDQNAgent(DQNAgent):
    def __init__(self, state_space, action_space, learning_rate=3e-4, gamma=0.99, buffer_size=10000, eps=1,
                 eps_decay=.99, aux=False):
        DQNAgent.__init__(self, state_space, action_space, learning_rate, gamma, buffer_size, eps, eps_decay, aux)
        self.model = CNNDQN(state_space, action_space, aux).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters())

    def load_model(self, path):
        checkpoint = torch.load(path)
        # self.model = CNNDQN(checkpoint['state_space'], checkpoint['action_space'])
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
