import torch as th
import torch.nn as nn
import torch.nn.functional as F


class ConditionalRNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(ConditionalRNNAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        if self.args.use_rnn:
            self.rnn = nn.GRUCell(args.hidden_dim, args.hidden_dim)
        else:
            self.rnn = nn.Linear(args.hidden_dim, args.hidden_dim)

        self.fc2 = nn.Sequential(
            nn.Linear(args.hidden_dim + args.latent_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, args.n_actions)
        )

    def init_hidden(self):
        return self.fc1.weight.new(1, self.args.hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        x, latent = inputs
        x = F.relu(self.fc1(x))
        h_in = hidden_state.reshape(-1, self.args.hidden_dim)
        if self.args.use_rnn:
            h = self.rnn(x, h_in)
        else:
            h = F.relu(self.rnn(x))

        q = th.cat([h, latent], dim=-1)
        q = self.fc2(q)
        return q, h