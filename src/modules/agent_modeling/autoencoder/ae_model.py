import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

class ProbabilisticEncoder(nn.Module):
    def __init__(self, input_shape, args):
        super(ProbabilisticEncoder, self).__init__()
        self.args = args
        self.latent_dim = args.latent_dim
        self.hidden_dim = args.vae_hidden_dim

        self.fc1 = nn.Linear(input_shape, self.hidden_dim)
        if self.args.use_vae_rnn:
            self.rnn = nn.GRUCell(self.hidden_dim, self.hidden_dim)
        else:
            self.rnn = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.latent_dim * 2)

    def init_hidden(self, batch_size=1):
        return self.fc1.weight.new(batch_size, self.hidden_dim).zero_()

    def forward(self, inputs, hiddens):
        x = F.relu(self.fc1(inputs))
        h_in = hiddens.reshape(-1, self.hidden_dim)
        if self.args.use_vae_rnn:
            h = self.rnn(x, h_in)
        else:
            h = F.relu(self.rnn(x))
        out = self.fc2(h)

        mu = out[...,:self.latent_dim]
        log_var = out[...,self.latent_dim:]
        std = th.clamp(th.exp(0.5 * log_var), min=0.01)
        gaussian_embed = D.Normal(mu, std)
        latent = gaussian_embed.rsample()

        return (latent, mu, log_var), h

class Decoder(nn.Module):
    def __init__(self, input_shape, output_shape, args):
        super(Decoder, self).__init__()

        self.hidden_dim = args.vae_hidden_dim
        self.latent_dim = args.latent_dim
        # shared
        self.fc1 = nn.Linear(input_shape, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim + self.latent_dim, self.hidden_dim)
        # decoder head
        self.heads = nn.ModuleList()
        for out_dim in output_shape:
            self.heads.append(nn.Linear(self.hidden_dim , out_dim))

    def forward(self, inputs, latent):
        h = F.relu(self.fc1(inputs))
        h = F.relu(self.fc2(h))

        h = th.cat([h, latent], dim=-1)
        h = F.relu(self.fc3(h))

        outputs = []
        for head in self.heads:
            outputs.append(head(h))
        return outputs
