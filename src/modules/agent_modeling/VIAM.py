import torch as th
import torch.nn as nn
from modules.agent_modeling.autoencoder.ae_model import ProbabilisticEncoder, Decoder


class VIAM(nn.Module):
    def __init__(self, scheme, args):
        super(VIAM, self).__init__()
        self.args = args
        self.n_agents = args.n_agents

        # 1.Built Encoder
        encoder_input_shape = self._get_encoder_shape(scheme)
        self.encoder = ProbabilisticEncoder(encoder_input_shape, args)
        self.hidden_states = None

        # 2.Built Decoder
        decoder_input_shape, obs_head_output_shape, rew_head_output_shape = self._get_decoder_shape(scheme)
        self.decoder = Decoder(decoder_input_shape, [obs_head_output_shape, rew_head_output_shape], args)

    def forward(self, ep_batch, t):
        encoder_inputs = self._build_encoder_inputs(ep_batch, t)
        x, self.hidden_states = self.encoder(encoder_inputs, self.hidden_states)
        latent, mu, log_var = x
        return latent, mu, log_var

    def am_loss(self, batch):
        seq_len = batch.max_seq_length - 1
        # Encoder
        mus = []
        log_vars = []
        latents = []
        self.init_hidden(batch.batch_size)
        for t in range(seq_len):
            latent, mu, log_var = self.forward(batch, t)
            mus.append(mu)
            log_vars.append(log_var)
            latents.append(latent)
        mus = th.stack(mus, dim=1)
        log_vars = th.stack(log_vars, dim=1)
        latents = th.stack(latents, dim=1)

        # Decoder
        inputs, targets = self._build_decoder_inputs_and_target(batch) # (bs*n, seq_len,*shape)

        # Loss
        rec_obs_loss, rec_rew_loss = self.recon_loss(latents, inputs, targets)
        kl_loss = self.kl_loss(mus, log_vars)

        rec_loss = [rec_obs_loss, rec_rew_loss]
        return rec_loss, kl_loss

    def recon_loss(self, latent, inputs, targets):
        out = self.decoder(inputs, latent)
        rec_obs_loss = (targets[0] - out[0]).pow(2).mean()
        rec_rew_loss = (targets[1] - out[1]).pow(2).mean()
        return rec_obs_loss, rec_rew_loss

    def kl_loss(self, mu, log_var):
        kl_loss = (- 0.5 * th.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=-1)).mean()
        return kl_loss

    def init_hidden(self, batch_size):
        self.hidden_states = self.encoder.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav

    def _build_encoder_inputs(self, batch, t):
        bs = batch.batch_size
        inputs = []

        inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))
        if self.args.encoder_input_obs:
            inputs.append(batch["obs"][:, t])
        if t == 0:
            if self.args.encoder_input_act:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            if self.args.encoder_input_rew:
                inputs.append(th.zeros_like(batch["reward"][:, t]).unsqueeze(-2).expand(-1, self.n_agents, -1))
            if self.args.encoder_input_terminated:
                inputs.append(th.zeros_like(batch["terminated"][:, t]).unsqueeze(-2).expand(-1, self.n_agents, -1))
        else:
            if self.args.encoder_input_act:
                inputs.append(batch["actions_onehot"][:, t - 1])
            if self.args.encoder_input_rew:
                inputs.append(batch["reward"][:, t - 1].unsqueeze(-2).expand(-1, self.n_agents, -1))
            if self.args.encoder_input_terminated:
                inputs.append(batch["terminated"][:, t - 1].unsqueeze(-2).expand(-1, self.n_agents, -1))

        inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=-1)
        return inputs

    def _build_decoder_inputs_and_target(self, batch, t=None):
        bs = batch.batch_size
        seq_len = batch.max_seq_length-1

        # 1.construct decoder input
        inputs = []
        if self.args.decoder_input_obs:
            inputs.append(batch["obs"][:, :-1])
        if self.args.decoder_input_act:
            inputs.append(batch["actions_onehot"][:, :-1])
        inputs = th.cat([x.permute(0, 2, 1, 3).reshape(bs * self.n_agents, seq_len, -1) for x in inputs], dim=-1)

        # 2.construct decoder target, 1,...,N
        targets = []
        if self.args.rec_obs:
            target_obs = batch["obs"][:, 1:].permute(0, 2, 1, 3).reshape(bs * self.n_agents, seq_len, -1)
            targets.append(target_obs)
        if self.args.rec_rew:
            target_rew = batch["reward"][:, :-1].unsqueeze(-2).expand(-1, -1, self.n_agents, -1).permute(0, 2, 1, 3).reshape(bs * self.n_agents, seq_len, -1)
            targets.append(target_rew)

        return inputs, targets

    def _get_encoder_shape(self, scheme):
        input_shape = 0
        if self.args.obs_agent_id:
            input_shape += self.args.n_agents
        if self.args.encoder_input_obs:
            input_shape += scheme["obs"]["vshape"]
        if self.args.encoder_input_act:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.encoder_input_rew:
            input_shape += scheme["reward"]["vshape"][0]
        if self.args.encoder_input_terminated:
            input_shape += scheme["terminated"]["vshape"][0]
        return input_shape

    def _get_decoder_shape(self, scheme):
        decoder_input_shape = 0
        if self.args.decoder_input_obs:
            decoder_input_shape += scheme["obs"]["vshape"]
        if self.args.decoder_input_act:
            decoder_input_shape += scheme["actions_onehot"]["vshape"][0]

        obs_head_output_shape = scheme["obs"]["vshape"]
        rew_head_output_shape = scheme["reward"]["vshape"][0]

        return decoder_input_shape, obs_head_output_shape, rew_head_output_shape