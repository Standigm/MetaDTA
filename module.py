import torch as t
import torch.nn as nn
import math

class Linear(nn.Module):
    """
    Linear Module
    """
    def __init__(self, in_dim, out_dim, bias=True, w_init='linear'):
        """
        :param in_dim: dimension of input
        :param out_dim: dimension of output
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Linear, self).__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)

        nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=nn.init.calculate_gain(w_init))

    def forward(self, x):
        return self.linear_layer(x)


class LatentEncoder(nn.Module):
    """
    Latent Encoder [For prior, posterior]
    """
    def __init__(self, num_hidden, num_latent, input_dim):
        super(LatentEncoder, self).__init__()
        self.input_projection = Linear(input_dim, num_hidden)
        self.self_attentions = nn.ModuleList([Attention(num_hidden) for _ in range(2)])
        self.penultimate_layer = Linear(num_hidden, num_hidden, w_init='relu')
        self.mu = Linear(num_hidden, num_latent)
        self.log_sigma = Linear(num_hidden, num_latent)

    def forward(self, x, y):
        # concat location (x) and value (y)
        encoder_input = t.cat([x,y], dim=-1)

        # project vector with dimension 3 --> num_hidden
        encoder_input = self.input_projection(encoder_input)

        # self attention layer
        for attention in self.self_attentions:
            encoder_input, _ = attention(encoder_input, encoder_input, encoder_input)

        # mean
        hidden = encoder_input.mean(dim=1)
        hidden = t.relu(self.penultimate_layer(hidden))

        # get mu and sigma
        mu = self.mu(hidden)
        log_sigma = self.log_sigma(hidden)
        sigma = 0.1 + 0.9 * t.sigmoid(log_sigma)

        return t.distributions.normal.Normal(loc=mu, scale=sigma)


class DeterministicEncoder(nn.Module):
    """
    Deterministic Encoder [r]
    """
    def __init__(self, num_hidden, num_latent, n_CA=2, n_SA=2, input_dim=32):
        super().__init__()
        self.input_attentions = nn.ModuleList([Attention(num_hidden) for _ in range(2)])
        self.cross_attentions = nn.ModuleList([Attention(num_hidden) for _ in range(n_CA)])
        self.self_attentions= nn.ModuleList([Attention(num_hidden) for _ in range(n_SA)])
        self.input_projection = Linear(input_dim, num_hidden)
        self.context_projection = Linear(2048, num_hidden)
        self.target_projection = Linear(2048, num_hidden)

    def forward(self, context_x, context_y, target_x):
        # project vector with dimension n_bins(context_y) --> num_hidden
        encoder_input = self.input_projection(context_y)

        # self attention layer
        for attention in self.input_attentions:
            encoder_input, _ = attention(encoder_input, encoder_input, encoder_input)

        # query: target_x, key: context_x, value: representation
        query = self.target_projection(target_x)
        keys = self.context_projection(context_x)

        # cross attention layer
        for attention in self.cross_attentions:
            query, _ = attention(keys, encoder_input, query)

        # additional self attention layer
        for attention in self.self_attentions:
            query, _ = attention(query, query, query)

        return query


class Decoder(nn.Module):
    """
    Decoder for generation
    """
    def __init__(self, num_hidden, use_latent_path):
        super().__init__()

        self.target_projection = Linear(2048, num_hidden)

        if use_latent_path:
            self.linears = nn.ModuleList([Linear(num_hidden * 3, num_hidden * 3, w_init='relu') for _ in range(3)])
            self.final_projection = Linear(num_hidden * 3, 2)
        else:
            self.linears = nn.ModuleList([Linear(num_hidden * 2, num_hidden * 2, w_init='relu') for _ in range(3)])
            self.final_projection = Linear(num_hidden * 2, 2)
        self._use_latent_path = use_latent_path

    def forward(self, rep, target_x=None):

        target_x = self.target_projection(target_x)
        hidden = t.cat([rep, target_x], dim=-1)

        # mlp layers
        for linear in self.linears:
            hidden = t.relu(linear(hidden))
        hidden = self.final_projection(hidden)

        res = t.split(hidden, 1, dim=-1)
        mu = res[0]
        log_sigma = res[1]

        # Bound the variance
        sf = t.nn.Softplus()
        sigma = 0.1 + 0.9 * sf(log_sigma)

        # Get the distribution
        dist = self.MultivariateNormalDiag(
                loc=mu, scale_diag=sigma)

        return dist, mu, sigma

    def MultivariateNormalDiag(self, loc, scale_diag):
        if loc.dim() < 1:
            raise ValueError("loc must be at least one-dimensional.")
        return t.distributions.Independent(t.distributions.Normal(loc, scale_diag), 1)



class MultiheadAttention(nn.Module):
    """
    Multihead attention mechanism (dot attention)
    """
    def __init__(self, num_hidden_k):
        """
        :param num_hidden_k: dimension of hidden
        """
        super(MultiheadAttention, self).__init__()

        self.num_hidden_k = num_hidden_k
        self.attn_dropout = nn.Dropout(p=0.1)

    def forward(self, key, value, query):
        # Get attention score
        attn = t.bmm(query, key.transpose(1, 2))
        attn = attn / math.sqrt(self.num_hidden_k)

        attn = t.softmax(attn, dim=-1)

        # Dropout
        attn = self.attn_dropout(attn)

        # Get Context Vector
        result = t.bmm(attn, value)

        return result, attn


class Attention(nn.Module):
    """
    Attention Network
    """
    def __init__(self, num_hidden, h=4):
        """
        :param num_hidden: dimension of hidden
        :param h: num of heads
        """
        super(Attention, self).__init__()

        self.num_hidden = num_hidden
        self.num_hidden_per_attn = num_hidden // h
        self.h = h

        self.key = Linear(num_hidden, num_hidden, bias=False)
        self.value = Linear(num_hidden, num_hidden, bias=False)
        self.query = Linear(num_hidden, num_hidden, bias=False)

        self.multihead = MultiheadAttention(self.num_hidden_per_attn)

        self.residual_dropout = nn.Dropout(p=0.1)

        self.final_linear = Linear(num_hidden * 2, num_hidden)

        self.layer_norm = nn.LayerNorm(num_hidden)

    def forward(self, key, value, query):

        batch_size = key.size(0)
        seq_k = key.size(1)
        seq_q = query.size(1)
        residual = query

        # Make multihead
        key = self.key(key).view(batch_size, seq_k, self.h, self.num_hidden_per_attn)
        value = self.value(value).view(batch_size, seq_k, self.h, self.num_hidden_per_attn)
        query = self.query(query).view(batch_size, seq_q, self.h, self.num_hidden_per_attn)

        key = key.permute(2, 0, 1, 3).contiguous().view(-1, seq_k, self.num_hidden_per_attn)
        value = value.permute(2, 0, 1, 3).contiguous().view(-1, seq_k, self.num_hidden_per_attn)
        query = query.permute(2, 0, 1, 3).contiguous().view(-1, seq_q, self.num_hidden_per_attn)

        # Get context vector
        result, attns = self.multihead(key, value, query)

        # Concatenate all multihead context vector
        result = result.view(self.h, batch_size, seq_q, self.num_hidden_per_attn)
        result = result.permute(1, 2, 0, 3).contiguous().view(batch_size, seq_q, -1)

        # Concatenate context vector with input (most important)
        result = t.cat([residual, result], dim=-1)

        # Final linear
        result = self.final_linear(result)

        # Residual dropout & connection
        result = self.residual_dropout(result)
        result = result + residual

        # Layer normalization
        result = self.layer_norm(result)

        return result, attns

class ReLuloss(nn.Module):
    def __init__(self):
        super().__init__()
        self.threshold = 4

    def forward(self, truth, pred):
        loss = t.empty_like(truth)
        hp_idx = truth >= self.threshold
        mse_loss = (truth[hp_idx] - pred[hp_idx]).pow(2)
        relu_loss = self.reluloss(truth[~hp_idx], pred[~hp_idx])
        loss[hp_idx] = mse_loss
        loss[~hp_idx] = relu_loss
        return t.mean(loss)

    def reluloss(self, truth, pred):
        loss = t.zeros_like(truth)
        loss_idx = pred >= self.threshold
        loss[loss_idx] = (pred[loss_idx] - truth[loss_idx]).pow(2)
        return loss

