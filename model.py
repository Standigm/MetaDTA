"""
The Gridless Docking model for supervised learning.
"""
from argparse import Namespace
from typing import List, Dict, Callable

import torch as t
import torch.nn as nn

from module import LatentEncoder, DeterministicEncoder, Decoder


class LatentBinModel(nn.Module):
    """
    Latent Model (Attentive Neural Process)
    """
    def __init__(self, input_dim, n_bins, num_hidden, n_CA, n_SA, use_latent_path):
        super().__init__()
        self._use_latent_path = use_latent_path
        if use_latent_path:
            self.latent_encoder = LatentEncoder(num_hidden, num_hidden, input_dim+n_bins)
        self.deterministic_encoder = DeterministicEncoder(num_hidden, num_hidden, n_CA, n_SA)
        self.decoder = Decoder(num_hidden, use_latent_path)

    def forward(self, context_x, context_y, target_x, target_y=None, target_y_f=None):
        num_targets = target_x.size(1)

        deterministic_rep = self.deterministic_encoder(context_x, context_y, target_x)

        if self._use_latent_path:
            prior = self.latent_encoder(context_x, context_y)

            # For training
            if target_y is not None:
                posterior = self.latent_encoder(target_x, target_y)

            # For Generation
            else:
                posterior = prior

            latent_rep = posterior.sample()
            latent_rep = t.unsqueeze(latent_rep, dim=1).repeat(1, num_targets, 1)

            rep = t.cat([deterministic_rep, latent_rep], axis=-1)

        else:
            rep = deterministic_rep
        dist, y_pred, sigma = self.decoder(rep, target_x)

        # For Training
        if target_y is not None:
            log_p = dist.log_prob(target_y_f)

            if self._use_latent_path:
                # get KL divergence between prior and posterior
                kl = t.sum(t.distributions.kl.kl_divergence(posterior, prior), dim=-1, keepdim=True)
                kl = kl.repeat(1, num_targets)

                # maximize prob and minimize KL divergence
                loss = -t.mean(log_p - kl / num_targets)

            else:
                loss = -t.mean(log_p)
                kl = None

        # For Generation
        else:
            kl = None
            loss = None

        return y_pred, sigma, kl, loss


    def kl_div(self, prior_mu, prior_var, posterior_mu, posterior_var):
        kl_div = (t.exp(posterior_var) + (posterior_mu-prior_mu) ** 2) / t.exp(prior_var) - 1. + (prior_var - posterior_var)
        kl_div = 0.5 * kl_div.sum()
        return kl_div


