import torch
from torch import nn


def get_x0_from_noise(sample, model_output, alphas_cumprod, timestep):
    """Convert epsilon prediction back to x0 estimate (DDIM closed form).
    Used by the 1-step diffusion decoder to obtain the reconstructed latent.
    """
    alpha_prod_t = alphas_cumprod[timestep].reshape(-1, 1, 1, 1)
    beta_prod_t = 1 - alpha_prod_t

    pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
    return pred_original_sample


class NoOpContext:
    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass


class DummyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = nn.Identity()

    def forward(self, x):
        return self.dummy(x)
