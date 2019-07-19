import math
from typing import Callable, Union, Dict, Tuple, Optional
import torch
from torch.utils import data

# import Dataset, DataLoader

class InpData(data.Dataset):
    """"""

    def __init__(self, X, Y):
        # self.X = torch.from_numpy(X).float()
        # self.Y = torch.from_numpy(Y).float()
        self.X = X
        self.Y = Y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

def random_classification(s, d, batch_size):
    inputs = torch.randn(s, d)
    labels = torch.randint(2, (s,))

    datasets = {'train': InpData(inputs, labels),
                'val': InpData(inputs, labels)}
    dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=batch_size, shuffle=True, num_workers=1,
                                                  drop_last=True)
                   for x in ['train', 'val']}
    return dataloaders

def latent_variable_receptive_field_encoding(
        num_trials: int,
        n_time: int,
        Inp_dim: int,
        tau: int = 1000,
        num_peaks: Optional[Tuple[int]] = None,
        peak_width_factors: Optional[Union[Tuple[Tuple[float]], float]] = 0.1,
        sigma_mult: float = 0.,
        sigma_add: float = 0.,
        latents: Tuple[Tuple[str]] = (('x',), ('y',), ('theta',)),
        frozen_epochs: bool = True
):
    """To have a smooth motion it the timescale and the variance of the field need to match. Roughly this means that
    tau^2~Inp_dim.

    Parameters
    ----------
    num_trials : int
    n_time : int
    Inp_dim : int
    tau : int
    num_peaks : Optional[Tuple[int]]
    peak_width_factors : Optional[Tuple[float]]
    sigma_mult : float
    sigma_add : float
    latents : Tuple[Tuple[str]]
    frozen_epochs : bool
        Whether or not the input set is the same across epochs. If this is false, input samples are like online learning
        -- you never get the same input sample twice, even across epochs.

    Returns
    -------

    """
    # loc = locals()
    # print(loc)
    num_latents = len(latents)

    # peak_width_factors = (.1, .1, .1)

    num_tensored = {len(x) for x in latents}
    if max(num_tensored) > 3:
        raise ValueError("Tensoring more than 3 variables not currently supported.")

    def circular_distances(i, j, circular=False):
        cdist = torch.cdist
        j = j.reshape(-1, 1)
        i = i.reshape(-1, 1)
        dists = cdist(j, i)
        if circular:
            dists = torch.min(torch.fmod(dists, 1), torch.fmod(1 - dists, 1))
        return dists

    responses = []
    latent_vals = []
    for i0, latent_group in enumerate(latents):
        latent_vals_group = []
        dists = torch.zeros(num_trials * n_time, Inp_dim, len(latent_group))
        for i1, x in enumerate(latent_group):
            if x == 'theta':
                circular = True
            elif x == 'x':
                circular = False
            else:
                raise AttributeError("Latent variable not recognized.")
            if hasattr(peak_width_factors, '__len__'):
                sigma = peak_width_factors[i0][i1]
            else:
                sigma = peak_width_factors
            latent_val = torch.zeros((num_trials, n_time))
            latent_val[:, 0] = torch.rand(num_trials)

            for i_time in range(n_time):
                if tau == 0:
                    latent_val[:, i_time] = torch.rand(num_trials)
                else:
                    temp = (1 / math.sqrt(tau)) * torch.randn(num_trials)
                    latent_val[:, i_time] = latent_val[:, i_time - 1] + temp
                    # theta[:, i_time, :] = theta[:, i_time - 1, :] + 1 / np.sqrt(tau) * np.random.randn(num_trials, 1)
            latent_vals_group.append(latent_val)
            dx = 1 / Inp_dim
            receptive_field_centers = torch.linspace(0, 1 - dx, Inp_dim)
            dist = circular_distances(receptive_field_centers, latent_val, circular)
            dists[:, :, i1] = dist
        latent_vals.append(latent_vals_group)
        dist_final = torch.sum(dists ** 2, dim=-1).reshape((num_trials, n_time, Inp_dim))
        responses.append(0.1 * torch.exp(-(dist_final / sigma) ** 2))

    responses = torch.cat(responses, dim=-1)
    # torch.stack(latent_vals).shape
    # from matplotlib import pyplot as plt
    # plt.figure()
    # plt.scatter(latent_vals[0][0][0], latent_vals[0][1][0], c=responses[0, :, 0]);
    # plt.show()

    if sigma_add != 0:
        responses = responses + sigma_add * torch.randn(*responses.shape)

    return responses, responses, latent_vals

if __name__ == '__main__':
    X = latent_variable_receptive_field_encoding(4, 100, 20, tau=0, sigma_mult=0,
                                                 sigma_add=0.00, latents=(('theta', 'x'), ('x',)))[0]
