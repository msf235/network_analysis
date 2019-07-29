import math
from typing import Callable, Union, Dict, Tuple, Optional
import torch
import torch.utils.data

# import Dataset, DataLoader

class InpData(torch.utils.data.Dataset):
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

def data(num_trials: int, n_time: int, num_receptive_fields: Union[int, Tuple[Tuple[int]]], tau: int = 1000,
         latents: Tuple[Tuple[str]] = (('x',), ('y',), ('theta',)), num_peaks: Optional[Tuple[int]] = None,
         peak_width_factors: Optional[Union[Tuple[Tuple[float]], float]] = 0.1, sigma_mult: float = 0.,
         sigma_add: float = 0.):
    """To have a smooth motion it the timescale and the variance of the field need to match. Roughly this means that
    tau^2~Inp_dim.

    Parameters
    ----------
    num_trials : int
    n_time : int
    num_receptive_fields : Union[int, Tuple[Tuple[int]]]
    tau : int
    latents : Tuple[Tuple[str]]
    num_peaks : Optional[Tuple[int]]
    peak_width_factors : Optional[Tuple[float]]
    sigma_mult : float
    sigma_add : float
    frozen_epochs : bool
        Whether or not the input set is the same across epochs. If this is false, input samples are like online learning
        -- you never get the same input sample twice, even across epochs.

    Returns
    -------

    """
    if not hasattr(num_receptive_fields, '__len__'):
        d = num_receptive_fields
        num_receptive_fields = []
        for i0 in range(len(latents)):
            num_receptive_fields.append([])
            for i1 in range(len(latents[i0])):
                num_receptive_fields[i0].append(d)

    def circular_distances(i, j, circular=False):
        cdist = torch.cdist
        j = j.reshape(-1, 1)
        i = i.reshape(-1, 1)
        dists = cdist(j, i)
        if circular:
            dists = torch.min(torch.remainder(dists, 1), torch.remainder(1 - dists, 1))
        return dists

    responses = []
    latent_vals = []
    for i0, latent_group in enumerate(latents):
        latent_vals_group = []
        len_latents = len(latent_group)
        p = 1
        for x in num_receptive_fields[i0]:
            p = p * x
        dists = torch.zeros(num_trials * n_time, p, len_latents)
        temp = [torch.linspace(0, 1 - 1 / num_receptive_fields[i0][i1], num_receptive_fields[i0][i1]) for i1 in
                range(len_latents)]
        out = torch.meshgrid(temp)
        receptive_field_centers = torch.stack(out, dim=-1).reshape(-1, len_latents)
        for i1, x in enumerate(latent_group):
            if x == 'theta':
                circular = True
                tau_eff = tau
            elif x == 'x':
                circular = False
                # tau_eff = 2*tau
                tau_eff = tau
            else:
                raise AttributeError("Latent variable not recognized.")
            if hasattr(peak_width_factors, '__len__'):
                sigma = peak_width_factors[i0][i1]
            else:
                sigma = peak_width_factors
            latent_val = torch.zeros((num_trials, n_time))
            latent_val[:, 0] = torch.rand(num_trials)
            print
            for i_time in range(1, n_time):
                if tau == 0:
                    latent_val[:, i_time] = torch.rand(num_trials)
                else:
                    temp = (1 / math.sqrt(tau_eff)) * torch.randn(num_trials)
                    latent_val[:, i_time] = latent_val[:, i_time - 1] + temp
                    # theta[:, i_time, :] = theta[:, i_time - 1, :] + 1 / np.sqrt(tau) * np.random.randn(num_trials, 1)
            # a = latent_val.clone()
            if not circular:
                latent_val = torch.abs(torch.remainder(2 * latent_val, 2) - 1)
            else:
                latent_val = torch.remainder(latent_val, 1)
            latent_vals_group.append(latent_val)
            # dx = 1 / Inp_dim
            # receptive_field_centers = torch.linspace(0, 1 - dx, Inp_dim)
            dist_x = circular_distances(receptive_field_centers[:, i1].unsqueeze(dim=-1), latent_val, circular)
            dists[:, :, i1] = dist_x
        latent_vals.append(latent_vals_group)
        dist_sq_final = torch.sum(dists ** 2, dim=-1).reshape((num_trials, n_time, p))
        responses.append(0.1 * torch.exp(-dist_sq_final / sigma ** 2))

    responses = torch.cat(responses, dim=-1)
    # from matplotlib import pyplot as plt
    # plt.figure()
    # idx = 200
    # plt.scatter(latent_vals[0][0][0], latent_vals[0][1][0], c=responses[0, :, idx]);
    # plt.show()

    if sigma_add != 0:
        responses = responses + sigma_add * torch.randn(*responses.shape)

    return responses, responses, latent_vals

def dataset(num_trials: int, n_time: int, Inp_dim: int, tau: int = 1000,
            latents: Tuple[Tuple[str]] = (('x',), ('y',), ('theta',)), num_peaks: Optional[Tuple[int]] = None,
            peak_width_factors: Optional[Union[Tuple[Tuple[float]], float]] = 0.1, sigma_mult: float = 0.,
            sigma_add: float = 0., freeze_epochs: bool = True, train_perc: float = 0.8, return_latents: bool = False):
    if not freeze_epochs:
        raise AttributeError("freeze_epochs=False option isn't implemented yet")
    else:
        # class LatentVariableReceptiveFieldEncoding(torch.utils.data.Dataset):
        #     def __init__(self):
        #         pass

        inputs, targets, latent_vals = data(num_trials, n_time, Inp_dim, tau, latents, num_peaks, peak_width_factors,
                                            sigma_mult, sigma_add)

        train_time = int(round(train_perc * n_time))

        train_inputs = inputs[:, :train_time]
        train_inputs = train_inputs.reshape(-1, train_inputs.shape[-1])
        val_inputs = inputs[:, train_time:]
        val_inputs = val_inputs.reshape(-1, val_inputs.shape[-1])

        train_targets = targets[:, :train_time]
        train_targets = train_targets.reshape(-1, train_targets.shape[-1])
        val_targets = targets[:, train_time:]
        val_targets = val_targets.reshape(-1, val_targets.shape[-1])

        train_dataset = InpData(train_inputs, train_targets)
        val_dataset = InpData(val_inputs, val_targets)

        train_latent_vals = []
        val_latent_vals = []
        for group in latent_vals:
            train_latent_vals.append([])
            val_latent_vals.append([])
            for x in group:
                train_latent_vals[-1].append(x[:, :train_time].reshape(-1))
                val_latent_vals[-1].append(x[:, train_time:].reshape(-1))

        if return_latents:
            return train_dataset, val_dataset, train_latent_vals, val_latent_vals
        else:
            return train_dataset, val_dataset

if __name__ == '__main__':
    # out = data(4, 100, 20, tau=0, sigma_mult=0,
    #            sigma_add=0.00, latents=(('theta', 'x'), ('x',)))[0]

    out = dataset(4, 120, 20, tau=0, latents=(('theta', 'x'), ('x',)), sigma_mult=0, sigma_add=0.00)

    print
