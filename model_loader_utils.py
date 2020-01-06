import torch
import numpy
from pathlib import Path
# from . import model_output_manager as mom
import model_output_manager as mom


import traceback
import warnings
import sys

def warn_with_traceback(message, category, filename, lineno, file=None, line=None):

    log = file if hasattr(file,'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))

warnings.showwarning = warn_with_traceback

def load_model(model_template, filename, optimizer=None, learning_scheduler=None):
    """
    Load the given torch model from the given file. WARNING: the model_template passed to this function is overwritten.

    Args:
        model_template : torch.nn.Module
            The neural network module which should have its state loaded. WARNING: This model will be overwritten.
        filename: string
            the file that contains the state of the model

    Returns:
        torch.nn.Module
            the loaded model
    """
    model_state_info = torch.load(filename)
    model_template.load_state_dict(model_state_info['model_state_dict'])
    if optimizer is not None and learning_scheduler is not None:
        optimizer.load_state_dict(model_state_info['optimizer_state_dict'])
        learning_scheduler.load_state_dict(model_state_info['learning_scheduler_state_dict'])
        return model_template, optimizer, learning_scheduler
    if learning_scheduler is not None:
        learning_scheduler.load_state_dict(model_state_info['learning_scheduler_state_dict'])
        return model_template, learning_scheduler
    if optimizer is not None:
        optimizer.load_state_dict(model_state_info['optimizer_state_dict'])
        return model_template, optimizer
    return model_template

def get_max_epoch(out_dir, max_look = 10000):
    """
    Get the oldest saved epoch in the given directory.

    Args:
        out_dir: (string) the path to the directory
        max_look: (int) Maximum number of epochs to iterate through before giving up.

    Returns:
        (number) the oldest epoch that is saved in the specified directory.
    """
    out_dir = Path(out_dir)
    # i0 = 0
    # dir_exists = True
    for i0 in range(max_look):
        filename = out_dir/'check_{}'.format(i0)
        dir_exists = Path.exists(filename)
        if not dir_exists:
            return i0 - 1
    raise ValueError("max_look not large enough to iterate through all epochs. Increase size of max_look.")

def load_model_from_epoch_and_dir(model, out_dir, epoch_num, optimizer=None, learning_scheduler=None):
    """
    Loads the module as it was on a specific epoch.

    Args:
        model: (torch.nn.Module) the neural network model which should have its state loaded
        out_dir: (string) the path to the directory the model was saved in
        epoch_num: (number) the epoch to load, or -1 to load the max epoch.

    Returns:
        (torch.nn.Module, dict) the model and the loaded state information
    """
    out_dir = Path(out_dir)
    if epoch_num == -1:
        epoch_num = get_max_epoch(out_dir)
    filename = out_dir/'check_{}'.format(epoch_num)
    return load_model(model, filename, optimizer, learning_scheduler)

def load_model_mom(model, epoch, arg_dict, table_path, compare_exclude=[], optimizer=None, learning_scheduler=None):
    """
    Load a specific model using the model output manager paradigm. This uses
    the model output manager to locate/create the folder containing the run,
    and then acts equivalently to load_model_from_epoch_and_dir.

    Args:
        model: (torch.nn.Module) the neural network model which should have its state loaded
        epoch: (number) the epoch to load or -1 for the latest epoch
        arg_dict:
            (dict) the row which should be saved to file. The file will be generated if
            this is the first row.
        table_path:
            (string) the path to where the file containing the table that the row should
            be added to
        run_name: (string) the name of the run

    Returns:
        (torch.nn.Module, dict) the loaded model
    """
    run_dir = mom.get_dirs_and_ids_for_run(arg_dict, table_path, compare_exclude)[0]
    if epoch == -1:
        epoch = get_max_epoch(run_dir)
    # Todo: use more sophisticated way of choosing the best directory
    return load_model_from_epoch_and_dir(model, run_dir[-1], epoch, optimizer, learning_scheduler)

class model_loader:
    def __init__(self, model_template, run_dir):
        self.run_dir = format_dir(run_dir)
        self.num_epochs = get_max_epoch(self.run_dir)
        self.model_template = model_template

    def __len__(self):
        return self.num_epochs + 1

    def __getitem__(self, idx):
        n_idx = np.arange(self.num_epochs + 1)[idx]
        if not hasattr(n_idx, '__len__'):
            n_idx = [n_idx]
        models = []
        for el0 in n_idx:
            models.append(load_model_from_epoch_and_dir(self.model_template, self.run_dir, el0)[0])

        return models

class weight_loader:
    def __init__(self, run_dir):
        self.run_dir = format_dir(run_dir)
        self.num_epochs = get_max_epoch(self.run_dir)

    def __len__(self):
        return self.num_epochs + 1

    def __getitem__(self, idx):
        w_idx = np.arange(self.num_epochs + 1)[idx]
        w_scal = False
        if not hasattr(w_idx, '__len__'):
            w_idx = [w_idx]
            w_scal = True
        filename = self.run_dir + 'check_{}'.format(w_idx[0])
        state_info = torch.load(filename)
        state_dict = state_info['model_state_dict']
        keys = list(state_dict.keys())
        w = dict()
        for key in keys:
            if w_scal:
                w[key] = np.zeros(state_dict[key].shape)
            else:
                w[key] = np.zeros((len(w_idx),) + state_dict[key].shape)

        for i0, el0 in enumerate(w_idx):
            filename = self.run_dir + 'check_{}'.format(el0)
            state_info = torch.load(filename)
            state_dict = state_info['model_state_dict']
            for key in keys:
                if w_scal:
                    w[key] = state_dict[key]
                else:
                    w[key][i0] = state_dict[key]

        return w


def get_activity(model, run_dir, inputs, epoch_idx, layer_idx=None, activations='post', return_as_Tensor=False):
    """

    Parameters
    ----------
    model
    run_dir
    inputs
    epoch_idx : Union[int, list, tuple, slice]
    activations : str
        Type of activation. Options are: 'post' for getting the activations after the nonlinearity is applied,
        'pre' for getting the activations  before the nonlinearity, and 'both' for getting the activations both
        before and after the nonlinearity is applied. In the case of 'both', it is assumed that the model method
        get_activations returns a list with preactivation for a layer, postactivation for the layer, preactivation
        for the next layer, and so forth, where pre and post activations always come in pairs.
    return_as_Tensor : bool
        If True, the return data will be attempted to be returned as a pytorch Tensor. This only works if the
        number of units in each of the layers specified by layer_idx are the same.

    Returns
    -------

    """
    if layer_idx is None:
        layer_idx = slice(None)
    def identity(x):
        return x
    # Here we define layer_stack_fun, which will either stack things or not depending on function arguments.
    if return_as_Tensor:
        layer_stack_fun = torch.stack
    else:
        layer_stack_fun = identity
    # We need to take care of three cases (1) layer_idx is an int, (2) layer_idx is a list or tuple, (3) layer_idx is a
    # slice.
    if isinstance(layer_idx, int): # (1)
        if activations == 'both': # In this case we are still indexing multiple layers, so we index as in case (2).
            def index_fun(data, idx):
                return layer_stack_fun([data[x] for x in idx])
        else: # In this we just need to grab the element specified by idx
            def index_fun(data, idx):
                return data[idx]
    elif hasattr(layer_idx, '__len__'): # (2)
        def index_fun(data, idx):
            return layer_stack_fun([data[x] for x in idx])
    elif isinstance(layer_idx, slice): # (3)
        def index_fun(data, idx):
            return layer_stack_fun(data[idx])
    else:
        raise AttributeError("layer_idx option not recognized.")

    num_epochs = get_max_epoch(run_dir)
    # Need to take care of three cases (1) epoch_idx is an int, (2) epoch_idx is a list or tuple, (3) epoch_idx is a
    # slice.
    epochs = range(num_epochs)
    scal = isinstance(epoch_idx, int)
    if hasattr(epoch_idx, '__len__'):
        epoch_idx = [epochs[k] for k in epoch_idx]
    else:
        epoch_idx = list(epochs)[epoch_idx]
    if scal:
        epoch_idx = [epoch_idx]

    acts = []
    if activations == 'post':
        for idx in epoch_idx:
            model = load_model_from_epoch_and_dir(model, run_dir, idx)
            act = index_fun(model.get_post_activations(inputs), layer_idx)
            acts.append(act)
    elif activations == 'pre':
        for idx in epoch_idx:
            model = load_model_from_epoch_and_dir(model, run_dir, idx)
            act = index_fun(model.get_pre_activations(inputs), layer_idx)
            acts.append(act)
    elif activations == 'both':
        for idx in epoch_idx:
            model = load_model_from_epoch_and_dir(model, run_dir, idx)
            act = model.get_activations(inputs)
            num_layers = len(act) / 2
            layer_idx_stretched = 2*torch.arange(num_layers)[layer_idx]
            layer_idx_adjusted = torch.sort(torch.cat((layer_idx_stretched, layer_idx_stretched+1)))[0]
            act = index_fun(act, layer_idx_adjusted)
            acts.append(act)
    if scal:
        return acts[0]
    if return_as_Tensor:
        return torch.stack(acts)
    return acts

    # all_acts = torch.stack(all_acts)

# class activity_loader:
#     def __init__(self, model, inputs, run_dir, layer_idx=None):
#         """
#
#         Args:
#             model ():
#             run_dir ():
#             X ():
#             layer_idx ():
#         """
#         self.run_dir = format_dir(run_dir)
#         self.num_epochs = get_max_epoch(self.run_dir)
#         self.model = model
#         self.layer_idx = layer_idx
#         self.inputs = inputs
#
#     def __len__(self):
#         return self.num_epochs + 1
#
#     def __getitem__(self, epoch_idx):
#         """
#         Load over range of epochs.
#         Args:
#             idx (int, slice): Load over range of epochs designated by idx.
#
#         Returns:
#
#         """
#         scal = False
#         if not hasattr(epoch_idx, '__len__'):
#             epoch_idx = [epoch_idx]
#             scal = True
#         epoch_idx = torch.arange(self.num_epochs + 1)[epoch_idx]  # This allows for negative indices
#         acts = []
#         if activations == 'post':
#             for idx in epoch_idx:
#                 model = load_model_from_epoch_and_dir(self.model, self.run_dir, idx)
#                 acts.append(model.get_post_activations(self.inputs))
#         elif activations == 'pre':
#             for idx in epoch_idx:
#                 model = load_model_from_epoch_and_dir(self.model, self.run_dir, idx)
#                 acts.append(model.get_pre_activations(self.inputs))
#         if activations == 'both':
#             for idx in epoch_idx:
#                 model = load_model_from_epoch_and_dir(self.model, self.run_dir, idx)
#                 acts.append(model.get_activations(self.inputs))
#         if scal:
#             acts = acts[0]
#         return acts


# class activity_loader:
#     def __init__(self, model, run_dir, X, layer_idx=None):
#         """
#
#         Args:
#             model ():
#             num_epochs (int): Number of epochs, not including the 0 epoch. Soo if num_epochs=2 then the epochs are
#                 [0,1,2].
#             run_dir ():
#             X ():
#             layer_idx ():
#         """
#         self.run_dir = format_dir(run_dir)
#         self.num_epochs = get_max_epoch(self.run_dir)
#         self.model = model
#         self.X = X
#         self.layer_idx = layer_idx
#
#     def __len__(self):
#         return self.num_epochs + 1
#
#     def __getitem__(self, idx):
#         """
#         Load over range of epochs.
#         Args:
#             idx (int, slice): Load over range of epochs designated by idx.
#
#         Returns:
#
#         """
#         if isinstance(idx, tuple):
#             idx = list(idx)
#         n_idx = np.arange(self.num_epochs + 1)[idx]
#         scal = False
#         if not hasattr(n_idx, '__len__'):
#             n_idx = [n_idx]
#             scal = True
#         model, state_info = load_model_from_epoch_and_dir(self.model, self.run_dir, n_idx[0])
#         acts = model.get_activations(self.X)
#         # acts_full = acts  # careful!
#         acts_full = []
#         # hid_sh = [0] * len(acts_full)
#         for i0, act in enumerate(acts):
#             if scal:
#                 acts_full.append(act.numpy().astype(float))
#             else:
#                 acts_full.append(np.zeros((len(n_idx),) + act.shape))
#                 acts_full[i0][0] = act.numpy().astype(float)
#
#         if scal:
#             if self.layer_idx is not None:
#                 acts_full = acts_full[self.layer_idx]
#             return acts_full
#
#         for i0, el0 in enumerate(n_idx[1:]):
#             model, state_info = load_model_from_epoch_and_dir(self.model, self.run_dir, el0)
#             acts = model.get_activations(self.X)
#             for i1, act in enumerate(acts):
#                 acts_full[i1][i0 + 1] = act.numpy().astype(float)
#         if self.layer_idx is not None:
#             acts_full = acts_full[self.layer_idx]
#         return acts_full