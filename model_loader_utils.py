import torch
import numpy
from pathlib import Path
from . import model_output_manager as mom


import traceback
import warnings
import sys
from typing import *

def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    log = file if hasattr(file, 'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))

warnings.showwarning = warn_with_traceback

def get_epoch_and_save_from_savestr(savestr):
    savestr = str(savestr)
    parts = savestr.split('_')
    epoch = int(parts[1])
    save = int(parts[3].split('.')[0])
    return epoch, save

def load_model(model, filename, optimizer=None, learning_scheduler=None):
    """
    Load the given torch model from the given file. Loads the model by reference, so the passed in model is modified.
    An optimizer and learning_scheduler object can also be loaded if they were saved along with the model.
    This can, for instance, allow training to resume with the same optimizer and learning_scheduler state.

    Parameters
    ----------
        model : torch.nn.Module
            The pytorch module which should have its state loaded. WARNING: The parameters of this model will be
            modified.
        filename: Union[str, Path]
            The file that contains the state of the model
        optimizer: torch.optim.Optimizer
            The pytorch optimizer object which should have its state loaded. WARNING: The state of this object will be
            modified.
        optimizer: Union[torch.optim._LRScheduler, object]
            The pytorch learning rate scheduler object which should have its state loaded. WARNING: The state of this
            object will be
            modified.

    Returns
    ----------
        Optional[int]
        Model is loaded by reference, so nothing is returned if loading is successful. If the file to load isn't found,
        returns -1.
    """
    try:
        model_state_info = torch.load(filename)
    except FileNotFoundError:
        return -1
    model.load_state_dict(model_state_info['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(model_state_info['optimizer_state_dict'])
    if learning_scheduler is not None:
        learning_scheduler.load_state_dict(model_state_info['learning_scheduler_state_dict'])

def get_epochs_and_saves(out_dir):
    """
    Get a list of all the epochs and saves in the current directory.

    Parameters
    ----------
    out_dir: string
        the path to the directory

    Returns
    -------
    list

    """
    out_dir = Path(out_dir)
    ps = Path(out_dir).glob('epoch_*_save_*.pt')
    epochs = sorted(list({int(str(p.parts[-1]).split('_')[1]) for p in ps}))
    saves = []
    for epoch in epochs:
        ps = Path(out_dir).glob(f'epoch_{epoch}_save_*.pt')
        saves.append(sorted([int(str(p.parts[-1]).split('_')[-1].split('.')[0]) for p in ps]))
    return epochs, saves

def get_max_epoch_and_save(out_dir):
    """
    Get the maximum epoch and save in save directory

    Parameters
    ----------
    out_dir: string
        the path to the directory

    Returns
    -------
    int
        the maximum checknum that is saved in the specified directory.
    """
    out_dir = Path(out_dir)
    ps = list(Path(out_dir).glob('epoch_*_save_*.pt'))
    # import ipdb; ipdb.set_trace()
    epochs = [get_epoch_and_save_from_savestr(str(p.parts[-1]))[0] for p in ps]
    saves = [get_epoch_and_save_from_savestr(str(p.parts[-1]))[1] for p in ps]
    max_epoch_id = torch.argmax(torch.tensor(epochs)).item()
    max_epoch = epochs[max_epoch_id]
    saves_for_max_epoch = [saves[k] for k in range(len(saves)) if epochs[k] == max_epoch]
    max_save = max(saves_for_max_epoch)
    return max_epoch, max_save

def get_max_epoch(out_dir, require_save_zero=True):
    """
    Get the maximum epoch in save directory

    Parameters
    ----------
    out_dir: string
        the path to the directory

    Returns
    -------
    int
        the maximum checknum that is saved in the specified directory.
    """
    out_dir = Path(out_dir)
    if require_save_zero:
        ps = Path(out_dir).glob('epoch_*_save_0.pt')
    else:
        ps = Path(out_dir).glob('epoch_*_save_*.pt')
    epochs = [get_epoch_and_save_from_savestr(str(p.parts[-1]))[0] for p in ps]
    if len(epochs) > 0:
        return max(epochs)
    else:
        return False

def load_model_from_epoch_and_dir(model, out_dir, epoch_num, save_num=0, optimizer=None, learning_scheduler=None):
    """
    Loads the model as it was on a specific epoch. Loads the model by reference, so the passed in model is modified.

    Parameters
    ----------
    model: torch.nn.Module
        Instantiation of the model which should have its state loaded
    out_dir: Union[str,Path]
        The path to the directory the model's states over training were saved in
    epoch_num: int
        the epoch to load, or -1 to load the max epoch.

    Returns
    -------
    Optional[int]
        Model is loaded by reference, so nothing is returned if loading is successful. If the file to load isn't found,
        returns -1.

    """
    out_dir = Path(out_dir)
    if epoch_num == -1:
        epoch_num = get_max_epoch(out_dir)
    filename = out_dir/f'epoch_{epoch_num}_save_{save_num}.pt'
    return load_model(model, filename, optimizer, learning_scheduler)
    # return load_model(model, filename, optimizer, learning_scheduler)

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

    Returns
    -------
    None
    """
    run_dir = mom.get_dirs_and_ids_for_run(arg_dict, table_path, compare_exclude)[0]
    if epoch == -1:
        epoch = get_max_epoch(run_dir)
    # Todo: use more sophisticated way of choosing the best directory
    return load_model_from_epoch_and_dir(model, run_dir[-1], epoch, 0, optimizer, learning_scheduler)

def smart_load(arg_dict, table_path, stop_num, model, optimizer, learning_scheduler, compare_exclude,
               require_complete=True):
    """
    Check to see if the same model has been trained before. If so, check to see if the number of epochs previously
    trained is at least as much as needed. Cases are as follows.

    1. The same model has been trained before and is found on disk
        A. The previously trained model has an output checkpoint that matches the requested stop_epoch.
           Load the model, optimizer, and learning_scheduler corresponding to stop_num.
           Return  load_dir, stop_check_num
        B. The number of check_numbers previously used to train is not as much as the current requested stop_num.
            Load the model, optimizer, and learning_scheduler corresponding to the highest checkpoint number available (
            let's call this "max_check_num").
            Return  load_dir, max_check_num
    2. The model has not been trained before.
        Return None, False

    CAUTION: Loading is done by reference, so the passed model, optimizer, and learning_scheduler are modified.

    Returns
    -------
    See above

    """

    run_dirs, ids, output_exists = mom.get_dirs_and_ids_for_run(arg_dict, table_path, compare_exclude)
    if require_complete:
        run_dirs = [x for x in run_dirs if Path.exists(x/'training_completed_token')]

    if len(run_dirs) == 0:
        return None, None
    max_check_num_less_than_stop_num = 0
    arg_max_check_num_less_than_stop_num = 0
    for i0, x in enumerate(run_dirs):
        check_nums = get_check_nums(x)
        max_less_than_i0 = 0
        for cn in check_nums:
            if cn <= stop_num:
                max_less_than_i0 = max(max_less_than_i0, cn)  # Distance from stop_num to the closest check_num
        if max_less_than_i0 >= max_check_num_less_than_stop_num:
            max_check_num_less_than_stop_num = max_less_than_i0
            arg_max_check_num_less_than_stop_num = i0

    run_dir = run_dirs[arg_max_check_num_less_than_stop_num]
    check_num = max_check_num_less_than_stop_num
    exists = load_model_from_epoch_and_dir(model, run_dir, check_num, optimizer,
                                           learning_scheduler)
    if exists == -1:
        return None, None

    return run_dir, check_num

class model_loader:
    def __init__(self, model_template, run_dir):
        # self.run_dir = format_dir(run_dir)
        self.num_epochs = get_max_epoch(self.run_dir)
        self.model_template = model_template

    def __len__(self):
        return self.num_epochs+1

    def __getitem__(self, idx):
        n_idx = torch.arange(self.num_epochs+1)[idx]
        if not hasattr(n_idx, '__len__'):
            n_idx = [n_idx]
        models = []
        for el0 in n_idx:
            models.append(load_model_from_epoch_and_dir(self.model_template, self.run_dir, el0)[0])

        return models

class weight_loader:  # Todo: get it working for multiple saves per epoch
    def __init__(self, run_dir):
        # self.run_dir = format_dir(run_dir)
        self.run_dir = Path(run_dir)
        self.num_epochs = get_max_epoch(self.run_dir)

    def __len__(self):
        return self.num_epochs+1

    def __getitem__(self, idx):
        w_idx = torch.arange(self.num_epochs+1)[idx]
        w_scal = False
        # if not hasattr(w_idx, '__len__'):
        try:
            len(w_idx)
        except TypeError:
            w_idx = [w_idx]
            w_scal = True
        filename = self.run_dir/'epoch_{}_save_0.pt'.format(w_idx[0])
        state_info = torch.load(filename)
        state_dict = state_info['model_state_dict']
        keys = list(state_dict.keys())
        w = dict()
        for key in keys:
            if w_scal:
                w[key] = torch.zeros(state_dict[key].shape)
            else:
                w[key] = torch.zeros((len(w_idx),)+state_dict[key].shape)

        for i0, el0 in enumerate(w_idx):
            filename = self.run_dir/'epoch_{}_save_0.pt'.format(el0)
            state_info = torch.load(filename)
            state_dict = state_info['model_state_dict']
            for key in keys:
                if w_scal:
                    w[key] = state_dict[key]
                else:
                    w[key][i0] = state_dict[key]

        return w

def get_activity(model, run_dir, inputs, layer_idx, epoch_idx, save_idx=0, return_as_Tensor=False, detach=True,
                 eval=True):
    """
    Gets the hidden unit activations of model in response to inputs at the savepoints specified by save_idx.

    Parameters
    ----------
    model : torch.nn.Module
        The model from which to get the hidden unit activations
    run_dir : Union[str, Path]
        Directory where the states of the network through training are saved
    inputs : torch.Tensor
        The inputs to give to the model
    layer_idx : Union[int, list, tuple, slice]
        Indices for the layers
    epoch_idx : Union[int, list, tuple, slice]
        Indices for the saves
    return_as_Tensor : bool
        If True, the return data will be attempted to be returned as a pytorch Tensor. This only works if the
        number of units in each of the layers specified by layer_idx are the same.
    detach : bool
        If True, then the return data will be detached from the torch compute graph. This should be True unless
        gradients are needed.
    eval : bool
        If True, set model to eval mode (model.eval() is called). Make sure model is set to train explicitly along with
        setting eval=False if that is desired, since setting eval=False doesn't change the model mode.

    Returns
    -------
    Union[List[List[torch.Tensor]], torch.Tensor]
        If return_as_Tensor is False, then the returned data structure has the form List[List[torch.Tensor]]. The
        first list index corresponds to the saves, the second list index corresponds to the layer,
        and the torch.Tensor object is the response of corresponding layer at the corresponding save (its size is
        determined by the network model).


    """
    if eval:
        model.eval()  # Needed to make sure batch normalization and dropout aren't activated
    if layer_idx is None:
        layer_idx = slice(None)

    def identity(x):
        return x

    # Here we define layer_stack_fun, which will either stack things or not depending on return_as_Tensor.
    if return_as_Tensor:
        layer_stack_fun = torch.stack
    else:
        layer_stack_fun = identity
    # We need to take care of three cases (1) layer_idx is an int, (2) layer_idx is a list or tuple, (3) layer_idx is a
    # slice object. Based on this, we define a handler function index_fun.
    if isinstance(layer_idx, int):  # (1)
        def index_fun(data, idx):
            return data[idx]
    elif hasattr(layer_idx, '__len__'):  # (2)
        def index_fun(data, idx):
            return layer_stack_fun([data[x] for x in idx])
    elif isinstance(layer_idx, slice):  # (3)
        def index_fun(data, idx):
            return layer_stack_fun(data[idx])
    else:
        raise AttributeError("layer_idx does not have a valid type.")

    num_epochs, num_saves = get_max_epoch_and_save(run_dir)

    # Now we need to take care of these three cases for epoch_idx. However, this time the case is slightly different
    # as we will need to find the maximum epoch on disk, and then loop over the appropriate epoch values.
    epochs = range(num_epochs+1)
    scal_epochs = isinstance(epoch_idx, int)
    if hasattr(epoch_idx, '__len__'):
        epoch_idx = [epochs[k] for k in epoch_idx]
    else:
        epoch_idx = list(epochs)[epoch_idx]
    if scal_epochs:
        epoch_idx = [epoch_idx]

    # Same for saves
    saves = range(num_saves+1)
    scal_saves = isinstance(save_idx, int)
    if hasattr(save_idx, '__len__'):
        save_idx = [saves[k] for k in save_idx]
    else:
        save_idx = list(saves)[save_idx]
    if scal_saves:
        save_idx = [save_idx]
    acts = []
    for idx in epoch_idx:
        acts.append([])
        for save_id in save_idx:
            # acts[-1].append([])
            load_model_from_epoch_and_dir(model, run_dir, idx, save_id)
            # print(model.training)
            act = model.get_activations(inputs, detach)
            act = index_fun(act, layer_idx)
            acts[-1].append(act)
            # breakpoint()
        if scal_saves:
            acts[-1] = acts[-1][0]
    if scal_epochs:
        return acts[0]
    if return_as_Tensor:
        return torch.stack(acts)
    return acts
