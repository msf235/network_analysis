import torch
import numpy
numpy.dot

def load_model(model, filename):
    """
    Load the given torch model from the given file.

    Args:
        model: (torch.nn.Module) the neural network module which should have its state loaded
        filename: (string) the file that contains the state of the model

    Returns:
        (torch.nn.Module, dict) the model passed in and the state_info loaded.
    """
    state_info = torch.load(filename)
    model.load_state_dict(state_info['state_dict'])
    return model, state_info

def get_max_epoch(out_dir):
    """
    Get the oldest saved epoch in the given directory.

    Args:
        out_dir: (string) the path to the directory

    Returns:
        (number) the oldest epoch that is saved in the specified directory.
    """
    out_dir = format_dir(out_dir)
    # i0 = 0
    # dir_exists = True
    max_look = 10000
    for i0 in range(max_look):
        filename = out_dir + 'check_{}'.format(i0)
        dir_exists = os.path.exists(filename)
        if not dir_exists:
            return i0 - 1

def load_model_from_epoch_and_dir(model, out_dir, epoch_num):
    """
    Loads the module as it was on a specific epoch.

    Args:
        model: (torch.nn.Module) the neural network model which should have its state loaded
        out_dir: (string) the path to the directory the model was saved in
        epoch_num: (number) the epoch to load, or -1 to load the max epoch.

    Returns:
        (torch.nn.Module, dict) the model and the loaded state information
    """
    out_dir = format_dir(out_dir)
    if epoch_num == -1:
        epoch_num = get_max_epoch(out_dir)
    filename = out_dir + 'check_{}'.format(epoch_num)
    state_info = torch.load(filename)
    model.load_state_dict(state_info['state_dict'])
    return model, state_info


def load_model_mom(model, epoch, arg_dict, table_path, run_name):
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
        (torch.nn.Module, dict) the loaded model and the state info
    """
    run_id, run_dir = mom.dir_for_run(arg_dict, table_path)
    if epoch == -1:
        epoch = get_max_epoch(run_dir)
    model, state_info = load_model_from_epoch_and_dir(model, run_dir, epoch)
    return model, state_info

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
        state_dict = state_info['state_dict']
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
            state_dict = state_info['state_dict']
            for key in keys:
                if w_scal:
                    w[key] = state_dict[key]
                else:
                    w[key][i0] = state_dict[key]

        return w

class activity_loader:
    def __init__(self, model, run_dir, X, layer_idx=None):
        """

        Args:
            model ():
            num_epochs (int): Number of epochs, not including the 0 epoch. Soo if num_epochs=2 then the epochs are
                [0,1,2].
            run_dir ():
            X ():
            layer_idx ():
        """
        self.run_dir = format_dir(run_dir)
        self.num_epochs = get_max_epoch(self.run_dir)
        self.model = model
        self.X = X
        self.layer_idx = layer_idx

    def __len__(self):
        return self.num_epochs + 1

    def __getitem__(self, idx):
        """
        Load over range of epochs.
        Args:
            idx (int, slice): Load over range of epochs designated by idx.

        Returns:

        """
        if isinstance(idx, tuple):
            idx = list(idx)
        n_idx = np.arange(self.num_epochs + 1)[idx]
        scal = False
        if not hasattr(n_idx, '__len__'):
            n_idx = [n_idx]
            scal = True
        model, state_info = load_model_from_epoch_and_dir(self.model, self.run_dir, n_idx[0])
        acts = model.get_activations(self.X)
        # acts_full = acts  # careful!
        acts_full = []
        # hid_sh = [0] * len(acts_full)
        for i0, act in enumerate(acts):
            if scal:
                acts_full.append(act.numpy().astype(float))
            else:
                acts_full.append(np.zeros((len(n_idx),) + act.shape))
                acts_full[i0][0] = act.numpy().astype(float)

        if scal:
            if self.layer_idx is not None:
                acts_full = acts_full[self.layer_idx]
            return acts_full

        for i0, el0 in enumerate(n_idx[1:]):
            model, state_info = load_model_from_epoch_and_dir(self.model, self.run_dir, el0)
            acts = model.get_activations(self.X)
            for i1, act in enumerate(acts):
                acts_full[i1][i0 + 1] = act.numpy().astype(float)
        if self.layer_idx is not None:
            acts_full = acts_full[self.layer_idx]
        return acts_full